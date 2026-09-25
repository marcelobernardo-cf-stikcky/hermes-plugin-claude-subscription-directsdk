"""Request-level failover across isolated Claude Code config directories."""
from __future__ import annotations

import asyncio
import os
import re
import threading
import time


# Deliberately conservative: retry only errors that identify quota/rate limiting.
_QUOTA_ERROR = re.compile(
    r"(?:rate[_ -]?limit|too many requests|quota|usage limit|exhausted|"
    r"5[- ]?hour|7[- ]?day|weekly limit|rate_limit_error|http\s*429)",
    re.IGNORECASE,
)


def account_config_dirs(env=None):
    """Return configured account directories, preserving their literal order."""
    source = dict(os.environ)
    if env:
        source.update(env)
    raw = source.get("CLAUDE_SUBSCRIPTION_DIRECTSDK_CONFIG_DIRS", "")
    return tuple(part for part in raw.split(os.pathsep) if part)


def is_quota_error(error):
    return bool(_QUOTA_ERROR.search(str(error)))


class _SyncStream:
    def __init__(self, router, kwargs):
        self.router, self.kwargs = router, kwargs
        self.stream = None
        self.index = None
        self.emitted = False
        self._advance_lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._advance_lock:
            while True:
                if self.stream is None:
                    self.index = self.router._eligible()
                    if self.index is None:
                        raise RuntimeError("All Claude subscription accounts are in quota cooldown")
                    try:
                        self.index, self.stream = self.router._start(self.kwargs, index=self.index)
                    except Exception as error:
                        self._failed(error)
                        continue
                try:
                    item = next(self.stream)
                    self.emitted = True
                    return item
                except StopIteration:
                    self.close()
                    raise
                except Exception as error:
                    self._failed(error)

    def _failed(self, error):
        index = self.index
        self.close()
        if self.emitted or not is_quota_error(error):
            raise error
        self.router._mark_quota(index)
        self.index, self.stream = self.router._start(self.kwargs, exclude={index})

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _AsyncStream:
    def __init__(self, sync_stream):
        self.sync_stream = sync_stream

    def __aiter__(self):
        return self

    async def __anext__(self):
        present, item = await asyncio.to_thread(self._next)
        if not present:
            raise StopAsyncIteration
        return item

    def _next(self):
        try:
            return True, next(self.sync_stream)
        except StopIteration:
            return False, None

    async def aclose(self):
        await asyncio.to_thread(self.sync_stream.close)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()


class AccountRouter:
    """Route one request across clients bound to separate Claude config dirs."""

    def __init__(self, client_factory, client_kwargs, config_dirs, cooldown=300):
        self._client_factory = client_factory
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._cooldowns = [0.0] * len(config_dirs)
        self._clients = []
        for config_dir in config_dirs:
            env = dict(client_kwargs.get("env") or os.environ)
            env.pop("CLAUDE_SUBSCRIPTION_DIRECTSDK_CONFIG_DIRS", None)
            env.pop("CLAUDE_SUBSCRIPTION_DIRECTSDK_CONFIG_DIR", None)
            env["CLAUDE_CONFIG_DIR"] = config_dir
            kwargs = dict(client_kwargs)
            kwargs["env"] = env
            self._clients.append(client_factory(**kwargs))
        self.api_key = "external-process"
        self.base_url = "process://claude-subscription-directsdk-experimental"
        self.chat = type("Chat", (), {"completions": type("Completions", (), {"create": self.create})()})()

    def _eligible(self, exclude=()):
        now = time.monotonic()
        with self._lock:
            candidates = [
                i for i, until in enumerate(self._cooldowns)
                if i not in exclude and until <= now
            ]
            if not candidates:
                return None
            # Primary order is stable; the first account is preferred when healthy.
            return candidates[0]

    def _start(self, kwargs, exclude=(), index=None):
        if index is None:
            index = self._eligible(exclude)
        if index is None:
            raise RuntimeError("All Claude subscription accounts are in quota cooldown")
        request = dict(kwargs)
        request["stream"] = True
        return index, self._clients[index].chat.completions.create(**request)

    def _mark_quota(self, index):
        with self._lock:
            self._cooldowns[index] = time.monotonic() + self._cooldown

    def _create(self, kwargs):
        index = self._eligible()
        if index is None:
            raise RuntimeError("All Claude subscription accounts are in quota cooldown")
        try:
            return self._clients[index].chat.completions.create(**kwargs)
        except Exception as error:
            if not is_quota_error(error):
                raise
            self._mark_quota(index)
            fallback = self._eligible({index})
            if fallback is None:
                raise
            try:
                return self._clients[fallback].chat.completions.create(**kwargs)
            except Exception as fallback_error:
                if is_quota_error(fallback_error):
                    self._mark_quota(fallback)
                raise

    def create(self, **kwargs):
        if kwargs.get("stream"):
            stream = _SyncStream(self, kwargs)
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return stream
            return _AsyncStream(stream)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._create(kwargs)
        return self._acreate(kwargs)

    async def _acreate(self, kwargs):
        return await asyncio.to_thread(self._create, kwargs)

    def cancel(self):
        for client in self._clients:
            client.cancel()

    def close(self):
        for client in self._clients:
            client.close()
