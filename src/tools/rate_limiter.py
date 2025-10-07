"""速率限制器 - 支持 Token Bucket 算法的异步速率控制组件。"""

import asyncio
import time
from typing import ClassVar


class RateLimiter:
    """
    Token Bucket 速率限制器。

    使用令牌桶算法平滑控制请求速率，支持异步操作和多实例共享。
    """

    def __init__(
        self,
        rate: float = 1.0,
        capacity: int | None = None,
        initial_tokens: int | None = None,
    ):
        """
        初始化速率限制器。

        Args:
            rate: 每秒生成的令牌数（默认1.0，即每秒1个请求）
            capacity: 桶容量（默认等于rate，允许短时突发）
            initial_tokens: 初始令牌数（默认等于capacity）
        """
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1, int(rate))
        self.tokens = initial_tokens if initial_tokens is not None else self.capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """
        获取令牌，如果令牌不足会等待。

        Args:
            tokens: 需要的令牌数（默认1）
            timeout: 最大等待时间（秒），None表示无限等待

        Returns:
            是否成功获取令牌
        """
        start_time = time.time()

        while True:
            async with self._lock:
                # 更新令牌桶
                self._refill()

                # 检查是否有足够的令牌
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                # 计算需要等待的时间
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate

            # 检查超时
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)

            # 等待令牌生成
            await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        """根据时间流逝补充令牌。"""
        now = time.time()
        elapsed = now - self.last_update

        # 计算应该生成的令牌数
        new_tokens = elapsed * self.rate

        # 更新令牌数（不超过容量）
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        尝试立即获取令牌，不等待。

        Args:
            tokens: 需要的令牌数

        Returns:
            是否成功获取令牌
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_available_tokens(self) -> float:
        """获取当前可用的令牌数。"""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.rate
        return min(self.capacity, self.tokens + new_tokens)

    async def wait_for_token(self) -> None:
        """等待直到至少有一个令牌可用。"""
        await self.acquire(tokens=1)


class GlobalRateLimiter:
    """
    全局速率限制器（类级别共享）。

    用于跨多个实例共享同一个速率限制器，适合控制外部API调用。
    """

    _instances: ClassVar[dict[str, RateLimiter]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def get_limiter(
        cls,
        name: str,
        rate: float = 1.0,
        capacity: int | None = None,
    ) -> RateLimiter:
        """
        获取或创建命名的速率限制器。

        Args:
            name: 限制器名称
            rate: 每秒请求数
            capacity: 桶容量

        Returns:
            速率限制器实例
        """
        async with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = RateLimiter(rate=rate, capacity=capacity)
            return cls._instances[name]

    @classmethod
    async def acquire(
        cls,
        name: str,
        tokens: int = 1,
        timeout: float | None = None,
    ) -> bool:
        """
        从命名限制器获取令牌。

        Args:
            name: 限制器名称
            tokens: 需要的令牌数
            timeout: 最大等待时间

        Returns:
            是否成功获取令牌
        """
        limiter = await cls.get_limiter(name)
        return await limiter.acquire(tokens, timeout)


class ExponentialBackoff:
    """指数退避策略，用于智能重试。"""

    def __init__(
        self,
        base: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True,
    ):
        """
        初始化指数退避策略。

        Args:
            base: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            multiplier: 延迟倍增因子
            jitter: 是否添加随机抖动
        """
        self.base = base
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempt = 0

    def get_delay(self) -> float:
        """计算当前重试的延迟时间。"""
        delay = min(self.base * (self.multiplier**self.attempt), self.max_delay)

        if self.jitter:
            import random

            delay = delay * (0.5 + random.random() * 0.5)

        self.attempt += 1
        return delay

    def reset(self) -> None:
        """重置重试计数器。"""
        self.attempt = 0


__all__ = ["ExponentialBackoff", "GlobalRateLimiter", "RateLimiter"]
