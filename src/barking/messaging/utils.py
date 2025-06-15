import asyncio
from collections.abc import Callable, Coroutine


def is_async_callable(fn: Callable) -> bool:
    """Test if the callable is asynchronous.

    Args:
        fn: The callable to be tested.
    Returns:
        bool: True if the callable is asynchronous.
    """
    from inspect import iscoroutinefunction

    if hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return iscoroutinefunction(fn)


async def gather_with_concurrency(limit: int, *fns: Coroutine) -> list:
    semaphore = asyncio.Semaphore(limit)

    async def limited_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(limited_coro(fn) for fn in fns))
