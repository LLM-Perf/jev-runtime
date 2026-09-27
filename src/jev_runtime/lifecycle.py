from __future__ import annotations

import asyncio
from collections.abc import Iterable


async def cancel_and_drain(tasks: Iterable[asyncio.Task]) -> None:
    """Keep ownership through repeated parent cancellation; re-raise after drain.

    Scoring children have their own bounded abort deadline. Shielding only a
    backend abort is insufficient: every parent must also wait for those children
    before releasing a bundle lease or closing its transport.
    """
    owned = tuple(tasks)
    for task in owned:
        if not task.done() and not task.cancelling():
            task.cancel()
    waiter = asyncio.gather(*owned, return_exceptions=True)
    interrupted = False
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            interrupted = True
    waiter.result()
    if interrupted:
        raise asyncio.CancelledError
