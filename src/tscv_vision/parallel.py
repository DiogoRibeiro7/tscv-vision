"""Parallel and distributed utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_parallel(
    func: Callable[[T], R], data: Iterable[T], workers: int | None = None
) -> list[R]:
    """Apply ``func`` to ``data`` in parallel.

    Parameters
    ----------
    func:
        Callable executed on each element of ``data``.
    data:
        Iterable of inputs.
    workers:
        Number of worker processes. If ``None`` or ``1`` the call is executed
        sequentially.

    Returns
    -------
    list[R]
        Results in the same order as ``data``.
    """

    if workers in (None, 1):
        return [func(d) for d in data]
    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        res = pool.map(func, data)
    return list(res)


def map_dask(func: Callable[[T], R], data: Iterable[T], workers: int | None = None) -> list[R]:
    """Apply ``func`` to ``data`` using Dask.

    Requires :mod:`dask` to be installed.
    """

    try:
        import dask
        from dask import delayed  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - import error
        raise ImportError("dask is required for map_dask") from exc

    delayed_tasks = [delayed(func)(d) for d in data]
    scheduler = "processes" if workers not in (None, 1) else "single-threaded"
    results = dask.compute(  # type: ignore[attr-defined, no-untyped-call]
        *delayed_tasks, scheduler=scheduler, num_workers=workers
    )
    return list(results)


__all__ = ["map_parallel", "map_dask"]
