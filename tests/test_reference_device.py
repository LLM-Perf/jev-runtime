"""Resource planning can be checked without importing optional Torch/CUDA."""

import pytest

from tests.integration.reference_device import cuda_budget


def test_reference_allocator_keeps_reserve_when_budget_exceeds_free_memory():
    mib = 1024**2
    assert cuda_budget(8000 * mib, 81559 * mib, 6144, 3072) == 4928 * mib
    assert cuda_budget(11990 * mib, 81559 * mib, 6144, 3072) == 6144 * mib


@pytest.mark.parametrize(
    "free,total,budget,reserve",
    [(0, 10, 6, 3072), (11, 10, 6, 3072), (10, 10, 0, 3072), (10, 10, 6, 3000)],
)
def test_reference_allocator_rejects_invalid_or_unreserved_memory(free, total, budget, reserve):
    with pytest.raises(ValueError):
        cuda_budget(free, total, budget, reserve)


def test_reference_allocator_rejects_exhausted_gpu():
    with pytest.raises(ValueError, match="Insufficient"):
        cuda_budget(3072 * 1024**2, 81559 * 1024**2, 6144, 3072)
