"""
Test: pvm.py's PortfolioVectorMemory -- gather/scatter read/write between
global space and slot space, all-cash init, and boundary liquidation of a
departing ticker (docs/EIIE_AGENT_PLAN.md Phase 3). Synthetic data only.

Run from project root:
    python tests/rl_agent/test_pvm.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent.data import CASH_GIDX  # noqa: E402
from src.rl_agent.pvm import PortfolioVectorMemory  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_init(passed, failed):
    n_global = 4  # cash + 3 union tickers
    pvm = PortfolioVectorMemory(T=5, n_global=n_global)

    row0 = pvm.read_global(torch.tensor([0]))
    ok = torch.allclose(row0, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    print_check("init: every row starts all-cash (eq. 5)", ok, str(row0.tolist()))
    passed, failed = passed + ok, failed + (not ok)

    all_rows = pvm.read_global(torch.arange(5))
    ok = bool(torch.allclose(all_rows[:, CASH_GIDX], torch.ones(5)))
    print_check("init: cash column is 1.0 across all rows", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_read_write_roundtrip(passed, failed):
    n_global = 4
    pvm = PortfolioVectorMemory(T=3, n_global=n_global)

    slot_gidx = torch.tensor([[1, 2]])  # 1 batch row, 2 slots -> global assets 1 and 2
    w = torch.tensor([[0.5, 0.3, 0.2]])  # cash=0.5, slot0=0.3, slot1=0.2
    pvm.write(torch.tensor([1]), slot_gidx, w)

    ok = torch.allclose(pvm.read_global(torch.tensor([1])), torch.tensor([[0.5, 0.3, 0.2, 0.0]]))
    print_check("write: scatters slots into the correct global columns, cash set directly", ok,
                str(pvm.read_global(torch.tensor([1])).tolist()))
    passed, failed = passed + ok, failed + (not ok)

    valid = torch.tensor([[True, True]])
    recovered = pvm.read_slots(torch.tensor([1]), slot_gidx, valid)
    ok = torch.allclose(recovered, w)
    print_check("read_slots: recovers exactly what write() stored, same slot layout", ok, str(recovered.tolist()))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_new_entrant_reads_zero(passed, failed):
    """A ticker with no prior weight (just entered the universe) must read
    0 from read_slots, not garbage -- gather from an untouched (all-zero,
    non-cash) global column naturally gives 0."""
    n_global = 4
    pvm = PortfolioVectorMemory(T=3, n_global=n_global)
    slot_gidx = torch.tensor([[3, 1]])  # global asset 3 has never been written to
    valid = torch.tensor([[True, True]])
    w_prev = pvm.read_slots(torch.tensor([0]), slot_gidx, valid)
    ok = torch.allclose(w_prev, torch.tensor([[1.0, 0.0, 0.0]]))  # cash=1 (init), both slots=0
    print_check("read_slots: a never-held asset reads exactly 0, not garbage", ok, str(w_prev.tolist()))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_boundary_liquidation(passed, failed):
    """Ticker X (global idx 3) is held with weight 0.4 at t-1, then drops out
    of the universe at t (its slot is no longer in slot_gidx_t at all). Its
    global weight must still be readable (raw) at t-1 for the cost solver's
    drift computation, and must be exactly 0 in the fresh row written at t
    (docs/EIIE_AGENT_PLAN.md "Boundary liquidation happens in global space")."""
    n_global = 5  # cash + 4 union tickers, ticker X = global idx 3
    pvm = PortfolioVectorMemory(T=3, n_global=n_global)

    slot_gidx_prev = torch.tensor([[1, 3]])  # X (3) held at t-1
    w_prev = torch.tensor([[0.1, 0.5, 0.4]])  # cash=0.1, slot0(asset1)=0.5, slot1(X)=0.4
    pvm.write(torch.tensor([0]), slot_gidx_prev, w_prev)

    global_prev = pvm.read_global(torch.tensor([0]))
    ok = torch.isclose(global_prev[0, 3], torch.tensor(0.4))
    print_check("boundary: X's real prior weight is still in the raw global row", ok, str(global_prev.tolist()))
    passed, failed = passed + ok, failed + (not ok)

    # X drops out at t=1 -- today's universe is {asset1, asset2}, not X
    slot_gidx_t = torch.tensor([[1, 2]])
    w_t = torch.tensor([[0.2, 0.5, 0.3]])  # cash=0.2, asset1=0.5, asset2=0.3 -- X gets no weight at all
    pvm.write(torch.tensor([1]), slot_gidx_t, w_t)

    global_t = pvm.read_global(torch.tensor([1]))
    ok = torch.isclose(global_t[0, 3], torch.tensor(0.0))
    print_check("boundary: X's global weight is exactly 0 in the fresh row (liquidated target)",
                ok, str(global_t.tolist()))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_padding_sentinel_safety(passed, failed):
    """A padding slot's dummy sentinel index (== n_global) must never let a
    stray nonzero weight corrupt the cash column, even if masking upstream
    somehow failed to force it to exact 0 -- the whole reason the sentinel
    is n_global and not 0/cash (data.py's _build_slot_calendar)."""
    n_global = 3  # cash + 2 union tickers
    pvm = PortfolioVectorMemory(T=2, n_global=n_global)

    slot_gidx = torch.tensor([[1, n_global]])  # slot 1 is padding -> dummy index
    adversarial_w = torch.tensor([[0.6, 0.4, 0.9]])  # cash=0.6, real slot=0.4, padding slot=0.9 (should NOT be 0.9 but test the blast radius anyway)
    pvm.write(torch.tensor([0]), slot_gidx, adversarial_w)

    row = pvm.read_global(torch.tensor([0]))
    ok = torch.isclose(row[0, CASH_GIDX], torch.tensor(0.6))
    print_check("padding sentinel: cash column unaffected by a stray padding-slot weight", ok, str(row.tolist()))
    passed, failed = passed + ok, failed + (not ok)

    ok = torch.isclose(row[0, 1], torch.tensor(0.4))
    print_check("padding sentinel: the real asset's column unaffected", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_batched(passed, failed):
    n_global = 4
    pvm = PortfolioVectorMemory(T=4, n_global=n_global)
    row_idx = torch.tensor([0, 1, 2])
    slot_gidx = torch.tensor([[1, 2], [1, 2], [1, 2]])
    w = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.1, 0.1, 0.8]])
    pvm.write(row_idx, slot_gidx, w)

    ok = torch.allclose(pvm.read_global(row_idx)[:, :3], w)
    print_check("batched write/read_global: independent rows for a batch of period indices", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_pvm")
    passed = failed = 0

    passed, failed = test_init(passed, failed)
    passed, failed = test_read_write_roundtrip(passed, failed)
    passed, failed = test_new_entrant_reads_zero(passed, failed)
    passed, failed = test_boundary_liquidation(passed, failed)
    passed, failed = test_padding_sentinel_safety(passed, failed)
    passed, failed = test_batched(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
