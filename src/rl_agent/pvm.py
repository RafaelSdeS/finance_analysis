"""
pvm.py — Portfolio-Vector Memory (paper Sec. 5.2), extended for a dynamic
top-50 universe (docs/eiie_agent/EIIE_AGENT_PLAN.md "PVM, OSBL training, split
protocol" and "Global asset indexing" sections).

The buffer stores each period's full GLOBAL portfolio-weight vector (cash +
171 union tickers), not the network's 50-wide slot layout -- this is what
lets environment.py compute a departing ticker's forced-sale cost correctly
even on the one day it's no longer among the network's input slots: its
last real weight is still sitting in the previous global row.

read_slots()/write() are the seam between the two coordinate systems:
  - read_slots: gather today's active-slot weights out of YESTERDAY's global
    row (today's just-departed/just-entered tickers naturally read 0, since
    they held no weight in the global vector's corresponding column yet).
  - write: scatter the network's slotted output back into TODAY's global
    row. A departing ticker isn't scattered into, so its column is 0 in the
    fresh row -- exactly "target weight 0", liquidated by environment.py's
    cost solver.

Vectorized via torch.gather/scatter per the approved design constraint; no
per-day Python loops.
"""

import torch

from .data import CASH_GIDX


def scatter_to_global_row(slot_gidx: torch.Tensor, w: torch.Tensor, n_global: int) -> torch.Tensor:
    """Slot-space -> (n_global+1)-wide global-space row (w: [B, n_slots+1],
    column 0 = cash). The dummy column at index n_global is always a safe
    scatter target for a padding slot. Shared by PortfolioVectorMemory.write()
    (which stores the result, detached) and train.py (which keeps it
    differentiable for the loss, then detaches before persisting) -- one
    scatter implementation, so the two can't drift apart.
    """
    B = w.shape[0]
    row = torch.zeros(B, n_global + 1, dtype=w.dtype, device=w.device)
    row[:, CASH_GIDX] = w[:, 0]
    row.scatter_(1, slot_gidx, w[:, 1:])
    return row


class PortfolioVectorMemory:
    def __init__(self, T: int, n_global: int, slot_gidx=None, valid=None, device="cpu", dtype=torch.float32):
        """n_global: real global-space width (cash + N_union tickers, e.g.
        172). The buffer allocates one extra dummy column (index n_global)
        as a safe scatter target for padding slots (data.py's dummy
        sentinel) -- never read back, so it can never leak into the cash or
        a real asset's weight.

        Initialization (paper p.14: "uniform weights"):
        - If slot_gidx/valid provided: each row = uniform over cash + that day's active slots.
          Matching paper's stated behavior (uniform over investable assets).
        - If not provided: all-cash fallback (backward compat; used by diagnose.py when
          loading a frozen checkpoint without reconstructing the full panel).
        """
        self.n_global = n_global
        self.buffer = torch.zeros(T, n_global + 1, dtype=dtype, device=device)

        if slot_gidx is not None and valid is not None:
            # Initialize each row uniform-over-active: compute count per row,
            # allocate uniform weight, scatter into global space.
            slot_gidx_t = torch.as_tensor(slot_gidx, dtype=torch.long, device=device)  # [T, n_slots]
            valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)          # [T, n_slots]

            # Number of active slots per row
            n_active = valid_t.sum(dim=1, dtype=dtype)  # [T]
            # Uniform weight = 1 / (n_active + 1)  for each asset (cash gets same)
            uniform_weight = 1.0 / (n_active + 1)  # [T]

            # Asset slots [T, n_slots]: active slots get uniform_weight, invalid 0
            w_slots = valid_t.to(dtype) * uniform_weight.unsqueeze(1)

            # Scatter to global space via scatter_to_global_row
            # scatter_to_global_row expects w: [T, n_slots+1] with w[:, 0] = cash
            w_slotted = torch.cat([uniform_weight.unsqueeze(1), w_slots], dim=1)
            self.buffer = scatter_to_global_row(slot_gidx_t, w_slotted, n_global)
        else:
            # Fallback: all-cash (backward compat for diagnose.py or tests)
            self.buffer[:, CASH_GIDX] = 1.0

    def read_global(self, row_idx: torch.Tensor) -> torch.Tensor:
        """Full previous-period global weight vector(s), shape [..., n_global]
        (dummy column dropped). Used by environment.py's cost/reward math,
        which needs a departing ticker's actual prior weight even though
        it's no longer a network input slot that day."""
        return self.buffer[row_idx, : self.n_global]

    def read_slots(self, row_idx: torch.Tensor, slot_gidx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Previous-period weight in slot space + cash, for network input.
        slot_gidx/valid describe TODAY's active slots; gathering yesterday's
        row at today's positions is well-defined even for a ticker that just
        entered the universe (yesterday's weight there is 0, since it wasn't
        tradable yet). Returns [B, n_slots + 1], column 0 = cash.
        """
        rows = self.buffer[row_idx]  # [B, n_global + 1]
        w_cash = rows[:, CASH_GIDX : CASH_GIDX + 1]
        w_slots = torch.gather(rows, 1, slot_gidx) * valid.to(rows.dtype)
        return torch.cat([w_cash, w_slots], dim=1)

    def write(self, row_idx: torch.Tensor, slot_gidx: torch.Tensor, w: torch.Tensor) -> None:
        """Store the network's output into PVM[row_idx] in global space.
        w: [B, n_slots + 1], column 0 = cash, columns 1.. = the slotted
        output (already exactly 0 on padding/masked slots, from the -inf
        logit mask before softmax). A fresh zero row each call means any
        ticker absent from slot_gidx this period is 0 in the new row --
        the liquidation isn't an extra step, it falls out of not writing
        to that column.
        """
        self.buffer[row_idx] = scatter_to_global_row(slot_gidx, w, self.n_global)
