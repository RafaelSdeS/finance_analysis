"""
walkforward.py -- Phase 0 purge/embargo wrapper over h_series' expanding-fold
generator (docs/conviction_model/CONVICTION_MODEL_PLAN.md, Phase 0/3/5).

Reuses iter_expanding_folds/FoldWindow directly rather than reimplementing
fold logic (plan's "reuse, don't duplicate" table) -- the only new logic here
is the purge/embargo filter: a training row is only safe to use once its
label's forward-looking window has fully resolved by the fold's train_end,
otherwise a walk-forward "OOS" evaluation would secretly be trained on rows
that peek past the cutoff being tested.

Purge/embargo lives ONLY here, not duplicated in labels.py -- labels.py
produces the raw, unfiltered label table; this module is the single place
that decides which rows are safe to train on as of a given date.
"""

import pandas as pd

from ..h_series.spine import FoldWindow, iter_expanding_folds, k_trading_days_later  # noqa: F401

MAX_LABEL_HORIZON = 504  # trading days -- the longest horizon in labels.py; sets the purge rule


def purge_embargo_mask(decision_dates, calendar: pd.DatetimeIndex, train_end,
                        max_k: int = MAX_LABEL_HORIZON):
    """True where a row's label window (decision_date + max_k trading days)
    fully resolves at or before `train_end` -- safe to use for training as of
    that cutoff. False where the label window extends past `train_end` (would
    leak future information relative to the cutoff being tested) OR runs past
    the end of `calendar` entirely (the label isn't computable at all,
    k_trading_days_later returns NaT for those rows -- excluded, not
    fabricated). Positional: returned mask aligns 1:1 with `decision_dates`
    in the order given, same convention `k_trading_days_later` already uses."""
    decision_dates = pd.DatetimeIndex(decision_dates)
    label_end = k_trading_days_later(calendar, decision_dates, max_k)
    train_end = pd.Timestamp(train_end)
    return label_end.notna() & (label_end <= train_end)


def iter_purged_folds(labels_df: pd.DataFrame, calendar: pd.DatetimeIndex, window_end,
                       initial_train_end, step_months: int = 3, max_k: int = MAX_LABEL_HORIZON):
    """Yields (FoldWindow, train_df, oos_df) for each expanding fold:
    - train_df: purge/embargo-filtered rows of `labels_df` whose label window
      fully resolves by fold.train_end (see purge_embargo_mask).
    - oos_df: every row of `labels_df` decided in (fold.train_end, fold.oos_end].
      Not purge-filtered -- an OOS row's own label is exactly what's being
      scored, not used to fit anything, so there's nothing to leak.

    `labels_df` must have a `decision_date` column (as produced by
    build_conviction_labels). Reuses iter_expanding_folds directly; only the
    purge/embargo split on top of each fold is new."""
    decision_dates = pd.DatetimeIndex(labels_df["decision_date"])

    for fold in iter_expanding_folds(window_end, initial_train_end, step_months):
        # purge_embargo_mask/DatetimeIndex comparisons return plain numpy bool
        # arrays (not pandas Series), already positionally aligned with
        # labels_df -- .loc[] takes them directly, no .to_numpy() needed.
        train_mask = purge_embargo_mask(decision_dates, calendar, fold.train_end, max_k)
        oos_mask = (decision_dates > fold.train_end) & (decision_dates <= fold.oos_end)

        yield fold, labels_df.loc[train_mask], labels_df.loc[oos_mask]
