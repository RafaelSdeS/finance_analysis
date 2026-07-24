"""
risk.py -- the risk model Σ (proposal §2, "The risk model Σ -- do not skip
this"): a sample covariance of 30-500 assets over overlapping windows is
numerically garbage (ill-conditioned, sometimes singular when the trailing
window is shorter than the asset count) -- Ledoit-Wolf shrinkage toward a
scaled identity is not optional here, it's what keeps the optimizer (§2.4)
solvable and its output attributable to α rather than to Σ noise.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def shrinkage_cov(returns_window: pd.DataFrame) -> pd.DataFrame:
    """returns_window: DataFrame[date, ticker] of daily simple returns (wide,
    one column per ticker). Tickers with any NaN in the window are dropped
    (LedoitWolf needs a complete matrix) -- returns a DataFrame indexed and
    columned by the surviving tickers, same order."""
    clean = returns_window.dropna(axis=1, how="any")
    if clean.shape[1] == 0:
        return pd.DataFrame(index=[], columns=[])
    lw = LedoitWolf().fit(clean.to_numpy())
    return pd.DataFrame(lw.covariance_, index=clean.columns, columns=clean.columns)


def add_cash_row_col(sigma: pd.DataFrame, cash_key: str = "cash") -> pd.DataFrame:
    """Append a zero-variance, zero-covariance cash row/col (proposal §5:
    "Σ row/col for cash ≈ 0" -- cash's return is the known CDI carry, not a
    random variable the optimizer needs to hedge against)."""
    sigma = sigma.copy()
    sigma[cash_key] = 0.0
    sigma.loc[cash_key] = 0.0
    return sigma


def condition_number(sigma: pd.DataFrame) -> float:
    """np.linalg.cond of the raw matrix. Don't call this on a cash-augmented
    Σ (add_cash_row_col makes it deliberately singular; cond() there is
    meaningless) -- check conditioning before adding the cash row/col."""
    return float(np.linalg.cond(sigma.to_numpy()))


def is_psd(sigma: pd.DataFrame, tol: float = 1e-8) -> bool:
    """Positive semi-definite check via eigenvalues (a small negative
    tolerance absorbs floating-point noise around exact-zero eigenvalues)."""
    eigvals = np.linalg.eigvalsh(sigma.to_numpy())
    return bool(np.all(eigvals >= -tol))
