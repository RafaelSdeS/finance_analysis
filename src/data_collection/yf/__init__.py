"""yf/ — yfinance-sourced collectors, split by concern (docs/DATA_LAYER_ORGANIZATION_PLAN.md §O3):
_common (shared fetch/repair helpers), prices, fundamentals, dividends (+ splits).

No re-exports here on purpose -- import from the specific submodule you need
(e.g. `from .yf import prices as yf_prices`). A package-level shim was
considered and rejected: nothing outside this package needs the pre-split
`yf_collectors` names, so forwarding them here would only be a second layer
over the real code.
"""
