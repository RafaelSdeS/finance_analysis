# Data Consistency Verification Guide

## Overview

`verify_data_consistency.py` is a comprehensive data validation script that ensures your financial data is consistent with real-world constraints. It validates:

- **Price Data** (OHLC, volume, dates)
- **Macro Indicators** (SELIC, IPCA, CDI rates)
- **Fundamental Data** (financial metrics, ratios)
- **Cross-Asset Consistency** (date alignment, data integrity)

## Features

### 1. Price Data Validation
- ✅ OHLC constraints (High ≥ Low, Open/Close within bounds)
- ✅ Price positivity (no negative or zero prices)
- ✅ Unrealistic jumps detection (>50% in one day)
- ✅ Date continuity (detects large gaps in trading)
- ✅ Volume consistency (no negative volumes)

### 2. Macro Data Validation
- ✅ SELIC rate range (0%-50% realistic range)
- ✅ IPCA inflation range (-10% to +30%)
- ✅ CDI/SELIC tracking (should be very similar)
- ✅ No future dates in data
- ✅ Abrupt rate changes detection

### 3. Fundamental Data Validation
- ✅ Financial metrics non-negativity (revenue, earnings)
- ✅ Financial ratio ranges (P/E, P/B, ROE, Dividend Yield)
- ✅ Data completeness percentage
- ✅ Extreme value detection

### 4. Cross-Asset Validation
- ✅ Date alignment (prices cover all macro dates)
- ✅ Data coverage analysis
- ✅ Orphaned data detection

## Installation

No additional dependencies needed - uses standard project requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Validation (Default)
```bash
cd src
python verify_data_consistency.py
```

### Detailed Output
```bash
python verify_data_consistency.py --detailed
```

Shows detailed findings for each validation check:
```
✓ PETR4
  Checks: 5/5 passed
    ✓ OHLC Constraints: All OHLC constraints satisfied
    ✓ Price Positivity: All prices are positive
    ...
```

### Save Report to JSON
```bash
python verify_data_consistency.py --output validation_report.json
```

Generates a structured JSON report with all findings:
```json
{
  "timestamp": "2026-05-19T22:38:37.775402",
  "overall_status": "pass",
  "tickers": [
    {
      "ticker": "PETR4",
      "checks": [...]
    }
  ],
  "macro_checks": [...],
  "cross_asset_checks": [...]
}
```

### Validate Specific Ticker
```bash
python verify_data_consistency.py --ticker PETR4
```

### Skip Certain Validations
```bash
# Skip price validation (faster for macro/fund checks)
python verify_data_consistency.py --skip-prices

# Skip macro validation
python verify_data_consistency.py --skip-macro

# Skip fundamental validation
python verify_data_consistency.py --skip-fund

# Combine multiple skips
python verify_data_consistency.py --skip-prices --skip-macro
```

### Custom Data Directory
```bash
python verify_data_consistency.py --data-dir /path/to/data
```

## Output Format

### Console Output

```
================================================================================
DATA CONSISTENCY VERIFICATION REPORT
================================================================================

Timestamp: 2026-05-19T22:38:37.775402
Data Directory: /home/rafael/Documents/finance_analysis/data/raw

✓ Overall Status: PASS

--------------------------------------------------------------------------------
PRICE & FUNDAMENTAL DATA
--------------------------------------------------------------------------------

✓ PETR4
  Checks: 5/5 passed
    ✓ OHLC Constraints: All OHLC constraints satisfied
    ✓ Price Positivity: All prices are positive
    ✓ Price Jumps: No unrealistic price jumps detected
    ✓ Date Continuity: Date range: 2010-03-25 to 2024-05-17
    ✓ Volume Consistency: Volume data is consistent

✓ WEGE3
  Checks: 2/3 passed
  Warnings: 1
    ✓ Non-Negative Metrics: Financial metrics are reasonable
    ⚠ Data Completeness: 1 columns >50% null
    ✓ Ratio Ranges: All financial ratios in realistic ranges
```

### Status Symbols

- ✓ = **PASS** - Validation successful
- ⚠ = **WARNING** - Validation found issues but not critical
- ✗ = **FAIL** - Validation failed, data quality issue

### Status Levels

- **pass** - All checks passed
- **warning** - Issues detected but not blocking (e.g., data gaps)
- **fail** - Data quality issue that may affect analysis

## Realistic Ranges Reference

### SELIC Rate (Brazilian Overnight Rate)
- **Normal range**: 2% - 20%
- **Extreme range**: 0% - 50%
- **Real scenario**: Brazil had 25% SELIC during 2022-2023 crisis

### IPCA (Brazilian Inflation)
- **Normal range**: -5% to +15% monthly
- **Extreme range**: -10% to +30%
- **Unusual**: More than 50% deflation months

### P/E Ratio (Price/Earnings)
- **Normal range**: -100 to +1000
- **Realistic**: Usually 5-30 for mature companies
- **Growth**: 30-100+ for high-growth companies
- **Negative**: Loss-making companies

### P/B Ratio (Price/Book)
- **Normal range**: 0.1 to 50
- **Realistic**: Usually 0.5-4 for equities
- **Extreme**: >10 for highly valued companies

### ROE (Return on Equity)
- **Normal range**: -100% to +100%
- **Mature**: Usually 5%-25%
- **High-quality**: >15%

### Dividend Yield
- **Normal range**: 0% to 100%
- **Realistic**: Usually 0% to 15%
- **High yield**: >8% (requires investigation)

## Interpreting Common Warnings

### "Date Alignment: ipca doesn't fully cover price data"
- **Cause**: Macro data (IPCA) ends before price data
- **Risk**: Cannot calculate inflation-adjusted returns for recent dates
- **Action**: Check if IPCA data is up-to-date; may need to fetch latest data

### "Data Completeness: X columns >50% null"
- **Cause**: Fundamental data has many missing values
- **Risk**: Some financial metrics unavailable for model training
- **Action**: Check which columns are missing; may indicate data collection issues

### "Price Jumps: X jumps >50% (max: Y%)"
- **Cause**: Large price movement in single day
- **Risk**: May indicate corporate actions (splits, dividends) or data errors
- **Action**: Verify dates; check if there were stock splits or dividend distributions

### "No realistic price jumps detected"
- **Status**: Expected for stable equities
- **Note**: Brazilian stocks can have volatile days during market crises

## Validation Process

### Step-by-Step Execution

1. **Load Data**
   - Reads parquet files from: `data/raw/prices/`, `data/raw/macro/`, `data/raw/fundamentals/`
   - Handles both indexed and columnar date storage

2. **Price Validation**
   - For each ticker: validates OHLC, prices, volume, dates

3. **Macro Validation**
   - Validates SELIC, IPCA, CDI ranges and tracking
   - Checks for future dates and abrupt changes

4. **Fundamental Validation**
   - For each ticker: validates financial metrics and ratios

5. **Cross-Asset Checks**
   - Verifies date alignment across all data sources
   - Checks CDI/SELIC tracking consistency

6. **Report Generation**
   - Aggregates all results
   - Determines overall status
   - Outputs to console and/or JSON

## Integration with ML Pipeline

### Before Building ML Dataset
```python
from src.verify_data_consistency import DataConsistencyVerifier

# Validate data before processing
verifier = DataConsistencyVerifier()
report = verifier.run_full_validation()

if report.overall_status() != 'pass':
    print("⚠️  Data validation warnings detected")
    # Handle warnings or proceed with caution
```

### Continuous Validation
```bash
# Run as part of your data pipeline
python src/verify_data_consistency.py --output data/validation/latest_report.json
```

### Automated Checks
```bash
# Use in CI/CD to catch data quality regressions
if python src/verify_data_consistency.py > /dev/null; then
    echo "✓ Data quality passed"
else
    echo "✗ Data quality check failed"
    exit 1
fi
```

## Exit Codes

- `0` = All validations passed
- `1` = At least one validation failed

Use this for automated pipelines:
```bash
python src/verify_data_consistency.py
if [ $? -eq 0 ]; then
    python src/build_ml_dataset.py
fi
```

## Performance

- **Quick run** (no details): ~1 second
- **With details**: ~2 seconds
- **With JSON output**: ~3 seconds

### Optimization Tips
```bash
# Skip expensive checks if running frequently
python verify_data_consistency.py --skip-prices  # ~0.5s

# Validate single ticker only
python verify_data_consistency.py --ticker PETR4  # ~0.5s
```

## Troubleshooting

### No data found
```
Error: data/raw/prices directory not found
```
**Solution**: Ensure data files exist in `data/raw/prices/`, `data/raw/macro/`, etc.

### KeyError: 'Column not found'
```
KeyError: 'reference_date'
```
**Solution**: Check that macro data files have expected columns (selic, ipca, cdi)

### SELIC rate shows 0.04%
**Note**: This is normal - SELIC is displayed as decimal (0.04 = 4%)

### All price dates show 1970-01-01
**Cause**: Date column loading issue
**Solution**: Check that price data files have Date index or Date column

## Examples

### Validate All Data with Details
```bash
python src/verify_data_consistency.py --detailed --output report.json
```

### Weekly Data Quality Check
```bash
#!/bin/bash
REPORT="data/validation/$(date +%Y%m%d).json"
python src/verify_data_consistency.py --output $REPORT

# Alert if issues found
if grep -q '"failed": 0' $REPORT; then
    echo "✓ Data quality OK"
else
    echo "✗ Data quality issues detected"
    cat $REPORT
fi
```

### Pre-Training Validation
```bash
#!/bin/bash
echo "Running pre-training data validation..."
python src/verify_data_consistency.py --detailed

if [ $? -ne 0 ]; then
    echo "⚠️  Data quality warnings detected. Review before training."
    read -p "Continue with training? (y/n) " -n 1 -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

python src/build_ml_dataset.py
```

## Technical Details

### Data Loading

The script intelligently handles different data formats:

```python
# Price data - expects index to be Date or column named 'Date'
df = pd.read_parquet('data/raw/prices/PETR4.parquet')
# Columns: Open, High, Low, Close, Adj Close, Volume

# Macro data - expects columns: reference_date, [selic|ipca|cdi]
df = pd.read_parquet('data/raw/macro/selic.parquet')
# Columns: reference_date, selic

# Fundamental data - expects various financial columns
df = pd.read_parquet('data/raw/fundamentals/PETR4.parquet')
# Columns: Date, PE, PB, ROE, DividendYield, Revenue, NetIncome, ...
```

### Validation Thresholds

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| Price jump | >50% | Brazilian stocks rarely move >50% in one day |
| Date gap | >30 days | Unusual for continuously traded stocks |
| SELIC range | 0% - 50% | Brazil's SELIC rarely exceeds these |
| IPCA range | -10% to +30% | Inflation rarely exceeds these |
| Volume nulls | >20% | More than 20% zeros is suspicious |
| Data completeness | >50% nulls | Columns with >50% nulls flagged |

## Contributing

To add new validation checks:

1. Create a new method in appropriate `*Validator` class
2. Return `ValidationCheck` with name, status, and message
3. Call method in corresponding `validate_*` function
4. Test with sample data

Example:
```python
@staticmethod
def validate_my_check(df: pd.DataFrame) -> ValidationCheck:
    """Check my custom condition."""
    if condition_fails:
        return ValidationCheck(
            name='My Check',
            status='fail',
            message='Description of what failed',
            details={'key': 'value'}
        )
    return ValidationCheck(
        name='My Check',
        status='pass',
        message='Check passed'
    )
```

## License

Part of the finance_analysis project. See LICENSE file.
