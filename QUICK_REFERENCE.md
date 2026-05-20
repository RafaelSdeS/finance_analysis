# Quick Reference: Data Consistency Verification

## TL;DR

```bash
# Quick check - takes ~1 second
cd src && python verify_data_consistency.py

# Detailed check - shows every finding
python verify_data_consistency.py --detailed

# Save report for automation
python verify_data_consistency.py --output report.json

# Monitor continuously
bash monitor_data_quality.sh --watch
```

## What It Does

Verifies your financial data is realistic and consistent:
- ✅ Prices: OHLC valid, no negative, no extreme jumps
- ✅ Rates: SELIC, IPCA, CDI in realistic ranges
- ✅ Fundamentals: Metrics, ratios, data completeness
- ✅ Cross-checks: Date alignment, data integrity

## Status Symbols

| Symbol | Meaning |
|--------|---------|
| ✓ | **PASS** - Check succeeded |
| ⚠ | **WARNING** - Issue found but not critical |
| ✗ | **FAIL** - Data quality problem |

## Exit Codes

- `0` = All checks passed (ready to use)
- `1` = Issues found (review warnings)

## Common Usage

```bash
# Validate single ticker
python src/verify_data_consistency.py --ticker PETR4

# Skip slow checks
python src/verify_data_consistency.py --skip-prices

# Use different data directory
python src/verify_data_consistency.py --data-dir /path/to/data

# All options combined
python src/verify_data_consistency.py \
  --ticker VALE3 \
  --detailed \
  --output report.json
```

## Integration

```python
# In your code
from src.verify_data_consistency import DataConsistencyVerifier

verifier = DataConsistencyVerifier()
report = verifier.run_full_validation()

if report.overall_status() != 'pass':
    print("⚠️  Data issues detected")
    for check in report.macro_checks:
        if check.status != 'pass':
            print(f"  - {check.message}")
```

## Expected Results

```
✓ Overall Status: PASS

✓ PETR4, VALE3, WEGE3
  Price validation: All checks passed
  Fundamental data: All metrics reasonable

✓ Macro Data
  SELIC: 0.04% - 0.06% (valid range)
  IPCA: -0.68% - 1.62% (valid range)
  CDI: Tracking SELIC perfectly

⚠ Note: IPCA data ends in past (gap for current months)
```

## Real-World Ranges

| Metric | Range | Note |
|--------|-------|------|
| SELIC | 0% - 50% | Brazil's key rate |
| IPCA | -10% to +30% | Monthly inflation |
| CDI/SELIC diff | <1% | Should be nearly identical |
| Price jump | <50%/day | Larger = suspicious |
| P/E ratio | -1000 to +10000 | Wide range = volatile earnings |

## Troubleshooting

**"No data found"**
→ Check that files exist in `data/raw/prices/`, `data/raw/macro/`

**"Date Alignment warnings"**
→ Normal if macro data doesn't cover entire price range; doesn't block analysis

**"Data Completeness warnings"**
→ Some fundamentals missing; won't block ML training

**All prices from 1970**
→ Date column loading issue; check parquet structure

## Performance

- Quick run: ~1 second
- With details: ~2 seconds
- With JSON: ~3 seconds

## See Also

- **Full Guide**: `DATA_VERIFICATION_GUIDE.md`
- **Monitoring**: `src/monitor_data_quality.sh`
- **Script**: `src/verify_data_consistency.py`

---

**Status**: ✅ Production Ready | **Last Updated**: 2026-05-19
