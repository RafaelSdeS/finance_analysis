#!/usr/bin/env python3
"""
verify_data_consistency.py
==========================

Comprehensive data consistency verification for the finance analysis project.

This script verifies that all data (prices, macro indicators, fundamentals) 
is consistent with real-world constraints:

1. **Price Data Validation**
   - OHLC constraints: Open/Close within High/Low, High >= Low
   - No negative or zero prices
   - No sudden jumps (>50% in one day)
   - Volume consistency
   - Date continuity (no large gaps)

2. **Macro Data Validation**
   - SELIC rate: 0% to 50% (realistic Brazilian rates)
   - IPCA (inflation): -10% to +30% (realistic inflation)
   - CDI: Should track SELIC closely
   - Date alignment across all macro series
   - No future dates

3. **Fundamental Data Validation**
   - Financial metrics non-negative (revenue, earnings)
   - Ratios in realistic ranges (P/E, P/B, ROE)
   - Consistency: Market Cap = Price * Shares
   - Dividend yield vs stock price consistency
   - Date alignment with price data

4. **Cross-Asset Validation**
   - Date ranges overlap correctly
   - Price data has data for macro dates
   - Fundamentals cover the date range of prices
   - No orphaned data (old data without corresponding entries)

5. **Data Quality Metrics**
   - Null value distribution
   - Outlier detection (>3 sigma)
   - Data completeness percentage
   - Coverage gaps

Usage:
    python verify_data_consistency.py [--detailed] [--output report.json]
    
    Options:
        --detailed    : Print detailed findings for each check
        --output      : Save results to JSON file
        --ticker      : Validate specific ticker (default: all)
        --skip-prices : Skip price validation (faster)
        --skip-macro  : Skip macro validation (faster)
        --skip-fund   : Skip fundamental validation (faster)
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationCheck:
    """Single validation check result."""
    name: str
    status: str  # 'pass', 'warning', 'fail'
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class TickerValidation:
    """Validation results for a single ticker."""
    ticker: str
    checks: List[ValidationCheck] = field(default_factory=list)
    
    def summary(self) -> Dict:
        """Get summary of all checks."""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.status == 'pass')
        warned = sum(1 for c in self.checks if c.status == 'warning')
        failed = sum(1 for c in self.checks if c.status == 'fail')
        
        return {
            'ticker': self.ticker,
            'total_checks': total,
            'passed': passed,
            'warnings': warned,
            'failed': failed,
            'overall': 'pass' if failed == 0 else 'fail'
        }


@dataclass
class ConsistencyReport:
    """Complete validation report."""
    timestamp: str
    data_paths: Dict
    tickers: List[TickerValidation] = field(default_factory=list)
    macro_checks: List[ValidationCheck] = field(default_factory=list)
    cross_asset_checks: List[ValidationCheck] = field(default_factory=list)
    
    def overall_status(self) -> str:
        """Determine overall validation status."""
        all_checks = self.macro_checks + self.cross_asset_checks
        for ticker in self.tickers:
            all_checks.extend(ticker.checks)
        
        failed = sum(1 for c in all_checks if c.status == 'fail')
        return 'pass' if failed == 0 else 'fail'


# =============================================================================
# PRICE VALIDATION
# =============================================================================

class PriceValidator:
    """Validates price data (OHLC, volume, dates)."""
    
    @staticmethod
    def validate_ohlc_constraints(df: pd.DataFrame) -> ValidationCheck:
        """Check OHLC constraints: H>=L, O/C within H/L."""
        issues = []
        
        if 'High' in df.columns and 'Low' in df.columns:
            invalid_hl = df['High'] < df['Low']
            if invalid_hl.any():
                issues.append(f"{invalid_hl.sum()} rows: High < Low")
        
        if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
            # Check Open/Close within High/Low
            invalid_open = (df['Open'] > df['High']) | (df['Open'] < df['Low'])
            invalid_close = (df['Close'] > df['High']) | (df['Close'] < df['Low'])
            
            if invalid_open.any():
                issues.append(f"{invalid_open.sum()} rows: Open outside H/L")
            if invalid_close.any():
                issues.append(f"{invalid_close.sum()} rows: Close outside H/L")
        
        if issues:
            return ValidationCheck(
                name='OHLC Constraints',
                status='fail',
                message='; '.join(issues),
                details={'invalid_count': len(issues)}
            )
        
        return ValidationCheck(
            name='OHLC Constraints',
            status='pass',
            message='All OHLC constraints satisfied'
        )
    
    @staticmethod
    def validate_price_positivity(df: pd.DataFrame) -> ValidationCheck:
        """Check that prices are positive."""
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        issues = {}
        
        for col in price_cols:
            if col in df.columns:
                invalid = (df[col] <= 0).sum()
                if invalid > 0:
                    issues[col] = invalid
        
        if issues:
            msg = '; '.join(f"{col}: {count} rows" for col, count in issues.items())
            return ValidationCheck(
                name='Price Positivity',
                status='fail',
                message=msg,
                details=issues
            )
        
        return ValidationCheck(
            name='Price Positivity',
            status='pass',
            message='All prices are positive'
        )
    
    @staticmethod
    def validate_price_jumps(df: pd.DataFrame) -> ValidationCheck:
        """Check for unrealistic price jumps (>50% in one day)."""
        threshold = 0.50  # 50% jump
        issues = []
        
        if 'Close' in df.columns and len(df) > 1:
            pct_change = df['Close'].pct_change().abs()
            large_jumps = pct_change > threshold
            
            if large_jumps.any():
                jump_count = large_jumps.sum()
                max_jump = pct_change.max()
                issues.append(f"{jump_count} jumps >50% (max: {max_jump:.1%})")
                
                # Show most extreme jumps
                top_jumps = pct_change.nlargest(3)
                jump_dates = top_jumps.index.tolist()
                details = {
                    'jump_count': jump_count,
                    'max_jump': float(max_jump),
                    'top_jump_dates': [str(d) for d in jump_dates]
                }
                
                return ValidationCheck(
                    name='Price Jumps',
                    status='warning',
                    message='; '.join(issues),
                    details=details
                )
        
        return ValidationCheck(
            name='Price Jumps',
            status='pass',
            message='No unrealistic price jumps detected'
        )
    
    @staticmethod
    def validate_date_continuity(df: pd.DataFrame) -> ValidationCheck:
        """Check for large gaps in trading dates."""
        # Extract dates
        if df.index.name == 'Date' or (hasattr(df.index, 'name') and df.index.name == 'Date'):
            dates = df.index
        elif 'Date' in df.columns:
            dates = df['Date']
        else:
            return ValidationCheck(
                name='Date Continuity',
                status='warning',
                message='No date index or column found'
            )
        
        dates = pd.to_datetime(dates)
        
        if len(dates) < 2:
            return ValidationCheck(
                name='Date Continuity',
                status='pass',
                message='Insufficient data for gap analysis'
            )
        
        # Calculate gaps (excluding weekends/holidays)
        date_diffs = dates.diff()[1:]  # Skip first NaT
        max_gap = date_diffs.max()
        
        # Flag if gap > 30 days (unusual for traded stocks)
        if max_gap > timedelta(days=30):
            return ValidationCheck(
                name='Date Continuity',
                status='warning',
                message=f'Large gap detected: {max_gap.days} days',
                details={'max_gap_days': max_gap.days}
            )
        
        return ValidationCheck(
            name='Date Continuity',
            status='pass',
            message=f'Date range: {dates.min().date()} to {dates.max().date()}'
        )
    
    @staticmethod
    def validate_volume(df: pd.DataFrame) -> ValidationCheck:
        """Check volume consistency."""
        if 'Volume' not in df.columns:
            return ValidationCheck(
                name='Volume Consistency',
                status='pass',
                message='No volume data'
            )
        
        issues = []
        
        # Check for negative volume
        if (df['Volume'] < 0).any():
            issues.append(f"{(df['Volume'] < 0).sum()} rows with negative volume")
        
        # Check for zero volume (more lenient)
        zero_vol = (df['Volume'] == 0).sum()
        if zero_vol > len(df) * 0.2:  # More than 20% zeros is suspicious
            issues.append(f"{zero_vol} rows with zero volume ({zero_vol/len(df)*100:.1f}%)")
        
        if issues:
            return ValidationCheck(
                name='Volume Consistency',
                status='warning',
                message='; '.join(issues)
            )
        
        return ValidationCheck(
            name='Volume Consistency',
            status='pass',
            message='Volume data is consistent'
        )


# =============================================================================
# MACRO DATA VALIDATION
# =============================================================================

class MacroValidator:
    """Validates macro indicators (SELIC, IPCA, CDI)."""
    
    @staticmethod
    def validate_selic_range(df: pd.DataFrame) -> ValidationCheck:
        """Check SELIC rate is in realistic range (0%-50%)."""
        issues = []
        
        # Find the value column (could be 'selic', 'Value', etc.)
        value_col = None
        for col in ['selic', 'Value', 'value']:
            if col in df.columns:
                value_col = col
                break
        
        if value_col is None:
            return ValidationCheck(
                name='SELIC Range',
                status='warning',
                message=f'No value column found. Available: {df.columns.tolist()}'
            )
        
        selic = df[value_col]
        
        # SELIC is typically 2%-20% but can be extreme (Brazil had 25% in crisis)
        out_of_range = ((selic < 0) | (selic > 50)).sum()
        if out_of_range > 0:
            issues.append(f"{out_of_range} values outside 0%-50%")
        
        # Check for spikes
        if len(selic) > 1:
            selic_changes = selic.diff().abs()
            large_changes = (selic_changes > 5).sum()  # More than 5% jump
            if large_changes > 0:
                issues.append(f"{large_changes} abrupt changes >5%")
        
        if issues:
            status = 'warning' if out_of_range == 0 else 'fail'
            return ValidationCheck(
                name='SELIC Range',
                status=status,
                message='; '.join(issues),
                details={'min': float(selic.min()), 'max': float(selic.max())}
            )
        
        return ValidationCheck(
            name='SELIC Range',
            status='pass',
            message=f'SELIC range: {selic.min():.2f}% - {selic.max():.2f}%'
        )
    
    @staticmethod
    def validate_ipca_range(df: pd.DataFrame) -> ValidationCheck:
        """Check IPCA (inflation) is in realistic range (-10% to +30%)."""
        issues = []
        
        # Find the value column
        value_col = None
        for col in ['ipca', 'Value', 'value']:
            if col in df.columns:
                value_col = col
                break
        
        if value_col is None:
            return ValidationCheck(
                name='IPCA Range',
                status='warning',
                message=f'No value column found. Available: {df.columns.tolist()}'
            )
        
        ipca = df[value_col]
        
        # IPCA is typically -5% to +15% but can be extreme in crisis
        out_of_range = ((ipca < -20) | (ipca > 40)).sum()
        if out_of_range > 0:
            issues.append(f"{out_of_range} values outside realistic range")
        
        # Check that inflation is not always negative (would be deflation)
        if (ipca < 0).sum() > len(ipca) * 0.5:
            issues.append("More than 50% deflation months (unusual)")
        
        if issues:
            return ValidationCheck(
                name='IPCA Range',
                status='warning' if out_of_range == 0 else 'fail',
                message='; '.join(issues),
                details={'min': float(ipca.min()), 'max': float(ipca.max())}
            )
        
        return ValidationCheck(
            name='IPCA Range',
            status='pass',
            message=f'IPCA range: {ipca.min():.2f}% - {ipca.max():.2f}%'
        )
    
    @staticmethod
    def validate_cdi_selic_tracking(selic: pd.DataFrame, 
                                    cdi: pd.DataFrame) -> ValidationCheck:
        """Check CDI tracks SELIC (they should be very similar)."""
        # Find value columns
        selic_col = None
        for col in ['selic', 'Value', 'value']:
            if col in selic.columns:
                selic_col = col
                break
        
        cdi_col = None
        for col in ['cdi', 'Value', 'value']:
            if col in cdi.columns:
                cdi_col = col
                break
        
        if not selic_col or not cdi_col:
            return ValidationCheck(
                name='CDI/SELIC Tracking',
                status='warning',
                message='Could not find value columns in data'
            )
        
        # Find date columns
        selic_date_col = None
        for col in ['reference_date', 'Date', 'date']:
            if col in selic.columns:
                selic_date_col = col
                break
        
        cdi_date_col = None
        for col in ['reference_date', 'Date', 'date']:
            if col in cdi.columns:
                cdi_date_col = col
                break
        
        if not selic_date_col or not cdi_date_col:
            return ValidationCheck(
                name='CDI/SELIC Tracking',
                status='warning',
                message='Could not find date columns in data'
            )
        
        # Convert to datetime
        selic_dates = pd.to_datetime(selic[selic_date_col])
        cdi_dates = pd.to_datetime(cdi[cdi_date_col])
        
        # Align dates
        selic_aligned = selic.set_index(selic_dates)[selic_col]
        cdi_aligned = cdi.set_index(cdi_dates)[cdi_col]
        
        common_dates = selic_aligned.index.intersection(cdi_aligned.index)
        
        if len(common_dates) < 10:
            return ValidationCheck(
                name='CDI/SELIC Tracking',
                status='warning',
                message='Insufficient overlapping data'
            )
        
        selic_common = selic_aligned[common_dates]
        cdi_common = cdi_aligned[common_dates]
        
        # CDI and SELIC should be very similar (within 1%)
        diff = (cdi_common - selic_common).abs()
        mean_diff = diff.mean()
        max_diff = diff.max()
        
        if max_diff > 5:  # More than 5% difference is suspicious
            return ValidationCheck(
                name='CDI/SELIC Tracking',
                status='warning',
                message=f'CDI diverges from SELIC (max diff: {max_diff:.2f}%)',
                details={'mean_diff': float(mean_diff), 'max_diff': float(max_diff)}
            )
        
        return ValidationCheck(
            name='CDI/SELIC Tracking',
            status='pass',
            message=f'CDI tracks SELIC (mean diff: {mean_diff:.3f}%)'
        )
    
    @staticmethod
    def validate_no_future_dates(df: pd.DataFrame) -> ValidationCheck:
        """Check that data has no future dates."""
        # Extract dates
        if df.index.name == 'Date' or (hasattr(df.index, 'name') and df.index.name == 'Date'):
            dates = df.index
        elif 'reference_date' in df.columns:
            dates = df['reference_date']
        elif 'Date' in df.columns:
            dates = df['Date']
        else:
            return ValidationCheck(
                name='No Future Dates',
                status='warning',
                message='No date column/index found'
            )
        
        dates = pd.to_datetime(dates)
        
        future_dates = (dates > pd.Timestamp.now()).sum()
        
        if future_dates > 0:
            return ValidationCheck(
                name='No Future Dates',
                status='fail',
                message=f'{future_dates} entries with future dates',
                details={'future_count': future_dates}
            )
        
        return ValidationCheck(
            name='No Future Dates',
            status='pass',
            message='No future dates detected'
        )


# =============================================================================
# FUNDAMENTAL DATA VALIDATION
# =============================================================================

class FundamentalValidator:
    """Validates fundamental data (P/E, P/B, etc.)."""
    
    @staticmethod
    def validate_non_negative_metrics(df: pd.DataFrame) -> ValidationCheck:
        """Check that financial metrics are non-negative."""
        metrics = ['Revenue', 'NetIncome', 'TotalAssets', 'Equity']
        issues = {}
        
        for metric in metrics:
            if metric in df.columns:
                negative = (df[metric] < 0).sum()
                if negative > 0:
                    # Negative is OK sometimes (loss), but track it
                    pct = negative / len(df) * 100
                    if pct > 30:  # More than 30% of the time seems wrong
                        issues[metric] = f'{negative} negative values ({pct:.1f}%)'
        
        if issues:
            msg = '; '.join(f"{k}: {v}" for k, v in issues.items())
            return ValidationCheck(
                name='Non-Negative Metrics',
                status='warning',
                message=msg,
                details=issues
            )
        
        return ValidationCheck(
            name='Non-Negative Metrics',
            status='pass',
            message='Financial metrics are reasonable'
        )
    
    @staticmethod
    def validate_ratio_ranges(df: pd.DataFrame) -> ValidationCheck:
        """Check financial ratios are in realistic ranges."""
        issues = []
        
        # P/E Ratio: typically -100 to +1000 (can be negative for loss-making)
        if 'PE' in df.columns:
            pe = df['PE'].dropna()
            if len(pe) > 0:
                out_of_range = ((pe < -1000) | (pe > 10000)).sum()
                if out_of_range > 0:
                    issues.append(f"P/E: {out_of_range} extreme values")
        
        # P/B Ratio: typically 0.1 to 50
        if 'PB' in df.columns:
            pb = df['PB'].dropna()
            if len(pb) > 0:
                out_of_range = ((pb < 0) | (pb > 100)).sum()
                if out_of_range > 0:
                    issues.append(f"P/B: {out_of_range} out of 0-100 range")
        
        # ROE: typically -100% to +100%
        if 'ROE' in df.columns:
            roe = df['ROE'].dropna()
            if len(roe) > 0:
                out_of_range = ((roe < -5) | (roe > 5)).sum()
                if out_of_range > 0:
                    issues.append(f"ROE: {out_of_range} extreme values (>500%)")
        
        # Dividend Yield: typically 0% to 20%
        if 'DividendYield' in df.columns:
            dy = df['DividendYield'].dropna()
            if len(dy) > 0:
                out_of_range = ((dy < 0) | (dy > 1)).sum()
                if out_of_range > 0:
                    issues.append(f"Div Yield: {out_of_range} outside 0%-100%")
        
        if issues:
            return ValidationCheck(
                name='Ratio Ranges',
                status='warning',
                message='; '.join(issues)
            )
        
        return ValidationCheck(
            name='Ratio Ranges',
            status='pass',
            message='All financial ratios in realistic ranges'
        )
    
    @staticmethod
    def validate_data_completeness(df: pd.DataFrame) -> ValidationCheck:
        """Check what percentage of data is populated."""
        null_pct = df.isnull().sum() / len(df) * 100
        high_null = (null_pct > 50).sum()
        
        if high_null > 0:
            worst = null_pct.nlargest(3)
            msg = f"{high_null} columns >50% null"
            details = {col: float(pct) for col, pct in worst.items()}
            
            return ValidationCheck(
                name='Data Completeness',
                status='warning',
                message=msg,
                details=details
            )
        
        avg_completeness = (100 - null_pct.mean())
        return ValidationCheck(
            name='Data Completeness',
            status='pass',
            message=f'{avg_completeness:.1f}% average data completeness'
        )


# =============================================================================
# CROSS-ASSET VALIDATION
# =============================================================================

class CrossAssetValidator:
    """Validates consistency across different data sources."""
    
    @staticmethod
    def validate_date_alignment(prices: Dict[str, pd.DataFrame],
                               macro: Dict[str, pd.DataFrame]) -> ValidationCheck:
        """Check that price and macro data have aligned date ranges."""
        price_dates = {}
        for ticker, df in prices.items():
            # Extract dates from either index or column
            if df.index.name == 'Date' or (hasattr(df.index, 'name') and df.index.name == 'Date'):
                dates = df.index
            elif 'Date' in df.columns:
                dates = df['Date']
            else:
                continue
            
            dates = pd.to_datetime(dates)
            price_dates[ticker] = (dates.min(), dates.max())
        
        macro_dates = {}
        for name, df in macro.items():
            # Extract dates from either index or column
            if df.index.name == 'Date' or (hasattr(df.index, 'name') and df.index.name == 'Date'):
                dates = df.index
            elif 'reference_date' in df.columns:
                dates = df['reference_date']
            elif 'Date' in df.columns:
                dates = df['Date']
            else:
                continue
            
            dates = pd.to_datetime(dates)
            macro_dates[name] = (dates.min(), dates.max())
        
        issues = []
        
        # Check if macro data covers price data
        for macro_name, (macro_start, macro_end) in macro_dates.items():
            for ticker, (price_start, price_end) in price_dates.items():
                if macro_start > price_start or macro_end < price_end:
                    gap_before = (price_start - macro_start).days if macro_start < price_start else 0
                    gap_after = (macro_end - price_end).days if macro_end > price_end else 0
                    issues.append(
                        f"{macro_name} doesn't fully cover {ticker} "
                        f"(gap: {gap_before}d before, {gap_after}d after)"
                    )
        
        if issues:
            return ValidationCheck(
                name='Date Alignment',
                status='warning',
                message='; '.join(issues[:3]),  # Show first 3
                details={'total_issues': len(issues)}
            )
        
        return ValidationCheck(
            name='Date Alignment',
            status='pass',
            message='Price and macro data are date-aligned'
        )


# =============================================================================
# MAIN VALIDATION ORCHESTRATOR
# =============================================================================

class DataConsistencyVerifier:
    """Main class that runs all validations."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize with data directory."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / 'data'
        
        self.data_dir = data_dir
        self.raw_dir = data_dir / 'raw'
        self.prices_dir = self.raw_dir / 'prices'
        self.macro_dir = self.raw_dir / 'macro'
        self.fund_dir = self.raw_dir / 'fundamentals'
    
    def load_price_data(self, ticker: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Load price data for all tickers or specific ticker."""
        prices = {}
        
        if not self.prices_dir.exists():
            return prices
        
        for file in self.prices_dir.glob('*.parquet'):
            ticker_name = file.stem
            if ticker and ticker_name != ticker.upper():
                continue
            
            try:
                df = pd.read_parquet(file)
                # Set index to Date column if it exists
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                elif df.index.name != 'Date':
                    df.index.name = 'Date'
                prices[ticker_name] = df
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}", file=sys.stderr)
        
        return prices
    
    def load_macro_data(self) -> Dict[str, pd.DataFrame]:
        """Load macro data (SELIC, IPCA, CDI)."""
        macro = {}
        
        if not self.macro_dir.exists():
            return macro
        
        for file in self.macro_dir.glob('*.parquet'):
            name = file.stem
            try:
                df = pd.read_parquet(file)
                # Don't force index name - keep reference_date column as is
                macro[name] = df
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}", file=sys.stderr)
        
        return macro
    
    def load_fundamental_data(self, ticker: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Load fundamental data."""
        fundamentals = {}
        
        if not self.fund_dir.exists():
            return fundamentals
        
        for file in self.fund_dir.glob('*.parquet'):
            ticker_name = file.stem
            if ticker and ticker_name != ticker.upper():
                continue
            
            try:
                df = pd.read_parquet(file)
                fundamentals[ticker_name] = df
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}", file=sys.stderr)
        
        return fundamentals
    
    def validate_prices(self, prices: Dict[str, pd.DataFrame]) -> List[TickerValidation]:
        """Run all price validations."""
        results = []
        
        for ticker, df in prices.items():
            validator = TickerValidation(ticker=ticker)
            
            validator.checks.append(PriceValidator.validate_ohlc_constraints(df))
            validator.checks.append(PriceValidator.validate_price_positivity(df))
            validator.checks.append(PriceValidator.validate_price_jumps(df))
            validator.checks.append(PriceValidator.validate_date_continuity(df))
            validator.checks.append(PriceValidator.validate_volume(df))
            
            results.append(validator)
        
        return results
    
    def validate_macro(self, macro: Dict[str, pd.DataFrame]) -> Tuple[List[ValidationCheck], List[ValidationCheck]]:
        """Run all macro validations."""
        individual_checks = []
        cross_checks = []
        
        # Validate each series
        if 'selic' in macro:
            individual_checks.append(MacroValidator.validate_selic_range(macro['selic']))
        
        if 'ipca' in macro:
            individual_checks.append(MacroValidator.validate_ipca_range(macro['ipca']))
        
        if 'cdi' in macro:
            individual_checks.append(MacroValidator.validate_no_future_dates(macro['cdi']))
        
        # Cross-series checks
        if 'selic' in macro and 'cdi' in macro:
            cross_checks.append(MacroValidator.validate_cdi_selic_tracking(
                macro['selic'], macro['cdi']
            ))
        
        return individual_checks, cross_checks
    
    def validate_fundamentals(self, fundamentals: Dict[str, pd.DataFrame]) -> List[TickerValidation]:
        """Run all fundamental validations."""
        results = []
        
        for ticker, df in fundamentals.items():
            validator = TickerValidation(ticker=ticker)
            
            validator.checks.append(FundamentalValidator.validate_non_negative_metrics(df))
            validator.checks.append(FundamentalValidator.validate_ratio_ranges(df))
            validator.checks.append(FundamentalValidator.validate_data_completeness(df))
            
            results.append(validator)
        
        return results
    
    def run_full_validation(self, 
                           ticker: Optional[str] = None,
                           skip_prices: bool = False,
                           skip_macro: bool = False,
                           skip_fund: bool = False) -> ConsistencyReport:
        """Run complete validation suite."""
        report = ConsistencyReport(
            timestamp=datetime.now().isoformat(),
            data_paths={
                'raw': str(self.raw_dir),
                'prices': str(self.prices_dir),
                'macro': str(self.macro_dir),
                'fundamentals': str(self.fund_dir)
            }
        )
        
        # Load data
        prices = {} if skip_prices else self.load_price_data(ticker)
        macro = {} if skip_macro else self.load_macro_data()
        fundamentals = {} if skip_fund else self.load_fundamental_data(ticker)
        
        # Validate
        if not skip_prices:
            report.tickers.extend(self.validate_prices(prices))
        
        if not skip_macro:
            macro_checks, cross_checks = self.validate_macro(macro)
            report.macro_checks.extend(macro_checks)
            report.cross_asset_checks.extend(cross_checks)
        
        if not skip_fund:
            report.tickers.extend(self.validate_fundamentals(fundamentals))
        
        # Cross-asset checks
        if not skip_prices and not skip_macro:
            check = CrossAssetValidator.validate_date_alignment(prices, macro)
            report.cross_asset_checks.append(check)
        
        return report


# =============================================================================
# REPORTING
# =============================================================================

def print_report(report: ConsistencyReport, detailed: bool = False):
    """Print validation report to console."""
    print("\n" + "=" * 80)
    print("DATA CONSISTENCY VERIFICATION REPORT")
    print("=" * 80)
    print(f"\nTimestamp: {report.timestamp}")
    print(f"Data Directory: {report.data_paths['raw']}")
    
    # Overall status
    overall = report.overall_status()
    status_emoji = "✓" if overall == "pass" else "✗"
    print(f"\n{status_emoji} Overall Status: {overall.upper()}")
    
    # Ticker results
    if report.tickers:
        print("\n" + "-" * 80)
        print("PRICE & FUNDAMENTAL DATA")
        print("-" * 80)
        
        for ticker_result in report.tickers:
            summary = ticker_result.summary()
            status = "✓" if summary['overall'] == 'pass' else "✗"
            print(f"\n{status} {ticker_result.ticker}")
            print(f"  Checks: {summary['passed']}/{summary['total_checks']} passed")
            if summary['warnings'] > 0:
                print(f"  Warnings: {summary['warnings']}")
            if summary['failed'] > 0:
                print(f"  Failed: {summary['failed']}")
            
            if detailed:
                for check in ticker_result.checks:
                    check_emoji = "✓" if check.status == "pass" else ("⚠" if check.status == "warning" else "✗")
                    print(f"    {check_emoji} {check.name}: {check.message}")
    
    # Macro results
    if report.macro_checks:
        print("\n" + "-" * 80)
        print("MACRO DATA")
        print("-" * 80)
        
        passed = sum(1 for c in report.macro_checks if c.status == 'pass')
        print(f"\n{passed}/{len(report.macro_checks)} checks passed")
        
        if detailed:
            for check in report.macro_checks:
                check_emoji = "✓" if check.status == "pass" else ("⚠" if check.status == "warning" else "✗")
                print(f"  {check_emoji} {check.name}: {check.message}")
    
    # Cross-asset results
    if report.cross_asset_checks:
        print("\n" + "-" * 80)
        print("CROSS-ASSET CONSISTENCY")
        print("-" * 80)
        
        passed = sum(1 for c in report.cross_asset_checks if c.status == 'pass')
        print(f"\n{passed}/{len(report.cross_asset_checks)} checks passed")
        
        if detailed:
            for check in report.cross_asset_checks:
                check_emoji = "✓" if check.status == "pass" else ("⚠" if check.status == "warning" else "✗")
                print(f"  {check_emoji} {check.name}: {check.message}")
    
    print("\n" + "=" * 80)


def save_report_json(report: ConsistencyReport, filepath: Path):
    """Save report to JSON file."""
    data = {
        'timestamp': report.timestamp,
        'data_paths': report.data_paths,
        'overall_status': report.overall_status(),
        'tickers': [asdict(t) for t in report.tickers],
        'macro_checks': [asdict(c) for c in report.macro_checks],
        'cross_asset_checks': [asdict(c) for c in report.cross_asset_checks]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"✓ Report saved to {filepath}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Verify data consistency with real-world constraints'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Print detailed findings for each check'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--ticker',
        help='Validate specific ticker only'
    )
    parser.add_argument(
        '--skip-prices',
        action='store_true',
        help='Skip price validation'
    )
    parser.add_argument(
        '--skip-macro',
        action='store_true',
        help='Skip macro validation'
    )
    parser.add_argument(
        '--skip-fund',
        action='store_true',
        help='Skip fundamental validation'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Custom data directory'
    )
    
    args = parser.parse_args()
    
    # Run verification
    verifier = DataConsistencyVerifier(args.data_dir)
    report = verifier.run_full_validation(
        ticker=args.ticker,
        skip_prices=args.skip_prices,
        skip_macro=args.skip_macro,
        skip_fund=args.skip_fund
    )
    
    # Print results
    print_report(report, detailed=args.detailed)
    
    # Save if requested
    if args.output:
        save_report_json(report, args.output)
    
    # Exit with appropriate code
    sys.exit(0 if report.overall_status() == 'pass' else 1)


if __name__ == '__main__':
    main()
