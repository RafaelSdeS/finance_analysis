#!/bin/bash
#
# monitor_data_quality.sh
# =======================
#
# Continuous data quality monitoring script for the finance_analysis project.
#
# Usage:
#   ./monitor_data_quality.sh              # Run once
#   ./monitor_data_quality.sh --watch      # Run every 24 hours
#   ./monitor_data_quality.sh --interval 3600  # Run every hour
#
# This script:
#   1. Runs data consistency verification
#   2. Saves timestamped reports
#   3. Compares with previous run
#   4. Generates alerts if issues detected
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_DIR/data/validation"
VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
WATCH_MODE=false
INTERVAL=86400  # Default: 24 hours

while [[ $# -gt 0 ]]; do
    case $1 in
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --interval)
            INTERVAL="$2"
            WATCH_MODE=true
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to run validation
run_validation() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local report_file="$REPORTS_DIR/report_${timestamp}.json"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "DATA QUALITY CHECK - $(date +'%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Activate virtual environment
    if [ -f "$VENV_ACTIVATE" ]; then
        source "$VENV_ACTIVATE"
    fi
    
    # Run verification
    if python "$SCRIPT_DIR/verify_data_consistency.py" --output "$report_file" --detailed; then
        echo -e "${GREEN}✓ PASS${NC} - All data consistency checks passed"
        STATUS=0
    else
        echo -e "${RED}✗ FAIL${NC} - Data consistency issues detected"
        STATUS=1
    fi
    
    # Compare with previous report
    if [ -f "$REPORTS_DIR/latest_report.json" ]; then
        echo ""
        echo "Comparing with previous report..."
        
        PREV_FAILED=$(grep -o '"failed": [0-9]*' "$REPORTS_DIR/latest_report.json" | grep -o '[0-9]*' | awk '{sum+=$1} END {print sum}')
        CURR_FAILED=$(grep -o '"failed": [0-9]*' "$report_file" | grep -o '[0-9]*' | awk '{sum+=$1} END {print sum}')
        
        if [ "$CURR_FAILED" -lt "$PREV_FAILED" ]; then
            echo -e "${GREEN}✓ Improvement${NC} - Failed checks reduced from $PREV_FAILED to $CURR_FAILED"
        elif [ "$CURR_FAILED" -gt "$PREV_FAILED" ]; then
            echo -e "${RED}✗ Regression${NC} - Failed checks increased from $PREV_FAILED to $CURR_FAILED"
        else
            echo "→ No change in failed checks ($CURR_FAILED)"
        fi
    fi
    
    # Update latest report
    cp "$report_file" "$REPORTS_DIR/latest_report.json"
    
    echo ""
    echo "Report saved to: $report_file"
    echo ""
    
    return $STATUS
}

# Function to run on interval
run_monitoring() {
    local next_run=""
    
    while true; do
        run_validation
        
        next_run=$(date -d "+$INTERVAL seconds" +'%Y-%m-%d %H:%M:%S' 2>/dev/null || \
                  date -v+${INTERVAL}S +'%Y-%m-%d %H:%M:%S' 2>/dev/null || \
                  echo "in $INTERVAL seconds")
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Next check: $next_run"
        echo "Press Ctrl+C to stop monitoring"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        sleep "$INTERVAL"
    done
}

# Main execution
if [ "$WATCH_MODE" = true ]; then
    echo "Starting continuous monitoring (interval: $INTERVAL seconds)..."
    run_monitoring
else
    run_validation
fi
