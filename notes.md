- Add more features to the training data
- The model should think in the long term
- Buying CASH shouldn't cost in trading cost
- Reward should be the model performing better than the marker in the long run
- Create a "separate" model/training for investing a fixed amount periodically (adjusted by inflation maybe)
- THe model should basically learn the fundamentals of long term investing (buy good stocks when prices are down, avoid stocks going down hill, buy some dividend stocks)
- Maybe enanle entropy driven exploration (maybe 0.001)
- Maybe try other log_std_init's (current = -2)
- Why is the processed fundamentals going in "blocky" lines. In the raw_findamentals, they are smooth continuous lines. In processed they are straight for a whole quarter (maybe they do represent the same thing 
but hte graph is slightly different)
- Put more emphasis on percentage and adjusted data and also their variance over time and their relation to their respective sector (absolute values don't really mean that much)
- Verify pl outliers in processe data and earnings_growth_yoy 
- ✗ FAIL: Missing features: pe_ratio, pb_ratio, date
- Improve way we deal with NaN/Null fundamentals (how does the model deal with it?)
- Which model built the "Portfolio Snapshot" part of the visualization?
- Agent should easily and decisevely beat equal-weight, SELIC, BOVA11 and market-cap

