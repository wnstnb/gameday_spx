# TL;DR on WTF
- The purpose of this project is to predict whether the current day's close will be above the previous day's close (`Target`).
- Two different ML models are used and their outputs are combined to produce a signal. True for a green day, False for a red day. 
- When both models are in agreement, there's a 70% chance it's the right call (`precision`).
- Both models utilize XGBoost, but one is a regressor and the other is a classifier.
- Both models were validated using walk-forward validation. 

# Results and Validation Process
Based on walk-forward validation, both models were able to achieve ~70% precision. This means that, for both models, their ability to classify the positive class was successful 70% of the time. Said another way: when both models predicted that the current day's close was going to be green, they were correct 70% of the time. This also translated to the opposite: when both models predicted red, it was also correct 70% of the time. 

These results were acheived by walk-forward validation, or feeding the model new data and then validating it. I chose this design because updating the models would happen every day with new data. 

# Features
- `BigNewsDay`: This feature represents whether the current day is a "big news day", eg. NFP, CPI, PPI, CPE, etc. 
- `Quarter`: The current quarter as of previous day (Q1,Q2,Q3,Q4) 
- `Perf5Day`: Whether the previous day's close is higher than it was 5 days earlier.
- `Perf5Day_n1`: Previous value of 👆🏽
- `DaysGreen`: Consecutive number of days green (close > previous close), as of the previous day.
- `DaysRed`: Consecutive number of days red (close <= previous close), as of the previous day.
- `CurrentGap`: The current day's gap as a percentage of the previous close, or (Open - Previous close) / Previous Close
- `RangePct`: The previous day's range as a percent of the prior day's close.
- `RangePct_n1`: Previous value of 👆🏽
- `RangePct_n2`: Previous value of 👆🏽
- `OHLC4_VIX`: The previous day's OHLC4 of VIX.
- `OHLC4_VIX_n1`: Previous value of 👆🏽
- `OHLC4_VIX_n2`: Previous value of 👆🏽