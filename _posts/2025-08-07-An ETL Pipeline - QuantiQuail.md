---
title: An ETL Pipeline - QuantiQuail
date: 2025-08-07
categories: [Machine Learning, Python, Quantitative Finance, ETL Pipeline]
tags: [random-forest, xgboost, etl, trading, data-science]
---
![QuantiQuail Project Header](/assets/quantiqual.png)
# An ETL Pipeline: QuantiQuail
<hr style="height: 2px; background-color: lightgray; border: none;">
---

## The name is certainly memorable, isn't it? 🐦

I recently finished my initial version of a machine learning model to predict if a stock/ticker will move upwards or downwards within a time window (currently next day). As of right now, it has a consistent accuracy on the testing set of ~45%. That level of low accuracy should be less accurate than random chance (since it is a binary classification problem - up or down - which has a random chance of 50% accuracy). However, at this point I can engineer different features and do transformations to reduce noise.

The goal of this project is to establish a full ETL pipeline with an accuracy that performs better than random chance - even if only a little! It's a difficult problem but that allows a lot of opportunities to explore different data science topics.

<hr style="height: 2px; background-color: lightgray; border: none;">
---

## The underlying model

The model I chose for this is a RandomForestClassifier. The problem becomes simpler if I aim to predict up (1) or down (0) as a class, rather than quantify the movement numerically. Additionally, the classifier allows for good feature evaluation which should help with early development - see below.

```python
    # Quick output of feature importances to avoid adding noise
    import pandas as pd
        feature_importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        print("Feature importances:\n", feature_importance_df)
```

Later, other models may perform better - such as XGBoost - but this one is insightful and quick to get up and running. XGBoost will also use decision trees, but sequentially rather than simultaneously. This allows continuous improvements on errors in earlier versions of the decision tree, illustrated below:

```python
# Round 1
Tree1_predicts = [0.6, 0.3, 0.8, 0.2]  # Probability of UP
Actual_results = [1, 0, 1, 0]           # 1=UP, 0=DOWN
Errors = [0.4, -0.3, 0.2, -0.2]        # What Tree1 got wrong

# Round 2
Tree2_learns_to_predict = [0.4, -0.3, 0.2, -0.2]  # The errors!
Tree2_predicts = [0.3, -0.2, 0.1, -0.1]           # Not perfect, but helps

# Combined prediction: Tree1 + Tree2
Combined = [0.6, 0.3, 0.8, 0.2] + [0.3, -0.2, 0.1, -0.1]
         = [0.9, 0.1, 0.9, 0.1]

New_errors = [1, 0, 1, 0] - [0.9, 0.1, 0.9, 0.1] = [0.1, -0.1, 0.1, -0.1]

# Round 3
Tree3_learns_to_predict = [0.1, -0.1, 0.1, -0.1]
```

<hr style="height: 2px; background-color: lightgray; border: none;">
---

## Planned Features

Initially, I had chosen SPY as my target for predictions - however, this particular ticker is relatively stable. It'll be difficult to pick up patterns, so I'll need tickers with more volatility.

The first feature I added was the label - an examination if the ticker provided a return over the specified period (one business day).

```python
    # Daily returns
    # We need a label to target based off of this feature for regression:
    # 1 for up prediction, 0 for down
    def daily_returns(self, data):
        data = data.copy()  # Avoid modifying the original DataFrame
        data['Daily Returns'] = data['Close'].pct_change().round(4)
        next_returns = data['Daily Returns'].shift(-1)
        
        # Some returns are so close to zero that they are not significant,
        # so we will use a threshold to avoid noise
        threshold = 0.001
        data['Label'] = -1 # Default label for insignificant returns
        data.loc[next_returns > threshold, 'Label'] = 1
        data.loc[next_returns < -threshold, 'Label'] = 0
        data = data[data['Label'] != -1].dropna().reset_index(drop=True)
        print(f"Daily returns feature added with {data['Label'].value_counts().to_dict()}")
        return data
```
To avoid noise, it drops anything that doesn't meet the threshold to provide a clear picture of when the data a decisive class label. At 0.001 as the initial threshold, there will be room later on to tune this value to potentially increase accuracy.

Revealing a pattern will require more data, transformed into features on the target ticker. I can compare the Relative Strength Index of another ticker to my target. For example, is there a pattern between Nvidia and TSMC? By calculating the RSI, the feature can capture patterns in the relative momentum between the two assets. In other words, is money likely to move from one similar asset to the other?


<hr style="height: 2px; background-color: lightgray; border: none;">
---

## Have a look

[Find the repo here](https://github.com/ossiesf/QuantiQuail)