# Feature Selection

**Data Split**

- MEMBER_KEY
- OFFER_RICHNESS_SERVED

**M1 - Probability Conversion**

-
-
-
-


**M2 - Points Purchased**

- HOUR_OF_DAY => [1,24]
- FLAG_FIRST_TIME_BUYER => Bool
- FLAG_FIRST_TIME_VISITOR => Bool:
- OFFER_RICHNESS_APPLIED => Categorize: 0, 1, 2
- CURRENT_BALANCED => outliers (95th), left skewed (capping 99th; IQR method; np.log1p)
- COUNT_TRANX_L12M => [1,22] => less variance as count increase

- POINTS_PURCHASED => left skewed


**Interesting Features**
- Probability of conversion is less when users see offer less than the one they saw before => same or better = 1; worse = 0 => `FLAG_WORSE_THAN_LAST_OFFER`
-

Huber loss → mix of L1 + L2
Quantile loss → robust to extremes

----

| Column Name                                  | Decision  | What to do                                                                                        |
|----------------------------------------------|-----------|---------------------------------------------------------------------------------------------------|
| SESSION_KEY                                  | drop      |                                                                                                   |
| SESSION_DATE                                 | transform | extract `HOUR_OF_DAY` and `DAY_OF_WEEK`                                                           |
| MEMBER_KEY                                   | drop      |                                                                                                   |
| FLAG_FIRST_TIME_VISITOR                      | as-is     |                                                                                                   |
| FLAG_FIRST_TIME_BUYER                        | as-is     |                                                                                                   |
| OFFER_RICHNESS_SERVED                        | as-is     |                                                                                                   |
| FLAG_TRANSACTION                             | target M1 |                                                                                                   |
| OFFER_RICHNESS_APPLIED                       | (drop)    | target derivative                                                                                 |
| POINTS_PURCHASED                             | target M2 | left skewed with outliers                                                                         |
| PRICE_PER_POINT                              | (drop)    | target derivative. computed from offer                                                            |
| CURRENT_BALANCE                              | transform | left skewed with tailed spike. apply `log1p` and `BALANCE_ZERO` flag;                             |
| COUNT_TRANX_L12M                             | transform | left skewed [1,22] . Bucket: [0,1,2,3-5,6-9,10+]                                                  |
| LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M  | transform | `CAT_OFFER_SERVED_WORSE_THAN_LAST` => 0 if same/better; 1 if never; 2 if worse                    |
| OFFER_RICHNESS_APPLIED_ON_LAST_PURCHASE_L12M | (drop)    |                                                                                                   |
| POINTS_PURCHASED_LAST_TRANX_L12M             | (drop?)   |                                                                                                   |
| DAYS_SINCE_LAST_PURCHASE_L12M                | transform | apply `sqrt` and `missing_DAYS_SINCE_LAST_PURCHASE_L12M`; sentinel value 9999 = 0                 |
| DAYS_SINCE_LAST_VISIT_NO_PURCHASE            | transform | left skewed, apply `log` and `missing_DAYS_SINCE_LAST_VISIT_NO_PURCHASE`; sentinel value 9999 = 0 |


----

Lasso → worst for tail (linear + shrinkage)
RandomForest → averages trees → smooths extremes
XGBoost / LightGBM → best candidates IF tuned + reweighted

----

Two towers methods with probabilistic blending:

y_pred = p_tail * y_tail + (1 - p_tail) * y_main
