---
title: "Plusgrade Take Home Assignment"
author: "by Emulie Chhor"
date: "March 16th, 2026"
# theme: focus
# theme: Nord
theme: Arguelles
# theme: metropolis
# header-includes:
#     - \metroset{progressbar=frametitle}
#     - \metroset{block=fill}
#   - \metroset{background=dark}
colortheme: default # owl
fontsize: 10pt
aspectratio: 169
---

# 1) Problem Statement

$$E[X] = \underbrace{P[\text{Buy} | \text{user, offer}]}_{\text{classification}} \times \underbrace{E[\text{Points} | \text{user, offer}]}_{\text{regression}} \times \text{price per points}$$

---

# 2) Offer Allocation Strategy

```
for offer in [0.4, 0.45, 0.5]:
    p_convert = classifier.predict(user, offer)
    points_purchased = regression.predict(user, offer)
    price_per_points = 0.03 * (1 - offer)
    expected_revenue = p_convert * points_purchased * price_per_points
```

---

# 3) EDA - pandas_profiling

- Class Imbalance
    * `OFFER_RICHNESS_SERVED` -> 0.4: 5%, 0.45: 18%, 0.5: 77%
    * `FLAG_TRANSACTION` -> 86% users don't convert
- Outlier
    * `POINT_PURCHASED` -> left skewed, but outlier at 60K
- Important features:
    * `FLAG_FIRST_TIME_BUYER`
    * `CURRENT_BALANCE`
    * `OFFER_RICHNESS_SERVED` vs `OFFER_APPLIED_LAST_TRANX_L12M`

---

# 4) Models

Classification: `AUC-ROC`

- Logistic Regression
- Naive Bayes
- Random Forest Classifier with `class_weight='balanced'`

Regression: `Huber Loss`, `RMSE`

- Linear Regression, Ridge, Lasso (scaled features with `StandardScaler`)
- Random Forest
- XGBoost, LightGBM with `scale_pos_weight`

Hyperparameters Tuning

- Backward Elimination
- Bayesian Optimization

---

# 5) Next Steps

- Feature Hashing with `MEMBER_ID` and `SESSION_ID`
- Use `imblearn`
- Ensemble methods with undersampling
- Two Towers
