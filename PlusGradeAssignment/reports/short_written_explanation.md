### Problem Framing

The reward signal for "Buy Points" storefront is revenue per session, defined
as zero for non-converting sessions and $\text{Points Purchased} \times \text{Price per point}$ for converting one. Because the expected points purchased can be
decomposed a user's propensity to convert and the number of points they will
buy, we get the following formula for the expected revenue:

$E[\text{Revenue} | \text{member}, \text{offer}] = P[\text{convert} | \text{member}, \text{offer}] \times E[\text{points} | \text{convert}, \text{member}, \text{offer}] \times \text{price\_per\_point}(\text{offer})$

$$E[X] = P[\text{Buy} | \text{user, offer}] \times E[\text{Points}] \times \text{price_per_points}$$


Two separate models will estimate each factor: one model for a user probability
to convert (model1) and one model for the points purchased (model2). Decomposing
the expected revenue is preferred over a single model because it isolates
conversion sensibility from purchase size, making each model easier to diagnose
in production.

### Assumptions & Offline Evaluation Limitations

Ensuing the EDA process, the following assumptions were made during the modeling
approach:

1. Stationarity. Given that the data provided only includes 2 days of data, we assume offer response curve are stable over the evaluation period, which wouldn't be the case with the complete historical data: we expect product changes and some seasonal effects (people tend to travel more during summer and around the holiday period) to violate this assumption.
2. Stable Unit Treatment Value Assumption (SUTVA). We assume that there are no social referral dynamics or network effects at play and thus, showing offer X to member A does not affect member B's conversion probability.
3. Feature Sufficiency. We assume that the provided member features are sufficient to characterize conversion propensity and that unobserved cofounders (promotions, app notifications, ...) doesn't affect convert it.

### Model & Strategy Design

Our strategy design consists of two steps:

1. Predictive Layer. 2 models where trained
    a. Model 1 - Conversion Probability (classification). Trained on all sessions with `FLAG_TRANSACTION` as target. Tested on logistic regression, naive bayes, random forest, lightGBM and XGBoost for interpretability and evaluated on AUC-ROC curves.
    b. Model 2 - Points Purchased (regression). Trained exclusively on the session with transactions with `log(POINTS_PURCHASED)` as a target to adress right skew. Predictions are clipped at the 3k point-business minimum. Tested on Ridge, Lasso, Random Forest, XGBoost and evaluated on RMSE.
2. Offer Allocation Strategy. The expected revenue for each test session is scored 3 times, once per offer. Offer allocation is first assigned greddily, where each test session is assigned the offer which maximize the highest expected revenue ie $E[revenue] = P[convert] \times E[points] \times price\_per\_points$. Once, all test sessions have been assigned an offer, we verify the business constraints. If price floor of offer coverage constraints are unmet, we flip the sessions where offering 50% discount to 45% cost the least.

To ensure no data leakage, training/testing/validation split were made at the user
level: `MEMBER_KEY` can be only part of a single split. Hyperparameter tuning
is performed using Bayesian Search and Backward Elimination and RFE are used post-tuning to prune low-signal features.

### Evaluation

Here are our strategy's performance compared to others naive methods:

| Strategy                        | Rev per session | Avg Price/Point | 40% share | 45% share | 50% share | Price floor | Coverage |
|---------------------------------|-----------------|-----------------|-----------|-----------|-----------|-------------|----------|
| Always-40% baseline             | 53.617033       | 0.018000        | 1.0000    | 0.0000    | 0.0000    | True        | False    |
| Always-45% baseline             | 60.247197       | 0.016500        | 0.0000    | 1.0000    | 0.0000    | True        | False    |
| Always-50% baseline             | 67.567676       | 0.015000        | 0.0000    | 0.0000    | 1.0000    | False       | False    |
| Uniform random                  | 60.721574       | 0.016295        | 0.3337    | 0.3297    | 0.3367    | True        | True     |
| Historical observed             | 65.349855       | 0.015260        | 0.0410    | 0.1828    | 0.7762    | False       | False    |
| Ours                            | 72.363258       | 0.016046        | 0.2028    | 0.1499    | 0.6474    | True        | True     |

This evaluation strategy has some risks:

- Counterfactual Extrapolation: revenue estimates for offers that have not been showed are extrapolated.
- Calibration Sensitivity: Expected revenue is a product of two models outputs, and as a result, miscalibration compounds.


### Scenario Questions - Unexpected Drop in Conversions

Detecting changes in conversion rates early relies on daily monitoring dashboard with
notifications when the inference data differs from the training
data by X% by segmentation (offer tier, average revenue per session, etc).
The initial threshold can be set at 2 standard deviation above/below the
30-day training baseline to start. When a drop is detected, the first
investigation steps would be:

1. Offer Allocation Stratification. Stratify the drop by offer tier and member segment to isolate which member type is most responsible for the drop.
2. Data Quality. Confirm that the user sessions are logged correctly and there are no data leakage when training. If pipeline issues is the cause, then fix them.
3. Data Drift. Determine if data distributions has inherently changed. If inference data is inherently different and all data is available, then trigger retraining (assuming re-training is not too costly).
4. Look for external confounders. If data distributions are relatively similar, then I would be looking for external confounders missing from the feature set (app releases, seasonal events, promotions, competitors) and determine if they should be included.
