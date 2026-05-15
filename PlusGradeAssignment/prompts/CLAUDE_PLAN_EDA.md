# Plan: EDA Notebook for DATA_OFFER_ALLOCATION.csv

## Context

Build a comprehensive EDA notebook in `playground/auto_EDA.ipynb` covering all 13 steps from the pricing/bandit EDA framework the user provided. The dataset is a points-purchase offer allocation log: 10,000 sessions over 2 days (2024-09-02 to 2024-09-03) with 17 columns. The existing `playground/EDA.ipynb` covers only basics (grain, univariate distributions, a few quality assertions). `auto_EDA.ipynb` is currently a blank skeleton.

---

## Dataset Quick Reference

| Aspect | Detail |
|---|---|
| Grain | SESSION_KEY (unique); MEMBER_KEY has duplicates (9,466 unique → 534 multi-session members) |
| Action space | OFFER_RICHNESS_SERVED ∈ {0.40, 0.45, 0.50} (77% / 18% / 5%) |
| Conversion target | FLAG_TRANSACTION (binary, 13.7% positive rate) |
| Revenue target | POINTS_PURCHASED × PRICE_PER_POINT (only when FLAG_TRANSACTION=1) |
| Sentinel values | -1 = no history / no transaction for offer/price fields; 9999 = never purchased for days fields |
| Key note | Only 2 days of data — time-series steps are limited to hourly aggregations |

**Columns by role:**
- IDs / time: `SESSION_KEY`, `SESSION_DATE`, `MEMBER_KEY`
- Segment flags (known at decision time): `FLAG_FIRST_TIME_VISITOR`, `FLAG_FIRST_TIME_BUYER`
- Treatment: `OFFER_RICHNESS_SERVED`
- Post-decision outcomes: `FLAG_TRANSACTION`, `OFFER_RICHNESS_APPLIED`, `POINTS_PURCHASED`, `PRICE_PER_POINT`
- Context features (known at decision time): `CURRENT_BALANCE`, `DAYS_SINCE_LAST_PURCHASE_L12M`, `COUNT_TRANX_L12M`, `LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M`, `OFFER_RICHNESS_APPLIED_ON_LAST_PURCHASE_L12M`, `POINTS_PURCHASED_LAST_TRANX_L12M`, `DAYS_SINCE_LAST_VISIT_NO_PURCHASE`

---

## Files to Create / Modify

- **Write**: `playground/auto_EDA.ipynb` — main deliverable, full notebook
- **Edit**: `pyproject.toml` — add `seaborn` (heatmaps), `scipy` (KS test, stats) if not already transitive deps

---

## Notebook Structure (13 Sections)

### Section 0 — Setup
- Imports: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`, `statsmodels`, `sklearn`
- Constants: `SENTINEL_NEG1 = [-1]`, `SENTINEL_9999 = [9999]`, column role lists
- Load `../assignments/DATA_OFFER_ALLOCATION.csv` and parse `SESSION_DATE` as datetime
- Create `df_clean`: sentinel values recoded to `NaN` (used in missingness + collinearity steps)
- Create `df_tx`: transactions-only subset (`FLAG_TRANSACTION == 1`)
- Create revenue column: `REVENUE = POINTS_PURCHASED * PRICE_PER_POINT`

---

### Step 1 — Data Inventory and Decision Grain
**Goal:** Establish grain, time window, action space, and leakage-risk field classification.

1. Schema audit table: column name, dtype, nunique, sample values
2. Entity-time uniqueness: assert `SESSION_KEY` is unique; count members with >1 session
3. Action space: `OFFER_RICHNESS_SERVED.value_counts()`
4. Time grain: min/max of `SESSION_DATE`, row counts by date (2024-09-02 vs 2024-09-03)
5. Known-at-decision-time vs post-decision field inventory table (markdown + code)
6. Leakage warning: flag `OFFER_RICHNESS_APPLIED`, `POINTS_PURCHASED`, `PRICE_PER_POINT` as post-decision; they must not be used as model features

---

### Step 2 — Treatment and Reward Audit
**Goal:** Confirm action is logged cleanly; define reward; check arm coverage per segment.

1. Action share overall and by `FLAG_FIRST_TIME_VISITOR` × `FLAG_FIRST_TIME_BUYER` (4 segments) — bar charts
2. Arm availability heatmap (segment × offer arm, cell = share; flag cells <5% as weak-support)
3. Reward decomposition:
   - Conversion rate: `FLAG_TRANSACTION.mean()` overall and by arm
   - Revenue formula: `POINTS_PURCHASED × PRICE_PER_POINT`; mean revenue per session by arm
4. Margin check: `PRICE_PER_POINT` distribution per `OFFER_RICHNESS_SERVED` arm (are prices consistent?)
5. Note: `OFFER_RICHNESS_APPLIED = 0.0` in 25 rows (0.25%) — investigate this edge case

---

### Step 3 — Data Quality Checks
**Goal:** Detect impossible values, sentinel misuse, consistency violations, and duplicates.

1. Duplicate `SESSION_KEY` count (expect 0)
2. Sentinel scan table: for each column, count of -1 values and 9999 values
3. Transaction consistency assertions:
   - When `FLAG_TRANSACTION=1`: `POINTS_PURCHASED > 0`, `PRICE_PER_POINT > 0`, `OFFER_RICHNESS_APPLIED != -1`
   - When `FLAG_TRANSACTION=0`: `POINTS_PURCHASED == 0`, `PRICE_PER_POINT == -1` (note: the existing notebook's assertion on PRICE_PER_POINT=0 may fail — investigate)
4. Impossible economics: negative `CURRENT_BALANCE`, `POINTS_PURCHASED` > plausible ceiling
5. `OFFER_RICHNESS_APPLIED=0.0` deep-dive: cross-tab with `OFFER_RICHNESS_SERVED`, `FLAG_FIRST_TIME_BUYER`
6. Frequency tables for all categorical/flag columns

---

### Step 4 — Missingness Analysis
**Goal:** Understand whether missingness (encoded as sentinel) is MCAR, MAR, or MNAR.

1. Recode sentinels to NaN in `df_clean`; display missing rate per column as bar chart
2. Missing rate by segment: for each L12M column, compute missing% by `FLAG_FIRST_TIME_BUYER` and `OFFER_RICHNESS_SERVED`
3. Missingness-target correlation: point-biserial correlation between each missing indicator and `FLAG_TRANSACTION`
4. Logistic model: predict `is_new_customer` (`DAYS_SINCE_LAST_PURCHASE_L12M == 9999`) from `FLAG_FIRST_TIME_BUYER` and `FLAG_TRANSACTION` — confirms expected MNAR pattern
5. Thresholds applied: 72.6% missingness in L12M fields → missing-indicator strategy required; fields with <5% missing can use median/mean imputation

---

### Step 5 — Univariate Target and Action Distributions
**Goal:** Understand the shape of outcomes and arm imbalance.

1. `OFFER_RICHNESS_SERVED`: bar chart with percentages
2. `FLAG_TRANSACTION`: pie chart + bar by segment
3. `POINTS_PURCHASED` (conditional on purchase): histogram raw + log scale, ECDF, violin by arm — check for spike at multiples of 1000, long tail up to 60K
4. `REVENUE = POINTS_PURCHASED × PRICE_PER_POINT`: same plots
5. Zero mass visualization: bar showing zero vs positive sessions for POINTS_PURCHASED → motivates two-part model
6. Action imbalance note: 77% at 0.50 arm; 5% at 0.40 arm is approaching the 1–5% weak-support threshold for counterfactual learning

---

### Step 6 — Bivariate Price-Response Analysis
**Goal:** Describe the raw and adjusted relationship between offer richness and outcomes.

1. Conversion rate by `OFFER_RICHNESS_SERVED` — bar chart with 95% CI (Wilson interval)
2. Mean points purchased (conditional) by arm — bar chart
3. Mean revenue per session by arm — bar chart
4. Segment-stratified conversion rates: 2×2 grid (first-time visitor × first-time buyer) by arm
5. Residualized price-outcome: regress `FLAG_TRANSACTION` on segment dummies; plot residuals vs `OFFER_RICHNESS_SERVED`; compare raw vs adjusted curves side-by-side
6. Caution note: raw upward or flat price-conversion plots don't imply inelasticity — confounding must be addressed (Step 12)

---

### Step 7 — Multivariate Structure and Collinearity
**Goal:** Identify redundant predictors and unstable coefficient risk.

1. Correlation heatmap of all numeric columns from `df_clean` (after sentinel-to-NaN recode, among non-NA rows)
2. Highlight pairs with |ρ| > 0.8 as collinearity alerts
3. VIF table for candidate model features (exclude post-decision columns):
   - `FLAG_FIRST_TIME_VISITOR`, `FLAG_FIRST_TIME_BUYER`, `CURRENT_BALANCE`, `COUNT_TRANX_L12M`, `DAYS_SINCE_LAST_PURCHASE_L12M`, `POINTS_PURCHASED_LAST_TRANX_L12M`, `DAYS_SINCE_LAST_VISIT_NO_PURCHASE`
   - Use `statsmodels.stats.outliers_influence.variance_inflation_factor`
   - Flag VIF > 5 (caution) and VIF > 10 (severe)
4. Condition index from eigenvalues of the scaled feature matrix (flag 30–100 moderate, >100 severe)
5. Expected finding: `FLAG_FIRST_TIME_VISITOR` and `FLAG_FIRST_TIME_BUYER` may correlate with L12M missingness flags

---

### Step 8 — Outliers and Influential Points
**Goal:** Find extreme rows that could distort elasticity estimates — not to drop, but to understand.

1. Boxplots for `POINTS_PURCHASED`, `CURRENT_BALANCE`, `DAYS_SINCE_LAST_PURCHASE_L12M` on raw scale (using `df_clean`)
2. Same on log scale (after +1 shift)
3. IQR-based outlier count per column (rows beyond 1.5×IQR)
4. Fit a simple logistic regression (`FLAG_TRANSACTION ~ CURRENT_BALANCE + COUNT_TRANX_L12M + OFFER_RICHNESS_SERVED_encoded`); compute leverage (hat values) and Cook's distance
5. Flag rows with leverage > 2p/n and Cook's D > F(0.5, p, n-p)
6. Manual review of top 10 influential rows (display key columns)
7. Sensitivity note: do not trim without domain justification — large `POINTS_PURCHASED` may be whales, not data errors

---

### Step 9 — Seasonality and Autocorrelation
**Goal:** Detect intraday and interday demand cycles even within the 2-day window.

1. Aggregate to hourly buckets: session volume and conversion rate per hour
2. Line plots of session volume and conversion rate over the ~21-hour window (13:00 Sep 2 → 10:00 Sep 3)
3. ACF and PACF plots (`statsmodels.graphics.tsaplots.plot_acf`, `plot_pacf`) on hourly session volume and hourly conversion rate
4. Check for repeated structure at lags consistent with business patterns (lunch peak, evening peak)
5. Limitation note: with ~21 hourly observations, formal seasonality tests are underpowered; interpret patterns descriptively

---

### Step 10 — Stationarity and Regime Shifts
**Goal:** Check whether the 2-day series is stationary or shows level/variance shifts.

1. Plot rolling 1-hour mean and ±1 SD bands for conversion rate (window=3 hours)
2. ADF test (`statsmodels.tsa.stattools.adfuller`) on hourly conversion rate series
3. KPSS test (`statsmodels.tsa.stattools.kpss`) on same series
4. Day-over-day comparison table: mean conversion, mean points per session, volume — Sep 2 vs Sep 3
5. Two-sample KS test (`scipy.stats.ks_2samp`) for `CURRENT_BALANCE` and `POINTS_PURCHASED` between the two days
6. Note: formal stationarity tests need longer series; report p-values as indicative, not conclusive

---

### Step 11 — Functional Form, Linearity, and Heteroscedasticity
**Goal:** Determine if linear effects are plausible or if splines/GLMs are needed.

1. Partial regression (added variable) plots for `CURRENT_BALANCE`, `COUNT_TRANX_L12M`, `DAYS_SINCE_LAST_PURCHASE_L12M` against `FLAG_TRANSACTION` using `statsmodels.graphics.regressionplots.plot_partregress`
2. For OLS on `POINTS_PURCHASED` (conditional on purchase): residuals vs fitted, QQ plot
3. Breusch-Pagan test (`statsmodels.stats.diagnostic.het_breuschpagan`) on the OLS residuals — expect p < 0.05 given skewed target
4. Log-transform diagnostic: fit OLS on log(POINTS_PURCHASED+1) and compare residual plots — motivates Poisson/Tweedie or log-OLS for revenue model

---

### Step 12 — Confounding and Causal Signals
**Goal:** Assess whether offer assignment is confounded with customer quality.

1. Propensity model: multinomial logistic regression predicting `OFFER_RICHNESS_SERVED` arm from `FLAG_FIRST_TIME_VISITOR`, `FLAG_FIRST_TIME_BUYER`, `CURRENT_BALANCE`, `COUNT_TRANX_L12M`, `DAYS_SINCE_LAST_PURCHASE_L12M`
2. Plot propensity score distributions by arm — check for overlap; flag extreme propensities near 0 or 1
3. Balance table: mean of each feature by offer arm (raw vs propensity-weighted)
4. IPW-adjusted conversion rate by arm vs raw conversion rate — if sign or magnitude changes sharply, flag the raw result as unreliable
5. DAG sketch in markdown: `Segment → Offer → Transaction ← Segment`; note `Segment` is both a confounder and mediator
6. Placebo test: assign a random shuffle of offers and re-estimate — expected null result if method is sound

---

### Step 13 — Stability Over Time and Concept Drift
**Goal:** Detect distribution shifts between the two days that could undermine model transfer.

1. PSI calculation for key features between Sep 2 and Sep 3 (use equal-frequency bins):
   - `CURRENT_BALANCE`, `COUNT_TRANX_L12M`, `OFFER_RICHNESS_SERVED`, `FLAG_TRANSACTION`
   - PSI thresholds: ≤ 0.10 stable, 0.10–0.25 watch, > 0.25 investigate
2. Two-sample KS test (scipy.stats.ks_2samp) for each numeric feature between the two days
3. Feature importance drift: fit a binary classifier predicting "is this row from Sep 2?" — if AUC >> 0.5, features differ across days
4. Conversion rate and volume rolling chart with day boundary marked
5. Summary drift table: PSI and KS p-value per feature

---

## Dependencies to Add

Add to `pyproject.toml` if not already resolvable as transitive deps:
```
seaborn >= 0.13
scipy >= 1.14
```
`numpy` is already available (transitive dep of pandas/sklearn). `statsmodels` is already declared.

---

## Verification

1. Run `uv sync` to install any added deps
2. Run all cells in `auto_EDA.ipynb` top-to-bottom via `jupyter nbconvert --to notebook --execute`
3. No cell should raise an exception
4. Assertions in Step 3 must pass (or produce clear diagnostic output if they reveal real quality issues)
5. All 13 section headers should be visible in the notebook outline
