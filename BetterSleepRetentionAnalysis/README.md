# BetterSleep Retention Analysis

**Goal**

We want to understand which user behaviors are correlated with retention, trials,
paid, refund, cancelation. Previously, these analyses were made on MixPanel, but
that method is not effective because the data is so dispersed and we cannot
re-use the solution for similar processes.

**How**

We want to have an `Events` dimensions table such that we have the following
columns:
- user_id, user_pseudo_id, country, platform: STRING
- from Subscription Table: has_trial, has_paid, has_canceled, has_refund: Bool
- list of app events every month
    * player/mixer events
    * tracker events
    * carousel events
    * routine events
    * chronotype events:

`Events` table:

| user_id | time_bucket | Events: listening_session | ... |
| 1       | day 1       |                           |     |
| 1       | day 3       |                           |     |
| 1       | week 1      |                           |     |
| 1       | week 2      |                           |     |
| 1       | month 1     |                           |     |
| 1       | month 2     |                           |     |
| 1       | month 3     |                           |     |
| 1       | month 3+    |                           |     |

`Subscriptions` table:

| user_id | has_trial | has_paid | has_canceled | has_refund |


**Use cases**

- Correlation Analysis
    * how does each event correlates to listening_session, long term retention,
      paid, cancellation, ...
    * `Corr(Xi, Xj) = \frac{Cov(Xi, Xj)}{\sigma_1 * \sigma_2}`
    * segmenting user by
	+ engagement level: low, medium, high
	+ product level: guided listening / player / mixer / tracker / both
	+ old vs new users:
	+ paid vs free users
- Understanding users behavior
    * cancel vs not
    * tracker retention
    * player retention
- User segmentation
    * group user with similar behavior together => guided content embedding recommendation based on similar users
- Push notification for users that will churn


## Resources
