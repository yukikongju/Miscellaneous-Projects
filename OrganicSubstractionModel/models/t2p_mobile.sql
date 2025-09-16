--- DOCS
--- https://cloud.google.com/bigquery/docs/linear-regression-tutorial


--- MODEL TRAINING + GLOBAL EXPLAIN
create or replace model test_t2p.t2p_model
options(input_label_cols=['t2p'], model_type='linear_reg', enable_global_explain = TRUE) as

with dataset as (
  select
    install_date,
    network,
    platform,
    country,
    cast(extract(dayofweek from install_date) as string) as day_of_week,
    cast(extract(isoweek from install_date) as string) as week_of_year,
    trial,
    paid,
    case when trial > 0
      then paid / trial
      else null
    end as t2p
  from `relax-melodies-android.ua_transform_prod.trial_and_paid_hau_utm_internal_aggregate`
), clean as (
  select *
  from dataset
  where t2p is not null
)

select * from clean

--- TRAINING METRICS: 89.9MB
select
*
from ml.evaluate(model `test_t2p.t2p_model`)

--- TESTING METRICS: 89.9MB
select
*
from ml.evaluate(model `test_t2p.t2p_model`,
(
  select
    install_date,
    network,
    platform,
    country,
    cast(extract(dayofweek from install_date) as string) as day_of_week,
    cast(extract(isoweek from install_date) as string) as week_of_year,
    trial,
    paid,
    case when trial > 0
      then paid / trial
      else null
    end as t2p

  from `relax-melodies-android.ua_transform_prod.trial_and_paid_hau_utm_internal_aggregate`
))

--- PREDICTION: 98.89MB
select
*
from ML.PREDICT(model `test_t2p.t2p_model`,
(
  select
    install_date,
    network,
    platform,
    country,
    cast(extract(dayofweek from install_date) as string) as day_of_week,
    cast(extract(isoweek from install_date) as string) as week_of_year,
    trial,
    paid,
    case when trial > 0
      then paid / trial
      else null
    end as t2p
  from  `relax-melodies-android.ua_transform_prod.trial_and_paid_hau_utm_internal_aggregate`
  where
    country = 'US'
));


--- EXPLAIN PREDICT
select
*
from ML.EXPLAIN_PREDICT(model `test_t2p.t2p_model`,
(
  select
    install_date,
    network,
    platform,
    country,
    cast(extract(dayofweek from install_date) as string) as day_of_week,
    cast(extract(isoweek from install_date) as string) as week_of_year,
    trial,
    paid,
    case when trial > 0
      then paid / trial
      else null
    end as t2p
  from  `relax-melodies-android.ua_transform_prod.trial_and_paid_hau_utm_internal_aggregate`
  where
    country = 'US'
));


--- GLOBAL EXPLANATION
select
*
from ML.GLOBAL_EXPLAIN(model `test_t2p.t2p_model`)
