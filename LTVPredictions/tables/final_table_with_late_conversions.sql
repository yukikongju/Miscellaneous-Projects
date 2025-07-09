create table `relax-melodies-android.late_conversions.final_table_with_late_conversions` (
    date datetime,
    network string,
    platform string,
    campaign_id string,
    campaign_name string,
    cost_cad float,
    cost_usd float,
    metrics_paid float,
    backend_paid float,
    backend_revenue float,
    modeled_paid,
    modeled_revenue,

    --- estimated
    late_renewal_revenue_modeled, -- revenue from year1 through year5
    late_renewal_revenue_metrics, -- revenue from year1 through year5
    late_renewal_revenue_backend, -- revenue from year1 through year5
)







    --  impressions float,
    --  clicks float,
    --  installs float,
    --  mobile_trials float,
    --  web_trials float,
    --  trials float,
    --  backend_refund float,
    --  backend_refunded_amount float,
    --  agency string,
    --  initial_revenue float,
    --  need_modeling boolean,
    --  t2p_web float,
    --  t2p_mobile float,
    --  rpp_mobile float,
    --  modeled_trial2paid float,
    --  modeled_refund,
    --  modeled_refunded_amount,
    --  paid,
    --  revenue,
    --  refund,
    --  refunded_amount,
    --  modeled_trial2paid float,
