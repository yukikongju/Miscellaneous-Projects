--- cost: 1.65 GB
with final_table as (
  SELECT
    date,
    network,
    platform,
    country,
    campaign_id,
    campaign_name,
    cost_cad,
    cost_usd,
    impressions,
    clicks,
    installs,
    mobile_trials,
    web_trials,
    trials,
    0 AS metrics_paid,
    backend_trials,
    backend_paid,
    backend_revenue,
    backend_refund,
    backend_refunded_amount,
    agency,
    0 AS initial_revenue,
    need_modeling,
    CASE
      WHEN trials = 0 THEN 0
      ELSE ((web_trials/trials)*t2p_web + (mobile_trials/trials) * t2p_mobile)
  END
    AS modeled_trial2paid,
    t2p_web,
    t2p_mobile,
    rpp_mobile,
    modeled_revenue_per_trial,
    modeled_revenue_per_paid,
    CASE
      WHEN trials = 0 THEN 0
      ELSE trials * ((web_trials/trials)*t2p_web + (mobile_trials/trials) * t2p_mobile)
  END
    AS modeled_paid,
    CASE
      WHEN trials = 0 THEN 0
      ELSE trials * ((web_trials/trials)*t2p_web + (mobile_trials/trials) * t2p_mobile) * modeled_revenue_per_paid
  END
    AS modeled_revenue,
    CASE
      WHEN trials = 0 THEN 0
      ELSE trials * ((web_trials/trials)*t2p_web + (mobile_trials/trials) * t2p_mobile) * modeled_paid2refund
  END
    AS modeled_refund,
    CASE
      WHEN trials = 0 THEN 0
      ELSE trials * ((web_trials/trials)*t2p_web + (mobile_trials/trials) * t2p_mobile) * modeled_refunded_amount_per_paid
  END
    AS modeled_refunded_amount,
    FALSE AS recent_date,
    paid,
    page_landing,
    revenues AS revenue,
    0 AS diff_revenues,
    0 AS refund,
    0 AS refunded_amount
  FROM
    `relax-melodies-android.ua_dashboard_prod.final_table_web`
  UNION ALL
  SELECT
    date,
    network,
    platform,
    country,
    campaign_id,
    campaign_name,
    cost_cad,
    cost_usd,
    impressions,
    clicks,
    installs,
    mobile_trials,
    web_trials,
    trials,
    metrics_paid,
    backend_trials,
    backend_paid,
    backend_revenue,
    backend_refund,
    backend_refunded_amount,
    agency,
    initial_revenue,
    need_modeling,
    modeled_trial2paid,
    0 AS t2p_web,
    0 AS t2p_mobile,
    0 AS rpp_mobile,
    modeled_revenue_per_trial,
    modeled_revenue_per_paid,
    modeled_paid,
    modeled_revenue,
    modeled_refund,
    modeled_refunded_amount,
    recent_date,
    paid,
    0 AS page_landing,
    revenue,
    diff_revenues,
    refund,
    refunded_amount
  FROM
    `relax-melodies-android.ua_dashboard_prod.final_table_mobile`
  WHERE
    platform <> "web"
), final_table_with_late_renewals as (
  select
    f.*,
    r.`1-Year` as renewal_rate_year1,
    r.`2-Years` as renewal_rate_year2,
    r.`3-Years` as renewal_rate_year3,
    p.`1-Year` as renewal_proceeds_year1,
    p.`2-Years` as renewal_proceeds_year2,
    p.`3-Years` as renewal_proceeds_year3,
    --  f.paid * r.`1-Year` * p.`1-Year` as renewal_rev_1Year,
    --  f.paid * r.`2-Years` * p.`2-Years` as renewal_rev_2Years,
    --  f.paid * r.`3-Years` * p.`3-Years` as renewal_rev_3Years,
    --  f.revenue + f.paid * r.`1-Year` * p.`1-Year` as late_revenue_year1,
    --  f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years` as late_revenue_year2,
    --  f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years` + f.paid * r.`3-Years` * p.`3-Years` as late_revenue_year3,
    --  case when f.cost_usd > 0
    --    then (f.revenue + f.paid * r.`1-Year` * p.`1-Year`) / f. cost_usd
    --    else null
    --  end as roas_year1,
    --  case when f.cost_usd > 0
    --    then (f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years`) / f. cost_usd
    --    else null
    --  end as roas_year2,
    --  case when f.cost_usd > 0
    --    then (f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years` + f.paid * r.`3-Years` * p.`3-Years`) / f. cost_usd
    --    else null
    --  end as roas_year3,
    --  case when f.cost_usd > 0
    --    then f.revenue / f.cost_usd
    --    else null
    --  end as roas
  from final_table f
  left join `relax-melodies-android.late_conversions.monthly_renewal_rates` r
  on
    f.date = datetime(r.year_month)
    and f.network = r.network
    and f.platform = r.platform
    and f.country = r.country_code
  left join `relax-melodies-android.late_conversions.monthly_renewal_proceeds` p
  on
    f.date = datetime(p.year_month)
    and f.network = p.network
    and f.platform = p.platform
    and f.country = p.country_code
)

select * from final_table_with_late_renewals
-- where
--   date = '2025-08-01'
--   and country = 'US'
--   and network in ('Apple Search Ads', 'Facebook Ads', 'tatari_linear', 'tatari_streaming', 'tiktokglobal_int', 'snapchat_int', 'googleadwords_int')
