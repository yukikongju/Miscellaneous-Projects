with final_table as (
  SELECT
    time as date,
    utm_source as network,
    platform,
    country,
    campaign_id,
    utm_campaign as campaign_name,
    cost,
    impressions,
    clicks,
    installs,
    mobile_trials,
    web_trials,
    trials,
    0 AS metrics_paid,
    paid,
    revenue,
    refund,
    refunded_amount,
    rpp_mobile,
    page_landing,
    --  backend_trials,
    --  backend_paid,
    --  backend_revenue,
    --  backend_refund,
    --  backend_refunded_amount,
    --  agency,
    --  initial_revenue,
    --  need_modeling,
    --  modeled_revenue_per_trial,
    --  modeled_revenue_per_paid,
    --  modeled
    from `relax-melodies-android.ua_dashboard_prod.ad_spend_table`
), final_table_with_late_renewals as (
  select
    f.*,
    f.paid * r.`1-Year` * p.`1-Year` as renewal_rev_1Year,
    f.paid * r.`2-Years` * p.`2-Years` as renewal_rev_2Years,
    f.paid * r.`3-Years` * p.`3-Years` as renewal_rev_3Years,
    f.revenue + f.paid * r.`1-Year` * p.`1-Year` as late_revenue_year1,
    f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years` as late_revenue_year2,
    f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years` + f.paid * r.`3-Years` * p.`3-Years` as late_revenue_year3,
    case when f.cost_usd > 0
      then (f.revenue + f.paid * r.`1-Year` * p.`1-Year`) / f. cost_usd
      else null
    end as roas_year1,
    case when f.cost_usd > 0
      then (f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years`) / f. cost_usd
      else null
    end as roas_year2,
    case when f.cost_usd > 0
      then (f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years` + f.paid * r.`3-Years` * p.`3-Years`) / f. cost_usd
      else null
    end as roas_year3,
    case when f.cost_usd > 0
      then f.revenue / f.cost_usd
      else null
    end as roas
  from `relax-melodies-android.ua_dashboard_prod.final_table` f
  left join relax-melodies-android.late_conversions.renewal_rates_ma_imputed r
  on
    f.network = r.network
    and f.platform = r.platform
    and f.country = r.country_code
  left join `relax-melodies-android.late_conversions.renewal_proceeds_ma_imputed` p
  on
    f.network = p.network
    and f.platform = p.platform
    and f.country = p.country_code
)

select * from final_table_with_late_renewals
