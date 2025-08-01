--- cost: 1.62 GB (final table is unpartitioned)
declare start_date datetime default '2025-06-01';
declare end_date datetime default '2025-06-10';

select
  f.*,
  r.`1-Year` as renew_rate_year1,
  p.`1-Year` as renew_proceeds_year1,
  r.`2-Years` as renew_rate_year2,
  p.`2-Years` as renew_proceeds_year2,
  f.paid * r.`1-Year` * p.`1-Year` as renewal_rev_1Year,
  f.paid * r.`2-Years` * p.`2-Years` as renewal_rev_2Years,
  f.paid * r.`3-Years` * p.`3-Years` as renewal_rev_3Years,
  f.revenue + f.paid * r.`1-Year` * p.`1-Year` as late_revenue_year1,
  f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years` as late_revenue_year2,
  -- f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years` + f.paid * r.`3-Years` * p.`3-Years` as late_revenue_year3,
  case when f.cost_usd > 0
    then (f.revenue + f.paid * r.`1-Year` * p.`1-Year`) / f. cost_usd
    else null
  end as roas_year1,
  case when f.cost_usd > 0
    then (f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years`) / f.cost_usd
    else null
  end as roas_year2,
  -- case when f.cost_usd > 0
  --   then (f.revenue + f.paid * r.`1-Year` * p.`1-Year` + f.paid * r.`2-Years` * p.`2-Years` + f.paid * r.`3-Years` * p.`3-Years`) / f.cost_usd
  --   else null
  -- end as roas_year3,
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
where
  f.date between start_date and end_date
  and f.paid > 20
  and f.cost_usd > 50
  and f.country = 'US'
  and f.network in ('Apple Search Ads', 'tiktokglobal_int', 'googleadsword_int', 'Facebook Ads')
