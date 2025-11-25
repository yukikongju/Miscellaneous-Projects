-- Final Table for main networks from 2024 to 21/11/2025
-- cost: 4.35 GB
-- bq-results-20251122-002153-1763770970014.csv
SELECT
  date, network, platform, country, campaign_name, cost_usd, impressions, clicks, installs, mobile_trials, web_trials, trials
  , metrics_paid, backend_paid, paid
  , modeled_trial2paid
  , t2p_web, t2p_mobile
  , modeled_revenue, modeled_revenue_per_paid, modeled_paid, modeled_refund, modeled_refunded_amount,
FROM `relax-melodies-android.ua_dashboard_prod.final_table`
WHERE
  date >= '2024-01-01'
  and network in ('Facebook Ads', 'Apple Search Ads', 'googleadwords_int', 'snapchat_int', 'tiktokglobal_int', 'tatari_linear', 'tatari_streaming')
ORDER BY
  date, network, platform, country, campaign_name
