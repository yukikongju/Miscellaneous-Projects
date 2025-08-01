--- query to compare with mixpanel
--- mixpanel: https://mixpanel.com/s/4bX4d9
select
    *
from `relax-melodies-android.late_conversions.latest_renewal`
where
  country_code = 'US'
  and network in ('Apple Search Ads', 'tiktokglobal_int', 'googleadwords_int', 'snapchat_int', 'Facebook Ads')
  and renewal_bucket = '1-Year'
order by
  network, platform, renewal_bucket
