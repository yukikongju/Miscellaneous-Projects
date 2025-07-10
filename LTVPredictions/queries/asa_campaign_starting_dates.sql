select
  network, platform, campaignid, campaignname, countryorregion, min(date)
from `relax-melodies-android.ua_extract_prod.apple_search_ads_complete_metrics`
group by network, platform, countryorregion, campaignid, campaignname
