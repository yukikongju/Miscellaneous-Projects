# Adset / Creative Performance

**Context**

A given campaign can run several ads, and each ads can run up to 50 creatives.
In order to better understand which creatives perform the best and optimize
the creative that are being produced, it would be ideal if we could
understand the performance at the adset and creative level, that is:
- CPI
- T2P
- ROAS

**To figure out**

- Where to find the data?
    * Appsflyer Raw Data has adset at the user level, but still unclear if
      we can link these user_id to our internal events data
    *

The Adset reports:
- Appsflyer Raw Report - Install Report
    * attributed touch time and install time with adset for each user
    * ex: `~/Downloads/id314498713_installs_2025-09-16_2025-09-23_America_Toronto.csv`
- Appsflyer Raw Report - In-App Report
    * Event at user-level; attributed touch time and install date
    * ex: `id314498713_in-app-events_2025-03-11_2025-03-18_America_Toronto.csv
`
The Creative Reports:
- Facebook => ads creative available, but not being pulled
    * can be done by specifying `level='adset/ad/'` with `account.get_insights(params=params_ad, fields=...)`
- Google
    * can be done by defining "query_adgroup" and "query_creatives"
- Tatari
    * already in the report
- ASA
- TikTok
    * No access for some reason
- Snapchat



## Docs

- [Facebook API](https://developers.facebook.com/docs/marketing-api/conversions-api/using-the-api)
- [Facebook Python Business SDK](https://github.com/facebook/facebook-python-business-sdk)
