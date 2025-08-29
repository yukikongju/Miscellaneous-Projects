--- cost:
--- query to compare if organic + attributed networks
--- is similar to the values we get in Appsflyer Aggregate /
--- mixpanel

--- TODO:
with attributed_networks as (
    select
	date, network, platform, country

    from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
    where
	network in ('Apple Search Ads', 'Facebook Ads', 'googleadwords_int', 'snapchat_int', 'tiktokglobal_int', 'tatari_linear', 'tatari_streaming')
    group by date, network, platform, country

)
