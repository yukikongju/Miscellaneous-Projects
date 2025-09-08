declare network_list default array<string> ['Apple Search Ads', 'Facebook Ads', 'googleadwords_int', 'snapchat_int', 'tiktokglobal_int'];
declare country_list default array<string> ['US', 'CA', 'UK', 'AU'];
declare start_date date default "2025-01-01";
declare end_date date default "2025-09-05";

with geobydate_t2p as (
    select
	install_date, network, platform, country,
	modeled_trial2paid,
	modeled_revenue_per_trial,
	modeled_revenue_per_paid,
	modeled_refund,
	modeled_refunded_amount,
	modeled_refunded_amount_per_paid,
	modeled_paid2refund,
    from `relax-melodies-android.ua_transform_prod.trial2paid_geobydate_unique`
), internal_t2p as (
    select
	install_date, network, platform, country,
	modeled_trial2paid,
	modeled_revenue_per_trial,
	modeled_revenue_per_paid,
	modeled_refund,
	modeled_refunded_amount,
	modeled_refunded_amount_per_paid,
	modeled_paid2refund,
    from `relax-melodies-android.ua_transform_prod.trial2paid_unique`
), t2p_comparison as (
    select
	i.install_date, i.network, i.platform, i.country,
	i.modeled_trial2paid as trial2paid_internal,
	g.modeled_trial2paid as trial2paid_geobydate,
	i.modeled_revenue_per_trial as rev_per_trial_internal,
	g.modeled_revenue_per_trial as rev_per_trial_geobydate,
	i.modeled_revenue_per_paid as rev_per_paid_internal,
	g.modeled_revenue_per_paid as rev_per_paid_geobydate,
	i.modeled_refund as refund_internal,
	g.modeled_refund as refund_geobydate,
	i.modeled_refunded_amount as refunded_amount_internal,
	g.modeled_refunded_amount as refunded_amount_geobydate,
	i.modeled_refunded_amount_per_paid as refunded_amount_per_paid_internal,
	g.modeled_refunded_amount_per_paid as refunded_amount_per_paid_geobydate,
	i.modeled_paid2refund as paid2refund_internal,
	g.modeled_paid2refund as paid2refund_geobydate,
    from
	internal_t2p as i
    full outer join geobydate_t2p g
    on
	i.install_date = g.install_date
	and i.platform = g.platform
	and i.network = g.network
	and i.country = g.country
)

select * from t2p_comparison
where
	install_date between start_date and end_date
	and network in unnest(network_list)
	and country in unnest(country_list)
order by network, platform, country, install_date;
