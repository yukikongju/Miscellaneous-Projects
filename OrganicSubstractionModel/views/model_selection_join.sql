--  `relax-melodies-android.ua_transform_prod.trial2paid_selected_model`
--- TODO: select default t2p for misc network from Appsflyer
with geobydate_t2p as (
    select
	g.*
    from `relax-melodies-android.ua_transform_prod.trial2paid_geobydate_unique` g
    left join `relax-melodies-android.ua_transform_prod.model_selection` m
    on g.platform = m.platform
	and g.network = m.network
	and g.install_date between m.start_date and m.end_date
    where
	m.model_source = 'Geobydate'
), internal_t2p as (
    select
	g.*
    from `relax-melodies-android.ua_transform_prod.trial2paid_unique` g
    left join `relax-melodies-android.ua_transform_prod.model_selection` m
    on g.platform = m.platform
	and g.network = m.network
	and g.install_date between m.start_date and m.end_date
    where
	m.model_source = 'Internal'
)
--  , default_t2p as (
--      select

--      from `relax-melodies-android.ua_transform_prod.trial2paid_geobydate_model`

all_t2p as (
    select
        install_date,
        network, platform, country, sub_continent, continent,
        trial, paid, revenue, aggregation_group, valid_name,
        rolling_total_trial, rolling_total_paid, modeled_trial2paid,
        modeled_revenue_per_trial, modeled_revenue_per_paid, modeled_refund,
        modeled_refunded_amount, refund, refunded_amount,
        modeled_refunded_amount_per_paid, modeled_paid2refund
    from geobydate_t2p
    union all (
	select
        install_date,
        network, platform, country, sub_continent, continent,
        trial, paid, revenue, aggregation_group, valid_name,
        rolling_total_trial, rolling_total_paid, modeled_trial2paid,
        modeled_revenue_per_trial, modeled_revenue_per_paid, modeled_refund,
        modeled_refunded_amount, refund, refunded_amount,
        modeled_refunded_amount_per_paid, modeled_paid2refund
    from internal_t2p
    )
)

select * from all_t2p
