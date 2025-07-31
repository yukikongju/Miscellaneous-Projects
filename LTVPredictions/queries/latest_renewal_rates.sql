with cohorts_ranked as (
    select
    *,
    rank() over (partition by network, platform, country_code, renewal_bucket order by paid_year_month desc) as rn
    from `relax-melodies-android.late_conversions.mature_renewal_cohorts`
)

select
    network,
    platform,
    country_code,
    renewal_bucket,
    num_renewals,
    num_paid,
    paid_proceeds,
    renewal_proceeds,
    renewal_percentage
from cohorts_ranked
where rn = 1
