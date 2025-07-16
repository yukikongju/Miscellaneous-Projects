with cohort_average as (
  select
    avg(`1-Year`) as `1-Year`,
    avg(`2-Years`) as `2-Years`,
    avg(`3-Years`) as `3-Years`,
    avg(`4-Years`) as `4-Years`,
    avg(`5-Years`) as `5-Years`,
  from `relax-melodies-android.late_conversions.matured_renewal_rates_cohorted`
), imputed as (
  select


  from cohort_average as c

)

select * from cohort_average
