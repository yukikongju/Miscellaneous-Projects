create or replace view `relax-melodies-android.organics.organic_estimation_all` as (
    select
	*
    from `relax-melodies-android.organics.organic_estimation_android`
    union all (
	select
	*
	from `relax-melodies-android.organics.organic_estimation_ios`
    )

)
