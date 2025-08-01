declare start_date date default '2025-01-01';
declare end_date date default '2025-06-30';

select
  network_attribution,
  extract(year from hau_date) as year,
  extract(month from hau_date) as month,
  count(*) as count,
from `relax-melodies-android.late_conversions.users_network_attribution`
where
  hau_date between start_date and end_date
  and network_attribution in ('Apple Search Ads', 'tiktokglobal_int', 'googleadwords_int', 'snapchat_int', 'Facebook Ads')
group by
  network_attribution, extract(year from hau_date), extract(month from hau_date)
order by
  network_attribution, extract(year from hau_date), extract(month from hau_date)
