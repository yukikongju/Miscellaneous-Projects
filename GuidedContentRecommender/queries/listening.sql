declare start_date STRING DEFAULT '20250224';
declare end_date STRING DEFAULT '20250224';

SELECT *
  FROM `relax-melodies-android.analytics_151587246.events_*`,
    unnest(event_params) as param
where
  _table_suffix = start_date and _table_suffix = end_date
  and event_name in ('listening') # , 'toggle_favorite', 'create_favorite_result'
  and (param.key = 'guided_content')
  and (param.value.string_value is not null and param.value.string_value != 'none')
limit 100
