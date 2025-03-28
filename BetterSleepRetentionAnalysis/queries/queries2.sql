declare start_date STRING DEFAULT '20240901';
-- declare end_date STRING DEFAULT '20241231';
declare end_date STRING DEFAULT '20240901';

# TODO: add user platform
# TODO: filter with event_param

with first_app_open as (
  select
    user_pseudo_id,
    max(device.operating_system) as os,
    PARSE_DATE('%Y%m%d', min(event_date)) as first_open_date,
    -- PARSE_DATE('%Y%m%d', MIN(event_date)) AS first_open_date_str
  from `relax-melodies-android.analytics_151587246.events_*`
  where
    _table_suffix >= start_date and _table_suffix <= end_date
    and user_pseudo_id is not null
    and event_name = 'first_open'
  group by user_pseudo_id
  -- Ensure only users who opened the app in the first month are included
  HAVING EXTRACT(YEAR FROM first_open_date) = EXTRACT(YEAR FROM PARSE_DATE('%Y%m%d', start_date))
    AND EXTRACT(MONTH FROM  first_open_date) = EXTRACT(MONTH FROM PARSE_DATE('%Y%m%d', start_date))
), event_sessions as (
  select
    user_pseudo_id,
    PARSE_DATE('%Y%m%d', event_date) as event_date,
    event_name,
    param.key as param_key,
    param.value.string_value as param_value,
  from `relax-melodies-android.analytics_151587246.events_*`,
    unnest(event_params) as param
  where
    _table_suffix >= start_date and _table_suffix <= end_date
    and user_pseudo_id is not null
    and event_name in (
      'listening_session',
      # --- player metrics
      'screen_content_playing',
      'toggle_favorite',
      'screen_recorder',
      'download_content',
      'click_explore',
      'screen_playlist_modal',
      'sleep_recorder_landed',
      'play_content',
      'pause_content',
      'play_next',
      'play_previous',
      # --- mixer metrics
      'create_favorite',
      'create_favorite_result',
      'add_to_playlist',
      'create_timer',
      'show_isochronic_dialog',
      'mixer_drawer_clear_all'
      )
), daily_sessions as (
  select
    f.user_pseudo_id,
    f.os,
    f.first_open_date,
    s.event_date,
    DATE_DIFF(s.event_date, f.first_open_date, DAY) as days_since_first_open,
    COUNTIF(event_name = 'listening_session') AS listening_sessions,
    COUNTIF(event_name = 'play_content') AS play_content,
    COUNTIF(event_name = 'pause_content') AS pause_content,
    COUNTIF(event_name = 'play_previous') AS play_previous,
    COUNTIF(event_name = 'play_next') AS play_next,
    COUNTIF(event_name = 'screen_content_playing') AS screen_content_playing,
    COUNTIF(event_name = 'screen_recorder') AS screen_recorder,
    COUNTIF(event_name = 'download_content') AS download_content,
    COUNTIF(event_name = 'screen_playlist_modal') AS screen_playlist_modal,
    COUNTIF(event_name = 'sleep_recorder_landed') AS sleep_recorder_landed,
    COUNTIF(event_name = 'create_timer') AS create_timer,

    COUNTIF(event_name = 'toggle_favorite') AS toggle_favorite,
    COUNTIF(event_name = 'create_favorite') AS create_favorite,
    COUNTIF(event_name = 'create_favorite_result') AS create_favorite_result,
    COUNTIF(event_name = 'click_explore') AS mixer_add_music,
    COUNTIF(event_name = 'show_isochronic_dialog') AS show_isochronic_dialog,
    COUNTIF(event_name = 'mixer_drawer_clear_all') AS mixer_drawer_clear_all,
    CASE
      WHEN DATE_DIFF(s.event_date, f.first_open_date, DAY)  BETWEEN 0 AND 1 THEN 'Day 1'
      WHEN DATE_DIFF(s.event_date, f.first_open_date, DAY)  BETWEEN 1 AND 2 THEN 'Day 2'
      WHEN DATE_DIFF(s.event_date, f.first_open_date, DAY)  BETWEEN 2 AND 3 THEN 'Day 3'
      WHEN DATE_DIFF(s.event_date, f.first_open_date, DAY)  BETWEEN 4 AND 7 THEN 'Week 1'
      WHEN DATE_DIFF(s.event_date, f.first_open_date, DAY)  BETWEEN 8 AND 14 THEN 'Week 2'
      WHEN DATE_DIFF(s.event_date, f.first_open_date, DAY)  BETWEEN 15 AND 21 THEN 'Week 3'
      WHEN DATE_DIFF(s.event_date, f.first_open_date, DAY)  BETWEEN 22 AND 31 THEN 'Week 4'
      WHEN DATE_DIFF(s.event_date, f.first_open_date, DAY)  BETWEEN 32 AND 59 THEN 'Month 2'
      WHEN DATE_DIFF(s.event_date, f.first_open_date, DAY)  BETWEEN 60 AND 90 THEN 'Month 3'
      ELSE 'Month 3+'
    END as time_bucket
  from first_app_open f
  join event_sessions s
  on f.user_pseudo_id = s.user_pseudo_id
    and s.event_date >= f.first_open_date
  group by f.user_pseudo_id, f.first_open_date, f.os, s.event_date
)

select
  *
from daily_sessions
-- group by user_pseudo_id, time_bucket




--  NOTE:

--  declare start_date STRING DEFAULT '20240901';
--  declare end_date STRING DEFAULT '20240901';

--  with event_sessions as (
--    select
--      user_pseudo_id,
--      PARSE_DATE('%Y%m%d', event_date) as event_date,
--      COUNTIF(event_name = 'click_explore'
--              AND EXISTS (
--                SELECT 1 FROM UNNEST(event_params) as param
--                WHERE param.key='location' AND param.value.string_value = 'mixer_drawer'
--              )
--        AND EXISTS (
--            SELECT 1 FROM UNNEST(event_params) as param
--            WHERE param.key='content' AND param.value.string_value = 'music'
--        )
--            ) AS mixer_add_music,

--    from `relax-melodies-android.analytics_151587246.events_*`,
--      unnest(event_params) as param
--    where
--      _table_suffix >= start_date and _table_suffix <= end_date
--      and user_pseudo_id is not null
--      and event_name in (
--        'listening_session',
--        # --- player metrics
--        'screen_content_playing',
--        'toggle_favorite',
--        'play_content',
--        'pause_content',
--        'screen_recorder',
--        'download_content',
--        'click_explore',
--        'screen_playlist_modal',
--        'sleep_recorder_landed',
--        'play_next',
--        'play_previous',
--        'click_explore',
--        # --- mixer metrics
--        'create_favorite',
--        'add_to_playlist',
--        'create_timer'
--        )
--    group by user_pseudo_id, event_date
--  )

--  select * from event_sessions
