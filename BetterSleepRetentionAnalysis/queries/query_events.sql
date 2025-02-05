declare start_date STRING DEFAULT '20240901';
-- declare end_date STRING DEFAULT '20241231';
declare end_date STRING DEFAULT '20240931';

with first_app_open as (
  select
    user_pseudo_id,
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
  from `relax-melodies-android.analytics_151587246.events_*`
  where
    _table_suffix >= start_date and _table_suffix <= end_date
    and user_pseudo_id is not null
    and event_name in (
      'listening_session',
      # --- player metrics
      'toggle_favorite',
      'play_content',
      'screen_recorder',
      'download_content',
      'click_explore',
      'screen_playlist_modal',
      'play_next',
      'play_previous',
      # --- mixer metrics
      'create_favorite',
      'add_to_playlist',
      'create_timer'
      )
), daily_sessions as (
  select
    f.user_pseudo_id,
    f.first_open_date,
    s.event_date,
    DATE_DIFF(s.event_date, f.first_open_date, DAY) as days_since_first_open,
    COUNTIF(event_name = 'listening_session') AS listening_sessions,
    COUNTIF(event_name = 'play_content') AS play_content,
    COUNTIF(event_name = 'toggle_favorite') AS toggle_favorite,
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
  group by f.user_pseudo_id, f.first_open_date, s.event_date
)

select
  *
from daily_sessions
-- group by user_pseudo_id, time_bucket
