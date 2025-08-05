--- cost: 2.35TB
WITH event_data AS (
  SELECT *
  FROM `relax-melodies-android.analytics_151587246.events_*`
  WHERE _table_suffix >= "20250507"
    AND _table_suffix < "20250616"
    AND user_id IS NOT NULL
    AND device.operating_system = "Android"
), user_info as (
SELECT
  user_id
  ,device.operating_system as platform
  ,max(device.language) as app_language
  ,max(geo.continent) as continent
  ,max(geo.country) as country
  ,max(geo.region) as region
  ,max(geo.city) as city
  ,COALESCE(SAFE_CAST(Max(case when prop.key="RMA_first_open_s" Then prop.value.string_value End) as INT64),0) as first_open_s
  ,COALESCE(SAFE_CAST(Max(case when prop.key='RMA_days_old' Then prop.value.string_value End) as INT64),0) as days_old
  ,Max(case when prop.key="RMA_app_lang" Then prop.value.string_value End) as app_lang
  ,Max(case when prop.key="RMA_onboarding_completed" Then prop.value.string_value End) as onboarding_complete
  ,Max(case when prop.key="RMA_utm_channel" Then prop.value.string_value End) as utm_channel
  ,Max(case when prop.key="RMA_utm_medium" Then prop.value.string_value End) as utm_medium
  ,Max(case when prop.key="RMA_utm_source" Then prop.value.string_value End) as utm_source
  ,Max(case when event_name = "edit_alarm" then "true" else "false" End) as edit_alarm
  ,Max(case when event_name = "onboarding_email_captured" then "true" else "false" End) as onboarding_email_captured
  ,Max(case when event_name = "sleepgoal_confirm" then "true" else "false" End) as sleepgoal_confirmed
  ,Max(case when event_name = "create_account_confirmed" then "true" else "false" End) as account_confirmed
  ,Max(case when event_name = "sleep_graph_loaded" then "true" else "false" End) as sleep_graph_loaded
  ,Max(case when prop.key = "RMA_microphone_enabled" then prop.value.string_value End) as mic_enabled
  ,Max(case when prop.key = "RMA_notif_enabled" then prop.value.string_value End) as notif_enabled
  ,Max(case when prop.key = "RMA_paired_watch" then prop.value.string_value End) as paired_watch
  ,Max(case when prop.key = "RMA_has_reviewed_app" then prop.value.string_value End) as reviewed_app
  --,Max(case when CAST(REGEXP_EXTRACT(device.mobile_os_hardware_model, r'(\d+),') AS INT64)=16 then "true" else "false" End) as newer_device
  ,Max(device.mobile_os_hardware_model) as newer_device
FROM event_data ,
UNNEST(user_properties) as prop
WHERE event_timestamp/1000000 - COALESCE(SAFE_CAST((SELECT value.string_value FROM UNNEST(user_properties) WHERE key = 'RMA_first_open_s') as INT64),0)  <= 86400
group by
user_id
,platform
-- ,app_language
),
question_reached as (
  SELECT
    user_id,
    max(case when params.value.string_value='question_satisfied'then 1 else 0 end) as question_satisfied_reached,
    max(case when params.value.string_value='question_age'then 1 else 0 end) as question_age_reached,
    max(case when params.value.string_value='question_gender'then 1 else 0 end) as question_gender_reached,
    max(case when params.value.string_value='question_current_hours_of_sleep'then 1 else 0 end) as question_current_hours_of_sleep_reached,
    max(case when params.value.string_value in ('question_goals','goals') then 1 else 0 end) as question_goals_reached,
    max(case when params.value.string_value='topics'then 1 else 0 end) as question_topics_reached,
    max(case when params.value.string_value='soundToDetect'then 1 else 0 end) as question_soundToDetect_reached,
    max(case when params.value.string_value in ('hearaboutus','hearAboutUs') then 1 else 0 end) as question_HAU_reached,
    max(case when params.value.string_value='sleepIssues'then 1 else 0 end) as question_sleepIssues_reached,
    max(case when params.value.string_value='content'then 1 else 0 end) as question_content_reached,
FROM
    event_data,
    UNNEST(event_params) AS params
WHERE
    event_name = "screen_onboarding_flexible_slideshow" or event_name='screen_onboarding_question'
    AND event_timestamp/1000000 - COALESCE(SAFE_CAST((SELECT value.string_value FROM UNNEST(user_properties) WHERE key = 'RMA_first_open_s') as INT64),0)  <= 86400
  group by user_id),
qid AS (
    SELECT
        user_id,
        event_timestamp,
        event_name,
        params.key AS question_key,
        params.value.string_value AS question_value
    FROM
        event_data,
        UNNEST(event_params) AS params
    WHERE
        event_name = "answer_question"
        AND (params.key = "question_id" or params.key="flexible_slide_show_id")
        AND event_timestamp/1000000 - COALESCE(SAFE_CAST((SELECT value.string_value FROM UNNEST(user_properties) WHERE key = 'RMA_first_open_s') as INT64),0)  <= 86400
),
answers AS (
    SELECT
        user_id,
        event_timestamp,
        event_name,
        params.key AS answer_key,
        params.value.string_value AS answer_value
    FROM
        event_data,
        UNNEST(event_params) AS params
    WHERE
        event_name = "answer_question"
        AND (params.key = "answer" OR params.key = "answers")
),
combined AS (
    SELECT
        qid.user_id,
        MAX(CASE WHEN qid.question_value = 'goals' or qid.question_value='question_goals' THEN answers.answer_value ELSE NULL END) AS goals,
        MAX(CASE WHEN qid.question_value in ('hearaboutus', 'hearAboutUs') THEN answers.answer_value ELSE NULL END) AS heard_about_us,
        MAX(CASE WHEN qid.question_value = 'sleepIssues' or qid.question_value='sleepissues' THEN answers.answer_value ELSE NULL END) AS sleep_issues,
        MAX(CASE WHEN qid.question_value = 'content' THEN answers.answer_value ELSE NULL END) AS content,
        MAX(CASE WHEN qid.question_value = 'question_satisfied' THEN answers.answer_value ELSE NULL END) AS question_satisfied,
        MAX(CASE WHEN qid.question_value = 'question_current_hours_of_sleep' THEN answers.answer_value ELSE NULL END) AS question_current_hours_of_sleep,
        MAX(CASE WHEN qid.question_value = 'topics' THEN answers.answer_value ELSE NULL END) AS topics,
        MAX(CASE WHEN qid.question_value = 'question_gender' THEN answers.answer_value ELSE NULL END) AS gender,
        MAX(CASE WHEN qid.question_value = 'question_age' THEN answers.answer_value ELSE NULL END) AS age,
        MAX(CASE WHEN qid.question_value = 'soundToDetect' THEN answers.answer_value ELSE NULL END) AS question_soundToDetect

    FROM
        qid
    LEFT JOIN
        answers
    ON
        qid.user_id = answers.user_id
        AND qid.event_timestamp = answers.event_timestamp
    GROUP BY
        qid.user_id
),
event_counts_within_24h AS (
    SELECT
        user_id,
        sum(case when event_name = 'sleep_recorder_landed' then 1 else 0 end) as sleep_recorded_count,
        sum(case when event_name = 'start_routine' then 1 else 0 end) as start_routine_count,
        sum(case when event_name = 'screen_routine' then 1 else 0 end) as screen_routine_count,
        sum(case when event_name = 'onboarding_skip' then 1 else 0 end) as onboarding_skip_count,
        sum(case when event_name = 'listening' then 1 else 0 end) as listening_count,
    FROM
        event_data
    WHERE
        event_timestamp/1000000 - COALESCE(SAFE_CAST((SELECT value.string_value FROM UNNEST(user_properties) WHERE key = 'RMA_first_open_s') as INT64),0)  <= 86400
    GROUP BY
        user_id
),
trial_upgrade_ref AS (
  SELECT
    user_id,
    event_timestamp,
    event_name,

    -- Extract specific keys as separate columns
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = "upgrade_referer") AS upgrade_referer,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = "sub_upgrade_referer") AS sub_upgrade_referer,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = "paywall_context") AS paywall_context,

    -- Keep the SKU from event_params
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = "feature_id") AS sku

  FROM event_data
  WHERE
    event_name = "subscription_process_succeed"
    AND event_timestamp/1000000 - COALESCE(
        SAFE_CAST(
          (SELECT value.string_value
           FROM UNNEST(user_properties)
           WHERE key = 'RMA_first_open_s') AS INT64
        ), 0
      ) <= 86400
),
event_info as (
SELECT
    user_info.*,
    combined.goals,
    combined.heard_about_us,
    combined.sleep_issues,
    combined.content,
    combined.question_satisfied,
    combined.question_current_hours_of_sleep,
    combined.topics,
    combined.gender,
    combined.age,
    combined.question_soundToDetect,
    event_counts_within_24h.sleep_recorded_count,
    event_counts_within_24h.start_routine_count,
    event_counts_within_24h.screen_routine_count,
    event_counts_within_24h.onboarding_skip_count,
    event_counts_within_24h.listening_count,
    trial_upgrade_ref.sku,
    trial_upgrade_ref.upgrade_referer,
    trial_upgrade_ref.sub_upgrade_referer,
    trial_upgrade_ref.paywall_context,
    question_reached.question_satisfied_reached,
    question_reached.question_age_reached,
    question_reached.question_gender_reached,
    question_reached.question_topics_reached,
    question_reached.question_goals_reached,
    question_reached.question_HAU_reached,
    question_reached.question_soundToDetect_reached,
    question_reached.question_sleepIssues_reached,
    question_reached.question_current_hours_of_sleep_reached,
    question_reached.question_content_reached
FROM user_info
LEFT JOIN
    combined
    on
    user_info.user_id=combined.user_id
LEFT JOIN
    event_counts_within_24h
    ON
    user_info.user_id = event_counts_within_24h.user_id
LEFT JOIN
    trial_upgrade_ref
    on
    user_info.user_id=trial_upgrade_ref.user_id
  LEFT JOIN
  question_reached
  on
  user_info.user_id=question_reached.user_id
),
backend_info as (
SELECT
    user_id,
    event_timestamp,
    1 AS paid,
    sum(ep.value.float_value) as revenue
FROM `relax-melodies-android.backend.events`,
      UNNEST(event_params) as ep
WHERE event_timestamp_s >= TIMESTAMP("2025-05-01")
  AND event_name IN ('subscription_start_paid')
  AND ep.key='converted_procceds'
GROUP BY user_id,event_timestamp
)
select
  event_info.*,
  backend_info.paid,
  backend_info.event_timestamp,
  backend_info.revenue,
"Android day0 May7-June7" as description,
CURRENT_DATETIME as extracted_datetime
from event_info
Left Join backend_info
on event_info.user_id=backend_info.user_id
