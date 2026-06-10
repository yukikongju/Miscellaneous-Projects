# Listening Cumulative Table

**Goals**

We want to build tables to get:
- sounds, music, mixes => can be found in `listening` event
- guided content => can be found in `screen_content_playing` (w/o sounds)
- both => can be found in `listening_session` ('guided_content_id', 'sounds_ids',
  'music_id', 'brainwaves_id','screen', 'mix_type') => TODO: check if 'sounds' with
  guided_ids

I think `listening` event in BigQuery is broken because it doesn't match
what we have on mixpanel. `guided_content` is always null.


**Cumulative Table for `listening`**

1. [X] Create Listening Table subset with the last 90 days of listening events
       partitioned on user_id: `listening_events_partitioned` => `queries/backfill_listening_events.sql`
2. [ ] Create Tables: `users_listening_hourly`, `users_listening_daily`, `users_listening_cumulated`
3. [ ] Create Populate


**Cumulative Table for `screen_content_playing`**

- `content_type, content_id`
