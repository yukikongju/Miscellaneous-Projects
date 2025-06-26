SELECT
  distinct event_name
FROM `relax-melodies-android.backend.events` WHERE TIMESTAMP_TRUNC(event_timestamp_s, DAY) = TIMESTAMP("2025-06-26") LIMIT 1000;

-- Jun 26, 2025
--  subscription_cancelled
--  subscription_entitlement_paid
--  subscription_expired
--  subscription_restarted
--  subscription_start_paid
--  subscription_renew_paid
--  subscription_in_grace_period
--  intercom
--  subscription_start_trial
--  subscription_refund
--  subscription_3h_in_trial
--  subscription_24h_in_trial
--  subscription_ua_signal
--  subscription_onhold
--  subscription_recovered
