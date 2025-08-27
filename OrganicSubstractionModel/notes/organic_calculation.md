--- cost:

--- Query Plan - iOS:
--- rough formula: Appsflyer Aggregate - double_counting_multiplier * ASA count
--- 1. appsflyer_aggregate => "ios Appsflyer Aggregate" from pre_final_view
--- 2. asa_double_counting =>
--- 3. ASA counts => "Apple Search Ads" from pre_final_view
--- 4. ios_organic_estimation =>



--- Query Plan - ANDROID:
--- rough formula: Appsflyer Aggregate - double_counting_multiplier * google count
--- 1. appsflyer_aggregate => "Appsflyer Aggregate" from pre_final_view
--- 2. google_double_counting =>
--- 3. android_organic_estimation =>


--- FINAL Query Plan - iOS & Android
--- 5. UNION ALL ios_organic_estimation and android_organic_estimation
