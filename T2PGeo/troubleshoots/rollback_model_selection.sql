--- insert statement
INSERT INTO `relax-melodies-android.ua_transform_dev.model_lookup` (
    platform,
    network,
    data_source,
    model_type,
    model_table,
    start_date,
    end_date,
    updated_at,
    deprecated
)
SELECT
    platform,
    network,
    'internal' AS data_source,
    'rolling t2p' AS model_type,
    'relax-melodies-android.ua_transform_dev.trial2paid_model_unique' AS model_table,
    start_date,
    end_date,
    CURRENT_DATETIME() AS updated_at,
    deprecated
FROM `relax-melodies-android.ua_transform_prod.model_lookup`
WHERE model_type = 'rolling less fallback t2p'
ORDER BY platform, network, start_date, end_date;


--- delete statement
DELETE FROM `relax-melodies-android.ua_transform_dev.model_lookup`
WHERE
    model_type = 'rolling less fallback t2p'
