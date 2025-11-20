DECLARE
  union_sql STRING;

SET
  union_sql = (
  WITH
    key_array AS (
    SELECT
      ARRAY_AGG(DISTINCT CONCAT("'", network, "'")) AS routed_keys
    FROM
      relax-melodies-android.`ua_transform_prod.model_lookup`),
    union_clauses AS (
    SELECT
      STRING_AGG( FORMAT("""
      SELECT
         install_date
      , extracted_datetime
      , platform
      , country
      , sub_continent
      , continent
      , network
      , trial
      , paid
      , revenue
      , aggregation_group
      , nb_days_rolled
      , valid_name
      , rolling_total_trial
      , rolling_total_paid
      , modeled_trial2paid
      , modeled_revenue_per_trial
      , modeled_revenue_per_paid
      , modeled_refund
      , modeled_refunded_amount
      , refund
      , refunded_amount
      , modeled_paid2refund
      , modeled_refunded_amount_per_paid
       FROM `%s` where platform = "%s" and network = "%s" and install_date >= "%s" and install_date <= "%s"
     """, model_table, platform, network, CAST(start_date AS string), CAST(end_date AS string)), " UNION ALL " ) AS all_route_queries
    FROM
      relax-melodies-android.`ua_transform_prod.model_lookup` )
  SELECT
    FORMAT("""
  SELECT * FROM `relax-melodies-android.ua_transform_prod.trial2paid_model`
  WHERE network NOT IN UNNEST([%s])
  UNION ALL
  %s
""", ARRAY_TO_STRING(routed_keys, ", "), all_route_queries)
  FROM
    key_array,
    union_clauses);
SELECT
  union_sql;

EXECUTE IMMEDIATE FORMAT("""
  CREATE OR REPLACE VIEW `relax-melodies-android.ua_transform_prod.trial2paid_ensemble` AS
  %s
""", union_sql);
