-- select
--   table_name
--   , column_name
--   , ordinal_position
--   , is_nullable
--   , data_type
-- from `mappings.INFORMATION_SCHEMA.COLUMNS` -- <DATASET_NAME>

select
  TO_JSON_STRING(
    ARRAY_AGG(STRUCT(
      IF(is_nullable = 'YES', 'NULLABLE', 'REQUIRED') as mode,
      column_name as name,
      data_type as type
    ) order by ordinal_position
    ), TRUE
  ) as schema
from `mappings.INFORMATION_SCHEMA.COLUMNS`
where
  table_name = 'country_maps' --- <TABLE_NAME>
