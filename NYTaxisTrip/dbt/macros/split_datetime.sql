{% macro split_datetime(column_name) %}
    {{ column_name }}::date as {{ column_name }}_date,
    {{ column_name }}::time as {{ column_name }}_time
{% endmacro %}
