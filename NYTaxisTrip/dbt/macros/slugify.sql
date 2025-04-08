{% macro slugify(column_name) %}
    lower(regexp_replace({{ column_name }}, '[^a-zA-Z0-9]+', '_', 'g'))
{% endmacro %}
