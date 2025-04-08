{% macro get_max_loaded_date(model_name, column_name) %}
    {{
        return(
            "select max(" ~ column_name ~ ") as max_date from " ~ model_name
        )
    }}
{% endmacro %}
