# DBT Models

Structure:

```{}
models
├── README.md
├── intermediate/
├── marts/
└── staging/
```

- Staging `stg_*` => cleans raw data and columns renaming
- Intermediate `int_*` => combines staging data, applies logic
- Marts `fct_*, dim_*` => final tables for analytics
