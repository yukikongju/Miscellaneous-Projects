Jul 24, 2025
- backfilled `users_network_attribution` from 2023-01-01 to 2025-07-22
    * need to remove from 07-22 because unmature yet

Jul 29, 2025
- Created `latest_renewal_rates`
- Created `renewal_rates_ma_imputed` and `renewal_proceeds_ma_imputed`, which
  compute the rates and proceeds from the latest renewal and ensure that
  no cells are empty. Logic:
    * For Renewal Rates:
    * For Proceeds: depreciation rate 0.8
