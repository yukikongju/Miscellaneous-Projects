# Notes

**Data Cleaning**
- Missing month for network-platform-country
- rolling average to smooth out
- take the latest available
- forecast unmature data
- tatari_linear should be mapped to tatari_streaming
- default rate for missing network-platform-country
- compare variance on weekly moving average vs monthly moving average
- compare query renewal rate for year 1 vs mixpanel


**TODOs**

[spreadsheet](https://docs.google.com/spreadsheets/d/1VNfA5q5SdOfmZ4EPrr8H9BdEB1UZA6sa2dQinBUp_oE/edit?gid=0#gid=0)

- [ ] Add multiplier in MixPanel for experiments => Late Trial Conversions, Revenues from renewal within 5 years,


**Done**

- [X] User-level HAU/UTM/Trials/paid/refund queries (used for this and T2P)
- [X] [Henry] Renewal conversions distribution => look at distribution
      of paid occuring after 60 days. These conversions won't be taken into
      account in the current T2P, so we can use the distribution to find a
      multiplier, which we can use to multiply the trials. => note: can't use
      mixpanel because the conversion window max is 367 days.
    - [X] Query: renewal distribution
    - [X] Query: how many users renew year over year => on excel
    - [X] Query: average revenue per paid/renew =>
- [X] [Henry] Late Trial Conversions => use mixpanel => % of late trials (after 60 days) is 3.34% for android; 1.806% for ios
- [X] Query: Look at renewal breakdown by monthly cohort => focus on annual skew .year (no .monthly)

**Sub-Tasks Ideas**

- [ ] [Using Late Trials and Renewal Conversion Distribution to compute revenues]
    * yearly trials = trials * 1.166% = trials within first month + rest of year
    * yearly paid = yearly trials x t2p
    * realized revenue = yearly paid x rev_per_paid
    * unrealized revenue = 1 year renewal + year 2 renewal + ... + year 5 renewal =>
- [ ] [Late Conversions based on network/platform]
    * Compute late trials and renewal estimation based on historical values.
    * Create Dashboard
- [Predicting Users Renewal Rate]
    * based off in-app events within period (first day)
    * based off time from install to trial => do user who take the trial faster pay/renew more?



**Questions**

- How to use late revenue model to inform business decisions?


**Boards**

- [Time to convert - UTM to HAU](https://mixpanel.com/s/1EV85S) => 28D max
