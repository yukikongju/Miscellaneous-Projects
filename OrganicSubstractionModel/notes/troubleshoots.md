# Troubleshoot

- US organic is way to big => 80% of installs are Organic?
- [X] Sanity check: add known networks + organic and see if the number of installs resemble to what we see on appsflyer
- [X] Compare aggregate with attributed networks for spot check
    - [ ] remove double counting percentage from google and ASA
- [ ] Fix HAU/UTM attribution
    - [ ] ios => users who have 'utm_source=Apple' without
	  'campaign=Apple Search Ads'
    - [ ] android => users who have 'utm_source=google' without
	  'campaign=googleadwords_int'

**Ideas**

- Can we use `perc_diff` to see whether we are confident in
  this number?
- Can we use `perc_diff` to make confidence intervals

**Spot Check**
- DE estimated organics
- MX android underestimating by 20% from 7/02 to 7/06, looks fine otherwise
- UK missing
- US android overestimating because attributed installs is too
  much, which means that there is one channel on US android
  that is overestimating
- US ios is overestimating by 10% => we can remove avg perc_diff
  for estimated organic, which would bring down overestimation
  to 3%
