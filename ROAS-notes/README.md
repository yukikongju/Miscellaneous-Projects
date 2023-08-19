# BetterSleep - ROAS/MMM Notes

## Leading questions

**The Business Vision**

- The company growth strategy is based on ROAS
    1. What is ROAS? How is it calculated?
	    - `ROAS = (revenue from ad campaigns) / (cost of ads campaigns)`
    2. What are the limitations of ROAS as a metric?
	    - doesn't account for organic traffic
    3. What are the challenges when estimating ROAS?
	    - Attribution: Because of SKAN and GDPR, we don't know the exact 
	      installs/trials/paid that can be attributed to each of the campaigns.
	      We need to estimate those values based on the spend.
		    * If the installs/trials/paid values aren't good, we are making 
		      decisions based on erroneous values
		    * Sometimes, we have access to the installs/trials/paid, but the 
		      values we get differ depending on each sources (individual networks 
		      vs Appsflyer vs In-House models). We are building a 
		      discrepancy board to detect whether the values we are using are 
		      correct (if we need to use direct network, appsflyer, modeled)
    4. How can we improve ROAS and revenues?
	    - Improving the volume of users installing the app through:
		    * (1) Organic Installs: getting featured on App Store, mouth to mouth, ...
			    + Getting good rating, being aligned with what App Store promotes
		    * (2) Ads Campaigns
			    + Improving ads quality -> Advertising team
			    + More money in ads that cost less but better installs/trials/paid
			    + Kevin/Lena vs RL
		    * We have to **identify the ads that have the best CPI/CPA**
	    - Better conversion from Install2Trial and Trial2Paid by improving app quality through A/B tests
		    * Our contents:
			    + Guided Content and Sounds
			    + Sleep Tracker
		    * Onboarding Process
		    * Paywall Reach Rate
		    * We can improve app quality by having **better incremental changes quicker and which minimize risk**
			    + Better incremental changes: do the feature we want to test out make sense?
			    + Faster feature implementation by having more devs/improving workflows
			      However, **the timeline of our tests is capped by the amount of users we are getting in 
			      our apps bc we need to be statistically significative**
			    + How can we manage timeline and risk we are willing to accept
			    + **How can we determine if making two quicker exps is better than make a longer exp?**
			    + Find proxy metrics that correlates with paid metrics
	

- What is MMM? How does it help us improve ROAS?
    * What is the current MMM model we have? How is it performing? How can we improve it?
    * How are we using it to guide our business decisions?
    * 

- Which leading metrics are related to ROAS?

**Data Science Projects**


1. Best Attribution Models for installs/trials/paid/spend/revenues for country/campaign level
	- We want to get these metrics from 3 sources: (1) individual network (2) appsflyer (3) in-house model
	- Given these 3 sources, for each network, which source should we use for metric X
		* ex: for Facebook Trials, use appsflyer
		* We will use our UA - Discrepancy board for that
	- How does our in-house model works? How can we improve it?
	    * Current model: estimate trials via Trial2Paid rate and compute smoothing average (dynamic paid model)
	    * 
2. Best MMM Models to optimize budget allocation, attribution and forecast future performance


- [Google Lightweight MMM](https://github.com/google/lightweight_mmm)
	* [paper](https://arxiv.org/pdf/1908.09257.pdf)
- [Stan MMM](https://github.com/sibylhe/mmm_stan/tree/main)



