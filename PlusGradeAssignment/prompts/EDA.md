Variables:

Outcomes:
- OFFER_RICHNESS_APPLIED, OFFER_RICHNESS_SERVED
- POINTS_PURCHASED
- PRICE_PER_POINT


Goal of EDA:
- Define the grain
- Data Quality
- Missing Data
- Univariate target and
- Splitting data without data leakage

EDA Results:
- Univariate Distribution
    * `FLAG_FIRST_TIME_VISITOR` => 80% first time visitors; 20% returning
    * `FLAG_FIRST_TIME_BUYER` => 54% first time buyer; 46%
    * `OFFER_RICHNESS_SERVED` => 50% discount appied 77% of the time; 45% applied 18%; 40% applied 4.75%
    * `FLAG_TRANSACTION` => Users buy points 13.71% of the sessions (ie no points boughts 86% of the sessions)
    * `OFFER_RICHNESS_APPLIED` => Offer not applied 86% of the transactions
    * `POINTS_PURCHASED` => ranging from 3K to 59K
    * `DAYS_SINCE_LAST_PURCHASE_L12M` => no purchase "9999" 72% of the sessions
    * `COUNT_TRANX_L12M` => 0-21




Decisions Points:
- What to do with users who have another session
    * 80% first time; 20% returning
    * 53% first time buyers;
- Should time be a feature?
    * Since the data we have only cover 2 days, no time-series considered

To Explore:
- Are users behaviors for users who first buy vs returning users different?


**Questions to ask Lori**

Hi Lori, I was just reading the problem statement and had a few questions I was hoping you could clarify. From our earlier conversation, I understand that Points/Plusgrade business has two components:
* A *loyalty program*, where users buy points to use them to redeem for flights, upgrade seats, hotels, etc.
* An *auction sale*, where users bid on last minute offers.

With that in mind:

- Just to be sure, is this exercice focused solely on the loyalty program? Is this why flight prices, premium upgrade prices and time of flight are not included?
- Regarding the statement "All offers require a minimum of 3,000 points purchased to qualify.", does this mean that a user need to buy a minimum of 3K points in a given transaction to have the discount applied, or simply have at least 3K points in their current balance in order to qualify for a discount?
- At first glance, the purchase conversion rate seems to be higher on September 2nd than it is on September 3rd. Should we interpret this as part of the analysis or are we supposed to make inference on a out-of-distribution dataset?
- For the offer allocation strategy (part c), is it acceptable to adjust offer assignments after an initial allocation? For example, if I apply a greedy strategy were each user are assigned the discount rate which maximize the expected revenue, but later find out that this strategy violate the business constraints, would changing some offer assignment later be valid? I know this can't happen in the real world, but I was just wondering if we had to keep that in mind for the sake of the exercice.

Best,
Emulie

----

Hi Lori, I was just reading the problem statement and had a few questions I was hoping you could clarify. From our earlier conversation, I understand that Points/Plusgrade business has two components:
A loyalty program, where users buy points to use them to redeem for flights, upgrade seats, hotels, etc.
An auction sale, where users bid on last minute offers.

With that in mind:

Just to be sure, is this exercice focused solely on the loyalty program? Is this why flight prices, premium upgrade prices and time of flight are not included?
Regarding the statement "All offers require a minimum of 3,000 points purchased to qualify.", does this mean that a user need to buy a minimum of 3K points in a given transaction to have the discount applied, or do they simply need to have at least 3K points in their current balance in order to qualify for a discount?
At first glance, the purchase conversion rate seems to be higher on September 2nd than it is on September 3rd. Should we interpret this as part of the analysis or are we supposed to make inference on a out-of-distribution dataset?
For the offer allocation strategy (part c), is it acceptable to adjust offer assignments after an initial allocation? For example, if I apply a greedy strategy were each user are assigned the discount rate which maximize the expected revenue, but later find out that this strategy violate the business constraints, would changing some offer assignment after the original assignment be valid? I know this can't happen in the real world, but I was just wondering if we had to keep that constraint in mind for the sake of the exercice.




- Why would users buy 50K points? Is it normal that `POINTS_PURCHASED`
- Variables definition:
    * `CURRENT_BALANCE`
