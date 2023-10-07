# Monte Carlo Simulations

---

**Coin Flips [ Completed ]**

Run n simulations of m coin flips. Calculate mean and standard deviation. 
This is the monte carlo simulation for the binomial distribution.

---

**Experiments [ In Progress ]**

- What is the true/false positive rate of our early peaking calls for a given 
  experiment
- If we decide to make the early peaking call, what should be the pvalue accepted?

Steps:
1. Generate experiment data by generating daily traffic with fixed mean and variance from a absolute delta 
    - In reality, we will do the inverse: we use the data we get from MixPanel to calculate the mean and the variance
    - Given that we are seeing x percent lift, what is the probability that the true lift is greater that $\delta$ ie the lift we need to see to call this experiment a win 
2. Everyday, we compute results as if we would call the experiment that day (early peaking). 
3. We will simulate the experiment several time and compute the false/true positive rate. This rate will correspond to type I and type II error and will determine the risk of calling the experiment based on that early peaking call.
    - Type I: reject H0 given H0 => we are deploying the feature in the app even though the lift is not statistically significant
	    * Bad because we may release a feature that end up making us loose revenue
    - Type II: accept H0 given HA => we are not deploying the feature in the app even though the positive lift we see if statistically significant
	    * Bad because we are not growing as fast as we could
    - We would rather have type II errors than type I

---

**Quarterly Holdback**

- Did our new releases really improve our main metric? Is this lift statistically 
  significant?
- The experiences the users get at the beginning of the quarter is not the 
  same as the one late in the quarter because users at the end of the quarter 
  are seeing more new features.
    * Should we throw out data at the beginning and make the hypothesis that 
      after a certain period, the experiences for all users is the same
    * Can we weight the experiences in a given period by the number of new 
      release a user is seeing?
- The baseline we use to compare to the users that get the full experience is 
  the conversion rate corresponding to the respective dates

Steps:
1. 


---

**Friendship Paradox**

- In a social network, why do your friend are more likely to have more friends than you 
- What if the number of friends follows: 
  (1) normal distribution
  (2) exponential
  (3) logarithmic

---

**PValue Optimization**

- When calling our A/B tests, we are using an arbitrary pvalue of 0.05 which 
  is too conservative for our needs. What should be the pvalue be given our 
  risk tolerance?

Steps:
1. 

---

