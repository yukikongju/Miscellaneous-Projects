# Henry Ford Stack

**Field Dimension**

- 70 yards by 40 yards; end zones 25 yards deep

**What's the stack**

- give the disc in teammate hand

**What's the stats**

- What should be completion rate of each hand-off to have a point conversion 
  rate of _ %?
    * To answer this question, we need a geometric function: we assume each 
      hand-off has a similar probability

Parameters to consider:
- How many yards are we going at each hand-off => how many passes do we need to 
  score a goal?
- completion rate at each hand-off?

Completion rate formula:
- HF turn over rate: proba turn at 1st hand-off + proba turn at 2nd hand-off + ...
    * proba that first failure occurs after the number of passes needed to 
      score a goal
- HF success rate: 1 - HF turn over rate

```{r}

# parameters of model
yards_per_handoff <- 5
num_pass_required <- ceiling(70/yards_per_handoff)
handoff_completion_rate <- 0.95

# compute Henry Ford conversion rate
hf_fail_rate <- 0
for ( n in 1:num_pass_required ) {
    p <- dgeom(n, prob=handoff_completion_rate)
    hf_fail_rate <- hf_fail_rate + p
}

hf_completion_rate <- 1 - hf_fail_rate
```

```{r}
# parameters of model
yards_per_handoff <- 5
num_pass_required <- ceiling(70/yards_per_handoff)
handoff_completion_rate <- 0.95

# G = p ^ n => hf_completion_rate = handoff_completion_rate ^ num_pass_required
# p = G ^ (1/n)

#
nrooth = function(x, n) {
(abs(x)^(1/n))*sign(x)
}
G <- 0.90
p <- nrooth(G, num_pass_required)

#
p <- 0.95
G <- p ^ num_pass_required
```

# Results

- If the hand-off probability is 95% and player make up 5 yards every hand-off, 
  the HF completion rate would be of 48.76% 
- If the hand-off probability is 92% and player make up 5 yards every hand-off, 
  the HF completion rate would be of 31.12%
- If we run the HF stack 15 times in a game, we should expect to convert 7.5
  of these points (np)
- The teams with the best offensive conversion rate range around 60% (Shred, 
  Breeze, Hustle, Empire), where are the worse offensive team range around 38-40% 
  (Royal, Nitro, Havoc, Mechanix), which means that HF would only be effective 
  for bottom feeding team.
    * [Team Stats](https://watchufa.com/stats/team?dir=desc&sort=dLineConversionPercentage&year=2023)

