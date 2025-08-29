# Experiment Simulations

This project aims to understand the following situations
when performing an A/B tests:
- `Power Estimation`: Estimate power as a function of its sample
  size (N) and its baseline.
     * Power = P(reject H0 | HA true) = True Positive = 1 - beta
     * alpha = False Positive
     * beta = False Negative
- `Peeking`: False Positive under peeking ie early stopping. Measure
  Type-I error inflation
- `Guardrail collisions`: probability that one metrics falsely moves
  when calling an exp on several metrics, specifically when
  calling on paid rate when exp was scoped on feature metrics
