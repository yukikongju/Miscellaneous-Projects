import numpy as np
import pandas as pd

def bernouilli(p: float):
    return 1 if p > np.random.uniform() else 0


def main(num_samples: int, simulation_length: int, p: float):
    np.random.seed(0)

    # --- 1. run coin simulation
    results = []
    for _ in range(num_samples):
        res = [bernouilli(p) for _ in range(simulation_length)]
        results.append(res)
    
    # --- 2. compute mean and standard deviation for all samples
    df = []
    for res in results:
        df.append([np.mean(res), np.std(res)])
    df = pd.DataFrame(df, columns=['mean', 'sd'])
    print(df)

    # --- 3. compute mean and sd of all simulations
    mc_mean, mc_sd = df['mean'].mean(), df['sd'].std()
    print(f"Simulations Mean: {mc_mean}")
    print(f"Simulations Standard Deviation: {mc_sd}")
    


if __name__ == "__main__":
    NUM_SAMPLES = 100
    NUM_SAMPLES_PER_SIMULATIONS = 10000
    p = 0.5
    main(num_samples=NUM_SAMPLES, simulation_length=NUM_SAMPLES_PER_SIMULATIONS, p=p)

