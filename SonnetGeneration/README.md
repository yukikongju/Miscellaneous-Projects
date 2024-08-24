# Sonnet Generation

Generating Shakespearean Sonnet using the following methods:
- [ ] Method 1: co-occurence matrix distribution
    * Idea: similar to bi-grams => given context of `block_size`, sample from 
      probability distribution
- Method 2: character level prediction
    * Idea: Given previous context of length `block_size`, predict the next 
      character
    * Sub-methods:
	1. [ ] MLP
	    a. Source: [Andrej Kaparthy - makemore pt 1]
	2. [ ] Convergence Embedding MLP
	    a. Source: [Andrej Kaparthy - makemore pt 2]
	3. [ ] Flattening neurons
	    a. Source: [Andrej Kaparthy - makemore pt 5 - wavenet]
	4. [ ] RNN, LSTM, GRU
	    a. Source:
	5. [ ] With Temperature
- Method 3: 


Next Steps:
- do it with guided meditation and bedtime stories
- using a context: GPT, seq-2-seq model

## Resources

**Projects**

- [Shakesperizing Modern English](https://github.com/harsh19/Shakespearizing-Modern-English/tree/master)
- [Generate Shakespeare Sonnets](https://github.com/enerrio/Generate-Shakespeare-Sonnets)

