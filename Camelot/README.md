# Problème du Camelot

Un camelot achète des journaux la veille et les revend le lendemain. Notre but est de maximiser le profit du camelot

- x: Nombre de journaux achetés
- d: demande des clients
- c: cout d'achat d'un journal
- p: prix de revente d'un journal (p>c)

Espérance du profit: 

```math
E\[A(x)\] = -cx + p(\sum_{d=d}{\bar{d}} min(x,d) P(D=d))
```

## Usage

##### Initialiser l'objet camelot

```python
    d = np.transpose([[3,4,5,6,7], [0.1, 0.2, 0.3, 0.3, 0.1]])
    distr = pd.DataFrame(data=d, columns=['x', 'Percentage'])
    camelot = Camelot(2, 2.5, distr)
```

##### Trouver la solution optimale

```python 
    num_journals, max_profit = camelot.get_optimal_solution()
```


