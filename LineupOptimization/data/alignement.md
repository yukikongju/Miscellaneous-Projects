---
title: "Lineup Valks"
author: [Mumu] 
date: "Jun 26, 2023"
keywords: []
numbersections: true
colorlinks: true
link-color: blue
geometry: "left=3cm,right=3cm,top=2cm,bottom=2cm"
toc-depth: 4
---

# Lineup with Fielder Spectral Clustering Method

**But**

On veut créer 2 lignes basées uniquement sur les préférences des filles

**Comment ça fonctionne**

1. On construit un **graphe dirigé** (représenté par un matrice d'adjacence) où 
   chaque joueuse est un noeud. On ajoute un arc partant d'une joueuse A à 
   une joueuse B si A veut jouer avec B
2. On construit la **matrice de degré** associé au graphe: c'est une matrice 
   diagonale où M[i][i] est le degré du noeud i
3. On construit la **matrice Laplacienne** du graphe
4. On calcule les **valeurs propres et les vecteurs propres** de la matrice Laplacienne
5. On **trie** chaque joueurs selon leurs valeurs propres
6. On créer les groupes avec un algorithme de **clustering**: KMeans ou Fielder


**Résultats**

|    | Britney     | Spears    |
| -- | ----------- | --------- |
| 1  | Juliette    | Clemence  |
| 2  | Annabelle   | Jenna     |
| 3  | Simone      | Lau       |
| 4  | Oriana      | Virg      |
| 5  | Pez         | Emma      |
| 6  | Potvin      | Emilie    |
| 7  | Allard      | Anne      |
| 8  | Audrey      | Clelia    |
| 9  | Bea         | Marion    |
| 10 | Dahlia      |           |

**Limitations**

- Les lignes sont uniquement basées sur les préférences des filles, pas si 
  elles sont handlers ou cutters
- Il peut y avoir plus de joueurs sur une ligne que sur l'autre 
- L'algorithme favorise les joueuses les plus populaires et pénalise beaucoup 
  les joueuses les moins populaires
- Les joueuses qui donne plus que 3 noms sont désavantagées
- L'algo fonctionne plus si les joueuses donne leur top5+ (on n'a pas de 
  vecteur nuls)

\break

# Lineup with Integer Linear Programming - Offensive & Defensive Lines

**But**

On veut créer 2 lignes: une offensive et une défensive. Pour ce faire, on 
définit chaque joueuse avec les informations suivantes:
- une liste de joueuse avec qui elle aimerait jouer
- score offensif et défensif

On met plus d'importance sur la force des lignes vs les préférences des filles 
en pondérant davantage le score offensif/défensif

**Comment ça fonctionne**

1. On définit des contraintes à respecter:
    - même nombre de filles par ligne
    - une fille ne peut être que sur une ligne
    - Degree Centrality
2. On définit une fonction linéaire à optimiser: on veut maximiser le score total 
   obtenu sur la ligne offense et défensive. Si une fille est sur la O, on utilise 
   son score de O; si elle est sur la D, son score de D
3. On applique l'algo CPLEX pour trouver la solution


**Résultats**

|    | Britney  | Spears    |
| -- | -------  | --------- |
| 1  | Juliette | Anne      |
| 2  | Clemence | Annabelle |
| 3  | Jenna    | Simone    |
| 4  | Pez      | Ori       |
| 5  | Lau      | Virg      |
| 6  | Allard   | Potvin    |
| 7  | Emilie   | Emma      |
| 8  | Clelia   | Audrey    |
| 9  | Dahlia   | Bea       |
| 10 | Marion   |           |


**Limitations**

- Les préférences de chaque fille ont la même importance

\break

# Lineup with Integer Linear Programming - Cutter & Handler Lines

**But**

On veut créer 2 lignes équilibrées. Pour ce faire, on 
définit chaque joueuse avec les informations suivantes:
- une liste de joueuse avec qui elle aimerait jouer
- score handling et cutting

On met plus d'importance sur la force des lignes vs les préférences des filles
en pondérant davantage le score handling/cutting

**Comment ça fonctionne**

1. On définit des contraintes à respecter:
    - même nombre de filles par ligne
    - une fille ne peut être que sur une ligne
    - Degree Centrality
    - 3 handler minimum par ligne
2. On définit une fonction linéaire à optimiser: on veut maximiser le score total 
   obtenu sur les 2 lignes. Si une fille est handler, on utilise son score de
   handling; si elle est cutter, son score de cutting
3. On applique l'algo CPLEX pour trouver la solution


**Résultats**

|    | Britney  | Spears    |
| -- | -------  | ------    |
| 1  | Juliette | Anne      |
| 2  | Simone   | Annabelle |
| 3  | Oriana   | Clemence  |
| 4  | Pez      | Jenna     |
| 5  | Lau      | Allard    |
| 6  | Virg     | Audrey    |
| 7  | Potvin   | Bea       |
| 8  | Emma     | Emilie    |
| 9  | Marion   | Clelia    |
| 10 |          | Dahlia    |


**Limitations**

- Les préférences de chaque fille ont la même importance

\break

# Code

[Lien au Code](https://github.com/yukikongju/Miscellaneous-Projects/tree/master/LineupOptimization)

