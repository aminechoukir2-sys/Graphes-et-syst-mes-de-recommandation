# Complexité théorique

## Notations

- U : nombre d'utilisateurs
- I : nombre de films
- V = U + I : nombre total de nœuds
- E : nombre de notes, donc nombre d'arêtes utilisateur-film
- K : nombre de recommandations
- T : nombre d'itérations PageRank
- S : nombre de pas Random Walk
- d : degré moyen du graphe

## Construction du graphe

Chaque note devient une arête entre un utilisateur et un film.

```text
Temps : O(E)
Mémoire : O(V + E)
```

## BFS

BFS explore le graphe par niveaux.

```text
Pire cas : O(V + E)
```

Dans le code, BFS est limité par :

- profondeur maximale
- nombre maximal de nœuds visités
- nombre maximal de films candidats

Donc en pratique, le coût est contrôlé.

## DFS

DFS explore le graphe en profondeur.

```text
Pire cas : O(V + E)
```

Dans le code, DFS est aussi limité par profondeur et nombre de nœuds visités.

## PageRank seul

PageRank propage les scores pendant T itérations.

```text
Temps : O(T × E_local)
Mémoire : O(V_local)
```

Dans une version globale sur tout le graphe :

```text
Temps : O(T × E)
```

Ici, on garde un sous-graphe local pour que le projet reste exécutable.

## BFS/DFS + PageRank

```text
Temps : O(BFS/DFS) + O(T × E_candidates)
```

BFS/DFS réduisent l'espace de recherche avant PageRank.

## Random Walk seul

Random Walk simule S déplacements.

```text
Temps : O(S)
Mémoire : O(nombre de films visités)
```

## BFS/DFS + Random Walk

```text
Temps : O(BFS/DFS) + O(S)
```

BFS/DFS proposent les candidats, puis Random Walk score ces candidats.

## Évaluation

Pour N utilisateurs évalués :

```text
Precision@K / Recall@K / F1@K : O(N × K)
```

## Conclusion

Quand la taille du dataset augmente :

- E augmente avec le nombre de notes
- la construction du graphe devient plus coûteuse
- BFS/DFS peuvent devenir coûteux au pire, mais les limites évitent l'explosion
- PageRank dépend fortement du nombre d'arêtes dans le sous-graphe
- Random Walk dépend surtout du nombre de pas
- les versions BFS/DFS + scoring peuvent être plus rapides, mais peuvent perdre en rappel si les candidats sont trop limités
