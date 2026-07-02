<<<<<<< HEAD
# Graphes-et-syst-mes-de-recommandation
=======
# Projet M1 Data — Graphes et systèmes de recommandation

Projet complet avec :

- Dataset synthétique configurable
- 5000 utilisateurs par défaut
- 2000 films par défaut
- 100000 notes par défaut
- Notes de 1 à 5
- Split train/test 80% / 20%
- Graphe bipartite utilisateurs × films
- BFS et DFS comme générateurs de candidats
- PageRank seul
- Random Walk seul
- BFS + PageRank
- DFS + PageRank
- BFS + Random Walk
- DFS + Random Walk
- Precision@K
- Recall@K
- F1@K
- Temps moyen d'exécution
- Complexité théorique
- Streamlit avec comparaison automatique
- POO + SOLID + classes abstraites

## Installation

```bash
pip install -r requirements.txt
```

## Lancer en console

```bash
python main.py
```

## Lancer avec une petite configuration de test

```bash
python main.py --users 500 --movies 200 --ratings 5000 --eval-users 20
```

## Lancer avec la configuration demandée

```bash
python main.py --users 5000 --movies 2000 --ratings 100000 --eval-users 50
```

## Lancer Streamlit

```bash
streamlit run app.py
```

## Méthodes disponibles

- pagerank_alone
- random_walk_alone
- pagerank_bfs
- pagerank_dfs
- random_walk_bfs
- random_walk_dfs

## Exemple de choix des méthodes en console

```bash
python main.py --methods pagerank_alone random_walk_alone pagerank_bfs random_walk_bfs
```

## Remarque importante

5000 utilisateurs × 2000 films = 10 000 000 couples possibles.

Le projet génère par défaut 100000 notes, car un vrai dataset de recommandation est sparse :
tous les utilisateurs ne notent pas tous les films.

Tu peux augmenter ou diminuer le nombre de notes avec :

```bash
python main.py --ratings 500000
```


## Mise à jour V2

Dans Streamlit :

- tu choisis exactement deux algorithmes dans la barre latérale ;
- le tableau affiche Precision@K, Recall@K, F1@K, temps d'exécution moyen et complexité ;
- un graphe compare uniquement ces deux algorithmes ;
- un deuxième graphe compare le temps d'exécution ;
- les films recommandés par les deux algorithmes sont affichés côte à côte.


## Mise à jour V3 — correction Precision/Recall et graphe linéaire

Cette version corrige le problème des Precision@K et Recall@K à zéro partout.

Le dataset n'est plus totalement aléatoire : les utilisateurs ont des préférences de genres et notent davantage des films populaires dans leurs genres préférés. Le test cache prioritairement des films bien notés. Les recommandations ont donc une vraie chance de retrouver des films pertinents.

Ajouts :
- Precision@K et Recall@K calculés pour les deux algorithmes sélectionnés
- F1@K
- temps d'exécution moyen
- complexité
- graphique en barres
- graphe linéaire Precision / Recall / F1
- films recommandés côte à côte


## Version finale V4

Cette version ajoute :

- comparaison graphique entre trois algorithmes sélectionnés ;
- calcul automatique de Precision@K, Recall@K, F1@K, temps moyen et complexité pour les six algorithmes :
  - pagerank_alone
  - random_walk_alone
  - pagerank_bfs
  - pagerank_dfs
  - random_walk_bfs
  - random_walk_dfs
- affichage des films recommandés par les trois algorithmes sélectionnés ;
- tableau complet des six algorithmes dans Streamlit.


## Version stable V5

Corrections :
- génération exacte du nombre de notes demandé ;
- dataset plus cohérent avec préférences utilisateurs ;
- moins de risques de Precision@K et Recall@K à zéro ;
- conservation de la comparaison des 6 algorithmes ;
- conservation de la comparaison graphique entre 3 algorithmes ;
- conservation du graphe linéaire.


## Version V6 — Correction BFS/DFS + Random Walk

Cette version garde le même projet mais corrige le problème suivant :
- pagerank_bfs ou random_walk_bfs pouvait obtenir Precision@K = 0 et Recall@K = 0 parce que BFS s'arrêtait parfois avant d'avoir récupéré assez de films candidats pertinents.

Corrections :
- BFS/DFS récupèrent maintenant aussi les films notés par les utilisateurs atteints dans le parcours ;
- les candidats sont scorés selon la distance et le poids de l'arête ;
- les nœuds des candidats sont ajoutés au sous-graphe local ;
- Random Walk possède un score de secours si aucun candidat n'est visité pendant la marche ;
- PageRank possède aussi un score de secours très faible pour éviter les scores totalement nuls.


## Version légère V7

Pour accélérer l'exécution, les valeurs par défaut ont été réduites :

- 500 utilisateurs
- 200 films
- 10 000 notes
- 50 utilisateurs évalués
- 1 500 pas Random Walk

Tu peux toujours augmenter les valeurs dans Streamlit ou avec la ligne de commande.

Exemple rapide :

```bash
python main.py
```

Exemple plus grand :

```bash
python main.py --users 5000 --movies 2000 --ratings 100000 --eval-users 50
```
>>>>>>> 886215f (Ajout du projet graphes et systemes de recommandation)
