# Récapitulatif de travail — Classificateur MasakhaNEWS

Ce document résume les choix techniques, le travail réalisé, les résultats obtenus et les raisons derrière ces choix. Il sert de support de révision, de préparation à la présentation orale et de mémorisation du pipeline complet du projet.

## 1. Objectif du projet

Le but du projet est de construire un classificateur d’articles en français à partir du dataset MasakhaNEWS. Le défi principal est de transformer du texte brut en représentation numérique exploitable par un réseau neuronal PyTorch.

Le pipeline complet suit cette logique :

1. nettoyage des données ;
2. tokenisation du texte ;
3. construction d’un vocabulaire ;
4. encodage des mots en indices ;
5. padding/troncature pour obtenir des séquences de même longueur ;
6. création d’un dataset PyTorch ;
7. entraînement d’un modèle de classification ;
8. évaluation avec métriques et matrice de confusion ;
9. comparaison avec une baseline TF-IDF.

Le dataset retenu contient cinq classes : `business`, `health`, `politics`, `sports` et `technology`. Ces classes sont celles du corpus MasakhaNEWS et correspondent à la version la plus stable du projet. Elles sont utilisées telles quelles, sans renommage inutile.

## 2. Les données : nettoyage et prévention des fuites

Avant de construire le modèle, on a vérifié les données et appliqué des règles strictes de nettoyage. Les données viennent de trois fichiers CSV officiels : train, validation et test.

Les étapes de nettoyage ont été les suivantes :

- suppression des doublons exacts dans le train ;
- suppression des textes partagés entre plusieurs splits avec labels différents ;
- suppression des textes partagés entre val/test avec le même label ;
- suppression des conflits internes dans le train où le même texte a plusieurs labels différents.

Ces règles sont importantes pour éviter les fuites d’information entre les splits, ce qui fausserait les résultats et rendrait le modèle trop optimiste en validation ou en test.

Les tailles finales du dataset après nettoyage sont :

- train : 1472 exemples ;
- validation : 210 exemples ;
- test : 420 exemples.

Les labels textuels sont ensuite convertis en indices numériques :

- `business` -> 0
- `health` -> 1
- `politics` -> 2
- `sports` -> 3
- `technology` -> 4

Cette conversion est nécessaire pour que PyTorch puisse utiliser les labels dans `CrossEntropyLoss`.

## 3. Préparer le texte : tokenisation

La fonction `tokenize` de `src/vocab.py` convertit une phrase en une liste de tokens. La logique est la suivante :

- tout est mis en minuscules ;
- les accents sont conservés ;
- les apostrophes courbes sont normalisées en apostrophes droites ;
- les mots conservent leur apostrophe ou leur trait d’union interne ;
- la ponctuation est traitée comme un token séparé.

Exemple :

```text
L’Ouest-africain, c'est prêt.
→ ["l'ouest-africain", ",", "c'est", "prêt", "."]
```

L’ordre des tokens est important car il reflète la structure de la phrase et permet de conserver le sens du texte avant conversion en indices.

## 4. Construire le vocabulaire

Le vocabulaire associe chaque token à un identifiant numérique. Il est construit uniquement à partir des textes d’entraînement. Cela évite toute fuite d’information depuis validation ou test.

Deux identifiants sont réservés :

- `<PAD>` : index 0, pour compléter les séquences courtes ;
- `<UNK>` : index 1, pour remplacer les mots absents du vocabulaire.

Les autres mots commencent à l’index 2. Cette convention est essentielle pour garantir la compatibilité avec `nn.Embedding`.

Le paramètre `min_freq` permet de filtrer les mots trop rares. Par exemple, avec `min_freq=2`, un mot vu seulement une fois est remplacé par `<UNK>` à l’encodage. Cela limite le vocabulaire et réduit le bruit.

Exemple :

```text
Texte d'entraînement : "chat chat chien"
Vocabulaire avec min_freq=2 : <PAD>=0, <UNK>=1, chat=2
Encodage de "chat chien" : [2, 1]
```

Le tri par fréquence décroissante puis par ordre alphabétique rend la construction du vocabulaire déterministe.

## 5. Encoder, compléter et tronquer les séquences

La méthode `encode()` transforme un texte en une liste d’indices. La méthode `pad_or_truncate()` applique ensuite la normalisation de longueur :

- si la séquence est trop longue, on la tronque ;
- si elle est trop courte, on la complète avec des `<PAD>`.

Cela permet de former des batchs de même taille. C’est indispensable car un modèle de Deep Learning ne peut pas traiter des séquences de longueurs différentes dans le même lot.

## 6. Choix de `max_length`

Le choix de `max_length` est très important. Nous avons calculé les statistiques de longueur sur le train nettoyé en utilisant le vrai tokenizer du projet.

Résultats observés :

| Mesure | Longueur en tokens |
|---|---:|
| Médiane (p50) | 808 |
| p90 | 1801 |
| p95 | 2205 |
| Maximum | 8990 |

Le paramètre choisi est `max_length=1800`, proche du p90. Cela signifie qu’environ 90 % des articles sont conservés entièrement, tandis que les plus longs sont tronqués. C’est un bon compromis entre performance et coût mémoire.

Cette approche évite de payer le coût d’un grand `max_length` pour chaque article, surtout si seulement une petite fraction des textes sont extrêmement longs.

## 7. Dataset PyTorch et DataLoader

Le fichier [src/dataset.py](./src/dataset.py) contient la classe `NewsDataset`, qui transforme les textes et labels en tenseurs PyTorch.

- `__init__` vérifie que les textes et labels ont la même longueur ;
- `__len__` retourne le nombre d’articles ;
- `__getitem__` encode un article, le complète ou le tronque, puis renvoie `(input_ids, label)`.

Les sorties sont de type `torch.long`, ce qui est nécessaire pour les embeddings et les labels d’entraînement.

La fonction `make_loaders()` crée ensuite trois DataLoader :

- train_loader ;
- val_loader ;
- test_loader.

Cela permet d’envoyer des batchs au modèle pendant l’entraînement et l’évaluation.

## 8. Architecture du modèle

Le modèle utilisé est un réseau simple et pédagogique :

1. les tokens sont transformés en embeddings ;
2. on calcule la moyenne des vecteurs de mots ;
3. on applique une couche linéaire ;
4. on ajoute une activation ReLU ;
5. on produit les logits pour les 5 classes.

La formule générale est :

Text -> Embedding -> Mean Pooling -> Linear -> ReLU -> Linear -> logits

Ce modèle est simple, robuste et facile à expliquer. Il est bien adapté à un projet de cours, tout en étant complètement fonctionnel.

## 9. Entraînement

Le fichier [src/train.py](./src/train.py) contient la boucle d’entraînement.

Les étapes clés sont :

- mise en mode apprentissage du modèle ;
- calcul des logits sur un batch ;
- calcul de la perte `CrossEntropyLoss` ;
- rétropropagation ;
- mise à jour des poids avec `Adam` ;
- évaluation sur la validation après chaque époque.

Les métriques suivies sont :

- loss ;
- accuracy ;
- macro-F1.

Le F1 macro est particulièrement utile ici car les classes peuvent être déséquilibrées.

Entraînement réalisé :

- 3 époques ;
- batch size = 16 ;
- max_length = 1800 ;
- learning rate = 1e-3.

Résultats obtenus après 3 époques :

- train loss : 1.1519
- validation accuracy : 0.5905
- validation macro-F1 : 0.4805
- test accuracy : 0.5833
- test macro-F1 : 0.4804

L’apprentissage progresse bien sur plusieurs époques, ce qui montre que le modèle apprend quelque chose de pertinent.

## 10. Courbes et visualisation

Le projet génère automatiquement plusieurs figures dans le dossier [models/figures](./models/figures) :

- `train_validation_loss.png` : évolution de la perte sur train et validation ;
- `validation_metrics.png` : évolution de l’accuracy et du F1 ;
- `confusion_matrix.png` : matrice de confusion sur le test.

Ces courbes sont très utiles pour expliquer le modèle à l’oral : elles montrent si le modèle apprend, si la validation se dégrade, et quelles classes sont les plus difficiles à classer.

## 11. Évaluation et analyse des erreurs

Le fichier [src/evaluate.py](./src/evaluate.py) permet d’évaluer le modèle et de comparer avec une baseline.

Les métriques calculées sont :

- loss ;
- accuracy ;
- macro-F1 ;
- matrice de confusion.

La baseline utilisée est une méthode classique :

- `TfidfVectorizer` ;
- `LogisticRegression`.

Résultats de la baseline :

- validation accuracy : 0.8524
- validation macro-F1 : 0.7563
- test accuracy : 0.8357
- test macro-F1 : 0.7180

Cela montre que la baseline est mieux performante que le modèle PyTorch simple à ce stade. Cela est normal pour un premier modèle neuronal très basique. Le but du projet n’était pas de battre immédiatement la baseline, mais de mettre en place le pipeline complet de NLP avec PyTorch et de comprendre les mécanismes d’apprentissage.

## 12. Tests et qualité du code

Les tests du projet sont validés via :

```bash
python -m pytest -q
```

Résultat final : **26 tests passés**.

Les tests couvrent notamment :

- tokenisation ;
- vocabulaire ;
- indices réservés ;
- padding/troncature ;
- encodage des tokens inconnus ;
- création des batchs ;
- validation du modèle.

## 13. Leçons apprises

Ce projet a permis de comprendre plusieurs points importants :

- le texte ne peut pas être donné directement à un réseau de neurones ;
- le vocabulaire est la clé pour passer du mot à un nombre ;
- les splits doivent rester propres pour éviter les fuites de données ;
- la troncature et le padding sont nécessaires pour travailler avec des batchs ;
- les métriques doivent être choisies selon le contexte ;
- le F1 macro est souvent plus informatif que l’accuracy seule ;
- un modèle simple peut apprendre, mais il est souvent moins fort qu’une baseline TF-IDF sur des données de texte classiques.

## 14. Conclusion

Le projet a permis de mettre en place un pipeline complet de classification de texte en français avec PyTorch :

- nettoyage des données ;
- tokenisation ;
- vocabulaire ;
- dataset et DataLoader ;
- modèle neuronal ;
- entraînement ;
- évaluation ;
- matrices et courbes.

L’objectif pédagogique est atteint : le pipeline est fonctionnel, lisible, explicable et prêt pour une présentation orale ou une démonstration.

## 15. Questions de présentation possibles

**1. Pourquoi le texte doit-il être converti en nombres ?**
Parce qu’un réseau neuronal manipule des vecteurs numériques, pas des chaînes de caractères.

**2. Pourquoi construire le vocabulaire sur le train uniquement ?**
Pour éviter toute fuite d’information et garder une vraie évaluation sur des données non vues.

**3. Que représente `<UNK>` ?**
Un mot qui n’est pas dans le vocabulaire, remplacé par un index de secours.

**4. Pourquoi utiliser le F1 macro ?**
Parce qu’il tient compte de toutes les classes de façon équilibrée, même si certaines sont moins représentées.

**5. Pourquoi la baseline TF-IDF est-elle meilleure ici ?**
Parce qu’un modèle linéaire sur des représentations TF-IDF est très robuste pour les tâches de classification textuelle et est souvent très performant dès le départ.

**6. Quelles sont les prochaines améliorations possibles ?**
Ajouter plus d’époques, choisir un meilleur modèle, utiliser des embeddings pré-entraînés, augmenter la taille du vocabulaire, tester des variantes de pooling ou des modèles plus avancés.
