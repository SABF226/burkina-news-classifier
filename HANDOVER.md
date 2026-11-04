# HANDOVER : Classificateur d'articles de presse (MasakhaNEWS fra)

## Objectif
Projet d'apprentissage à deux : classifier des articles de presse avec PyTorch
(nn.Embedding, nn.Linear, nn.ReLU, nn.CrossEntropyLoss).
Dataset : masakhane/masakhanews, config "fra" (CSV dans datasets/).

## Décisions prises
- 5 classes conservées : business, health, politics, sports, technology
  (aucun renommage, aucune suppression)
- Splits officiels train/validation/test, pas de re-split stratifié
- Métrique principale : F1 macro (technology : 76 exemples en train)
- Vocabulaire construit sur le train uniquement (évite la fuite de données)
- Nettoyage (src/data.py) :
  1. Doublon exact dans train : garder une copie (index 1427 supprimé)
  2. Article partagé entre splits avec labels différents : supprimé partout
     (index 209 du train)
  3. Article partagé entre splits avec même label : supprimé de val/test
  4. Conflit interne au train, même texte et labels différents :
     supprimé (index 933 et 1096)
- Tailles finales : train 1472, validation 210, test 420

## État d'avancement
| Élément | Statut |
|---|---|
| Audit du dataset (dataset_research.md) | Fait (commit à vérifier) |
| src/data.py + 6 tests pytest | Fait, 6/6 passés, commit a8ba7e3 |
| Règle 4 documentée dans data.py et dataset_research.md | À vérifier |
| Stats de longueur des textes (médiane, p90) | Non confirmées |
| src/vocab.py | Non confirmé, probablement non commencé |
| dataset.py, model.py, train.py, evaluate.py | À faire |
| Baseline TF-IDF, predict(), démo Gradio, README | À faire |

## Limites connues
- Licence du dataset non spécifiée dans les métadonnées : à vérifier sur Hugging Face
- Articles en français mais pas spécifiquement burkinabè : à écrire dans le README
  (corpus local Lefaso.net / Burkina24 en option, après vérification de leurs
  conditions d'utilisation)
- Python 3.14 est incompatible avec datasets 3.6 : utiliser Python 3.12 (.venv à la racine)

## Prochaine étape
1. Calculer les stats de longueur en mots sur le train nettoyé
   (médiane, p90, p95, max) et choisir max_len d'après le p90
2. Écrire src/vocab.py : tokenize, classe Vocab (<PAD>=0, <UNK>=1, min_freq=2),
   pad_or_truncate, save/load JSON, tests pytest

## Roadmap
- Sprint 1 (reste) : vocab.py, dataset.py (NewsDataset, make_loaders),
  sanity check d'un batch (batch, max_len)
- Sprint 2 : model.py (Embedding, moyenne, Linear, ReLU, Linear),
  train.py (CrossEntropyLoss, Adam, courbes train vs validation)
- Sprint 3 : evaluate.py (accuracy, F1 macro, matrice de confusion, analyse
  d'erreurs), baseline TF-IDF, predict(texte), démo Gradio, README

## Checklist pour ton coéquipier (à faire avant de coder)

```bash
git pull
git log --oneline          # a8ba7e3 doit apparaître
git status                 # fichiers non commités = travail perdu ou en cours
ls src tests models
python --version           # doit être 3.12.x dans .venv
pytest                     # les 6 tests de data.py doivent passer
```

Si `git status` montre des fichiers non commités (ex. `dataset_research.md`, `vocab.py` partiel), fais un commit avant de transmettre, sinon ils n'arriveront pas.

## Prompt de reprise pour son agent

```text
Reprends le projet décrit dans HANDOVER.md. D'abord, vérifie l'état réel :
git log, git status, ls src tests, pytest. Signale toute différence avec
HANDOVER.md. Puis enchaîne sur "Prochaine étape". Règles : explications en
français, code en anglais, une étape à la fois (concept, code, exécution,
mini-exercice), exécute toi-même les commandes, ne suppose rien sur les
données, arrête-toi après chaque fichier pour résumer.
```

Il devra aussi relire les concepts vus ensemble (problème et pipeline, texte vers nombres, embeddings). Je peux lui faire un résumé d'une page si tu veux, ou mettre le tout dans un Claude Doc partageable.
