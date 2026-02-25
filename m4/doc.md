# Récapitulatif complet du pipeline ML — Prédiction du Financial Health Index (FHI)

> **Objectif :** Prédire la classe de santé financière (`Low`, `Medium`, `High`) de PME d'Afrique australe (Eswatini, Lesotho, Zimbabwe, Malawi) à partir de données socio-économiques et d'enquête.
> **Métrique d'évaluation :** F1-score macro (moyenne non pondérée du F1 sur les 3 classes).

---

## 1. Données d'entrée

| Fichier                   | Rôle                                 | Dimensions                 |
| ------------------------- | ------------------------------------ | -------------------------- |
| `Train.csv`               | Données d'entraînement avec la cible | 9 618 lignes × 39 colonnes |
| `Test.csv`                | Données de test (sans cible)         | 2 405 lignes × 38 colonnes |
| `SampleSubmission.csv`    | Format attendu pour la soumission    | 2 405 lignes × 2 colonnes  |
| `VariableDefinitions.csv` | Dictionnaire des variables           | —                          |

### Variables brutes (38 features)

Les colonnes originales couvrent 5 grandes catégories :

- **Identité du propriétaire** : `owner_age`, `owner_sex`, `country`
- **Finances de l'entreprise** : `personal_income`, `business_expenses`, `business_turnover`, `business_age_years`, `business_age_months`
- **Accès aux services financiers** : `has_mobile_money`, `has_loan_account`, `has_internet_banking`, `has_debit_card`, `has_credit_card`
- **Couverture assurantielle** : `motor_vehicle_insurance`, `medical_insurance`, `funeral_insurance`, `has_insurance`
- **Attitudes, perceptions, comportements** : ~20 colonnes binaires/ordinales (ex. `attitude_stable_business_environment`, `keeps_financial_records`, `compliance_income_tax`)

### Distribution de la cible (déséquilibre de classes)

| Classe | Effectif | Proportion |
| ------ | -------- | ---------- |
| Low    | 6 280    | 65 %       |
| Medium | 2 868    | 30 %       |
| High   | 470      | 5 %        |

> ⚠️ Fort déséquilibre : la classe `High` est 13× moins représentée que `Low`. Le modèle utilise `class_weight='balanced'` pour compenser.

---

## 2. Feature Engineering — 17 nouvelles variables

La fonction `engineer_features(df)` crée 17 variables supplémentaires à partir des données brutes. L'idée est de synthétiser l'information pertinente que le modèle aurait du mal à construire seul.

### 2.1 Ratios financiers (3 variables)

```python
profit_margin        = (personal_income - business_expenses) / personal_income
expense_to_turnover  = business_expenses / business_turnover
income_to_turnover   = personal_income   / business_turnover
```

- **`profit_margin`** : Mesure l'efficacité financière. Une valeur proche de 1 signifie que l'entreprise garde presque tout son revenu après dépenses. Valeurs bornées entre −1 et 1 pour éviter les outliers extrêmes.
- **`expense_to_turnover`** : Ratio dépenses/chiffre d'affaires. Un ratio > 1 indique que l'entreprise dépense plus qu'elle ne gagne → signe de fragilité financière.
- **`income_to_turnover`** : Mesure la part du chiffre d'affaires qui revient personnellement au propriétaire, indicateur de rentabilité personnelle.

### 2.2 Transformations logarithmiques (3 variables)

```python
log_personal_income   = log1p(personal_income)
log_business_expenses = log1p(business_expenses)
log_business_turnover = log1p(business_turnover)
```

Les montants financiers suivent souvent une distribution log-normale (quelques très grandes valeurs « tirent » la moyenne). Le `log1p` (= log(1+x)) compresse cette distribution et rend le modèle moins sensible aux extrêmes. `log1p` est préféré à `log` car il gère les zéros sans erreur.

### 2.3 Maturité de l'entreprise (3 variables)

```python
total_business_age_months = business_age_years * 12 + business_age_months
is_young_business         = 1 si age < 24 mois
is_established            = 1 si age >= 60 mois
```

- **`total_business_age_months`** : Convertit l'âge en une seule unité (mois) pour faciliter les comparaisons.
- **`is_young_business`** : Indicateur binaire — les entreprises de moins de 2 ans sont typiquement plus fragiles (pas encore stabilisées).
- **`is_established`** : Indicateur binaire — les entreprises de plus de 5 ans ont prouvé leur résilience.

### 2.4 Démographie du propriétaire (2 variables)

```python
owner_age_bin = tranche d'âge (<25, 25-35, 35-45, 45-55, 55+)
is_female     = 1 si owner_sex == 'female'
```

- **`owner_age_bin`** : Discrétise l'âge en 5 tranches. Utile car la relation entre l'âge et la santé financière n'est pas linéaire.
- **`is_female`** : Variable binaire qui capture d'éventuelles disparités de genre dans l'accès aux services financiers.

### 2.5 Scores composites (5 variables) — cœur du FE

Ces scores synthétisent plusieurs colonnes binaires/ordinales en un seul indicateur normalisé entre 0 et 1. La logique de scoring est la suivante :

- Réponse positive (`Yes`, `Have now`, `Yes, always`, etc.) → **+1 point**
- Réponse partielle (`Used to have but don't have now`) → **+0.5 point**
- Valeur manquante → ignorée (le dénominateur n'est pas incrémenté)
- Score final = somme / nombre de colonnes non-nulles

#### `financial_access_score`

> Colonnes : `has_loan_account`, `has_internet_banking`, `has_debit_card`, `has_credit_card`, `has_mobile_money`

Mesure l'inclusion financière formelle. Un score de 1.0 signifie que l'entreprise utilise tous les services financiers formels disponibles.

#### `insurance_score`

> Colonnes : `motor_vehicle_insurance`, `medical_insurance`, `funeral_insurance`, `has_insurance`

Mesure la couverture assurantielle. Les entreprises bien assurées sont plus résilientes aux chocs.

#### `digital_adoption_score`

> Colonnes : `has_cellphone`, `has_mobile_money`, `has_internet_banking`

Mesure l'adoption des outils numériques. Forte corrélation attendue avec l'accès au crédit et à la formalisation.

#### `resilience_score`

> Colonnes : `attitude_stable_business_environment`, `attitude_satisfied_with_achievement`, `attitude_more_successful_next_year`

Mesure les attitudes positives du propriétaire. Un score élevé indique un optimisme et une confiance en l'avenir de l'entreprise.

#### `formality_score`

> Colonnes : `keeps_financial_records`, `compliance_income_tax`, `has_loan_account`

Mesure le degré de formalisation de l'entreprise. Les entreprises formelles (comptabilité, déclarations fiscales) ont généralement un meilleur accès au crédit.

#### `composite_health_score`

```python
composite_health_score = moyenne(financial_access_score, insurance_score,
                                  digital_adoption_score, resilience_score, formality_score)
```

Score agrégé résumant les 4 dimensions du FHI dans une seule valeur. C'est la 7ème feature la plus importante selon le modèle.

### 2.6 Quantile de revenu (1 variable)

```python
income_tier = quantile de log_personal_income en 5 tranches (0 à 4)
```

Divise les propriétaires en 5 quintiles de revenu (du plus pauvre au plus riche) sur l'échelle logarithmique, ce qui est plus robuste aux outliers qu'une simple discrétisation.

---

## 3. Pipeline de prétraitement (Encoding)

La fonction `encode_features()` transforme toutes les colonnes textuelles en valeurs numériques que LightGBM peut traiter.

### 3.1 Encodage ordinal (30 colonnes)

Les colonnes catégorielles ont un **ordre naturel** (ex. : `Never had < Used to have < Have now`). On leur attribue des entiers correspondant à leur rang dans cet ordre :

| Échelle        | Valeurs → Entiers                              |
| -------------- | ---------------------------------------------- |
| `HAVE_ORDER`   | `Never had`→0, `Used to have…`→1, `Have now`→2 |
| `YES_NO`       | `No`→0, `Yes`→1, `Don't know…`→2, etc.         |
| `CREDIT_ORDER` | `No`→0, `Yes, sometimes`→1, `Yes, always`→2    |
| `SEX_ORDER`    | `Female`→0, `Male`→1                           |

> ✅ Cet encodage est **préférable à un one-hot encoding** ici car il préserve le sens ordinal et ne crée pas de colonnes supplémentaires.
> Toute valeur non reconnue dans le mapping est convertie en `NaN`.

### 3.2 Label encoding (colonnes restantes)

Les colonnes `object`/`category` non couvertes par l'ordinal mapping (ex. `country`, `owner_age_bin`) sont encodées avec `LabelEncoder`. Le `LabelEncoder` est fitté sur train+test combinés pour éviter les catégories inconnues au moment de la prédiction.

### 3.3 Gestion des valeurs manquantes

**Aucune imputation** n'est appliquée. LightGBM gère nativement les `NaN` en apprenant le meilleur chemin de split pour chaque valeur manquante. Cela évite d'introduire des biais artificiels liés à une imputation incorrecte.

Les valeurs `±inf` créées par les ratios (ex. division par zéro) sont converties en `NaN`.

---

## 4. Entraînement — LightGBM avec Validation Croisée 5-Fold

### 4.1 Pourquoi LightGBM ?

LightGBM (Light Gradient Boosting Machine) est un algorithme basé sur les arbres de décision en gradient boosting. Ses avantages clés :

- **Rapide** : utilise un algorithme de croissance feuille par feuille (leaf-wise) au lieu de niveau par niveau
- **Efficace sur les données tabulaires** : souvent le meilleur algorithm sur les compétitions Kaggle/Zindi
- **Gestion native des NaN et des catégories**
- **Robuste aux outliers**

### 4.2 Hyperparamètres principaux

| Paramètre                | Valeur       | Rôle                                                                                |
| ------------------------ | ------------ | ----------------------------------------------------------------------------------- |
| `objective`              | `multiclass` | Classification à 3 classes                                                          |
| `n_estimators`           | 1 000        | Nombre max d'arbres (avec early stopping)                                           |
| `learning_rate`          | 0.05         | Taux d'apprentissage faible → meilleure généralisation                              |
| `num_leaves`             | 63           | Complexité des arbres (2^6−1)                                                       |
| `subsample`              | 0.8          | 80 % des lignes tirées aléatoirement par arbre                                      |
| `colsample_bytree`       | 0.8          | 80 % des features utilisées par arbre                                               |
| `reg_alpha / reg_lambda` | 0.1          | Régularisation L1 et L2 pour éviter le surapprentissage                             |
| `class_weight`           | `balanced`   | Pondère les classes inversement à leur fréquence → compense le déséquilibre         |
| `early_stopping`         | 50 rounds    | Arrête l'entraînement si la perte de validation ne s'améliore pas sur 50 itérations |

### 4.3 Validation croisée stratifiée (StratifiedKFold, k=5)

```
Train (9 618 exemples)
├── Fold 1 : train=7 694 | validation=1 924
├── Fold 2 : train=7 695 | validation=1 923
├── Fold 3 : train=7 695 | validation=1 923
├── Fold 4 : train=7 695 | validation=1 923
└── Fold 5 : train=7 695 | validation=1 923
```

**Stratifié** signifie que chaque fold respecte la même distribution des classes que le jeu de données complet. Cela est crucial ici car `High` ne représente que 5 % des données — sans stratification, certains folds pourraient ne contenir presque aucun exemple `High`.

Pour chaque fold :

1. Un nouveau modèle LightGBM est entraîné sur les 80 % restants
2. Les prédictions sur les 20 % de validation sont stockées dans `oof_preds` (Out-Of-Fold predictions)
3. Les prédictions sur le jeu de test complet sont accumulées dans `test_preds` (divisées par 5)
4. Le F1 macro du fold est calculé et affiché

### 4.4 Résultats obtenus

| Fold           | Best iteration | Macro-F1            |
| -------------- | -------------- | ------------------- |
| 1              | 97             | 0.7934              |
| 2              | 126            | 0.8196              |
| 3              | 132            | 0.7932              |
| 4              | 122            | 0.8013              |
| 5              | 115            | 0.8067              |
| **Moyenne**    | —              | **0.8028 ± 0.0098** |
| **OOF global** | —              | **0.8029**          |

**Rapport de classification OOF :**

| Classe        | Précision | Rappel   | F1       | Support |
| ------------- | --------- | -------- | -------- | ------- |
| High          | 0.76      | 0.68     | 0.72     | 470     |
| Low           | 0.91      | 0.92     | 0.92     | 6 280   |
| Medium        | 0.78      | 0.77     | 0.78     | 2 868   |
| **Macro avg** | **0.82**  | **0.79** | **0.80** | 9 618   |

> `Low` est bien prédit (F1=0.92) car dominant. `High` est le plus difficile (F1=0.72) car rare et potentiellement ambigu avec `Medium`.

---

## 5. Analyse de l'importance des features

Les importances sont **moyennées sur les 5 folds** pour être plus stables (une importance issue d'un seul fold peut fluctuer).

### Top 10 features par importance LGBM

| Rang | Feature                     | Catégorie                  |
| ---- | --------------------------- | -------------------------- |
| 1    | `expense_to_turnover`       | Ratio financier engineered |
| 2    | `income_to_turnover`        | Ratio financier engineered |
| 3    | `owner_age`                 | Démographie brute          |
| 4    | `business_expenses`         | Finance brute              |
| 5    | `personal_income`           | Finance brute              |
| 6    | `business_turnover`         | Finance brute              |
| 7    | `composite_health_score`    | Score composite engineered |
| 8    | `profit_margin`             | Ratio financier engineered |
| 9    | `total_business_age_months` | Maturité engineered        |
| 10   | `business_age_months`       | Finance brute              |

> **Observation clé :** Les 2 features les plus importantes (`expense_to_turnover`, `income_to_turnover`) sont des variables **créées** par feature engineering — elles n'existaient pas dans les données brutes. Cela valide l'utilité du FE.

---

## 6. Génération de la soumission — Soft Voting Ensemble

Au lieu d'utiliser un seul modèle, le pipeline utilise les **5 modèles des 5 folds** pour prédire le jeu de test.

```python
# Pour chaque fold i :
test_preds += model_i.predict_proba(X_test) / 5
```

`predict_proba` retourne un vecteur de 3 probabilités pour chaque exemple, ex. `[0.05, 0.70, 0.25]` → probabilité d'être `High`, `Low`, `Medium`.

La **moyenne** de ces probabilités sur 5 modèles (soft voting) est plus robuste qu'un seul modèle ou qu'un vote majoritaire (hard voting), car elle lisse les incertitudes.

La classe finale est celle avec la probabilité moyenne maximale :

```python
test_labels = argmax(test_preds, axis=1)  # → [1, 2, 1, 0, ...]
test_labels = target_le.inverse_transform(test_labels)  # → ['Low', 'Medium', 'Low', 'High', ...]
```

### Distribution des prédictions sur le jeu de test

| Classe | Effectif |
| ------ | -------- |
| Low    | 1 593    |
| Medium | 709      |
| High   | 103      |

Le fichier `submission.csv` contient 2 405 lignes avec les colonnes `ID` et `Target`.

---

## 7. Résumé du flux complet

```
Train.csv / Test.csv
       │
       ▼
engineer_features()      → +17 nouvelles features (ratios, scores composites, log)
       │
       ▼
encode_features()        → Ordinal encoding (30 cols) + Label encoding (reste)
       │                    NaN conservés (LightGBM les gère nativement)
       ▼
StratifiedKFold (k=5)    → 5 splits stratifiés (préserve la distribution des classes)
       │
       ▼
LGBMClassifier × 5       → 1 modèle par fold, early stopping sur 50 rounds
       │                    class_weight='balanced' contre le déséquilibre
       │
       ├── oof_preds  →  OOF Macro-F1 = 0.8029  (évaluation non biaisée)
       └── test_preds →  moyenne des probabilités sur 5 folds (soft voting)
                               │
                               ▼
                        submission.csv  (ID + Target)
```

---

## 8. Points forts et pistes d'amélioration

### ✅ Points forts

- Feature engineering solide couvrant les 4 dimensions du FHI
- Validation croisée stratifiée → estimation fiable de la performance réelle
- Soft voting ensemble → prédictions plus stables
- Gestion du déséquilibre avec `class_weight='balanced'`
- Pas d'imputation manuelle → LGBM apprend la meilleure stratégie pour les NaN

### 🚀 Pistes d'amélioration

1. **Optimisation des hyperparamètres** avec Optuna (bayesian search sur `num_leaves`, `learning_rate`, etc.)
2. **Target encoding** pour la variable `country` (encoder la moyenne de la cible par pays)
3. **Features d'interaction** (ex. `financial_access_score × income_tier`)
4. **Ensemble multi-modèles** : ajouter XGBoost ou CatBoost et moyenner les probabilités
5. **Augmentation de données** pour la classe `High` (SMOTE ou sur-échantillonnage)
6. **Pseudo-labelling** : utiliser les prédictions très confiantes du test pour ré-entraîner
