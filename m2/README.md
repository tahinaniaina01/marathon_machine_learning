# K-Nearest Neighbor (KNN) — Explications du code et concepts mathématiques

---

## 1. Distance Euclidienne

### Concept mathématique

La **distance euclidienne** entre deux vecteurs $a, b \in \mathbb{R}^D$ est définie par :

$$\|a - b\| = \sqrt{\sum_{d=1}^{D} (a_d - b_d)^2}$$

C'est la distance "à vol d'oiseau" dans un espace à $D$ dimensions. C'est la métrique utilisée dans le KNN pour mesurer la similarité entre deux points.

---

## 2. Matrice des distances

On cherche à calculer une matrice $\text{dist} \in \mathbb{R}^{N_{test} \times N_{train}}$ où :

$$\text{dist}[i, j] = \|X_{test}[i] - X_{train}[j]\|$$

### 2.1 Implémentation avec deux boucles (`get_distances_two_loops`)

```python
diff = X_test[i] - X_train[j]
distances[i, j] = np.sqrt(np.dot(diff, diff))
```

**Explication :**

- On soustrait les deux vecteurs pour obtenir le vecteur différence.
- `np.dot(diff, diff)` calcule le produit scalaire du vecteur avec lui-même, ce qui donne $\sum_d (a_d - b_d)^2$ — c'est-à-dire le **carré de la norme**.
- On prend la racine carrée pour obtenir la distance euclidienne.

> On évite `np.linalg.norm()` et à la place on utilise $\|v\|^2 = v \cdot v$.

**Complexité :** $O(N_{test} \times N_{train} \times D)$ — très lent pour de grands datasets.

---

### 2.2 Implémentation avec une boucle (`compute_distances_one_loop`)

```python
diff = X_test[i] - X_train           # shape (num_train, D)
distances[i, :] = np.sqrt(np.sum(diff ** 2, axis=1))
```

**Explication :**

- Pour un point test $X_{test}[i]$ fixé, on soustrait simultanément **tous les points d'entraînement** grâce au **broadcasting** NumPy.
- `X_test[i]` a la forme `(D,)` et `X_train` a la forme `(N_{train}, D)` → NumPy étend automatiquement `X_test[i]` pour faire la soustraction ligne par ligne.
- `np.sum(diff ** 2, axis=1)` somme sur la dimension des features pour chaque point d'entraînement.
- On prend la racine carrée de chaque résultat.

**Broadcasting NumPy :** si `a` a la forme `(D,)` et `b` a la forme `(N, D)`, alors `b - a` donne un tableau de forme `(N, D)` où chaque ligne est `b[i] - a`.

> On n'a plus qu'une seule boucle sur les points test.

---

### 2.3 Implémentation sans boucle (`get_distances_zero_loop`)

#### Identité algébrique clé

On utilise l'identité :

$$\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2\langle a, b \rangle$$

**Démonstration :**
$$\|a - b\|^2 = (a - b)^T(a - b) = a^Ta - 2a^Tb + b^Tb = \|a\|^2 - 2\langle a,b\rangle + \|b\|^2$$

#### Application à la matrice de distances

En étendant cette identité à toutes les paires $(i, j)$ d'un coup :

$$\text{dist}[i,j]^2 = \|X_{test}[i]\|^2 + \|X_{train}[j]\|^2 - 2 \cdot X_{test}[i] \cdot X_{train}[j]^T$$

En notation matricielle :

$$D^2 = \mathbf{t}_{test} \cdot \mathbf{1}^T + \mathbf{1} \cdot \mathbf{t}_{train}^T - 2 \cdot X_{test} X_{train}^T$$

où $\mathbf{t}_{test}[i] = \|X_{test}[i]\|^2$ et $\mathbf{t}_{train}[j] = \|X_{train}[j]\|^2$.

```python
test_sq  = np.sum(X_test  ** 2, axis=1, keepdims=True)  # (num_test, 1)
train_sq = np.sum(X_train ** 2, axis=1)                  # (num_train,)
distances = np.sqrt(test_sq + train_sq - 2 * X_test.dot(X_train.T))
```

**Explication ligne par ligne :**

- `test_sq` : norme au carré de chaque point test, forme `(N_test, 1)`.
- `train_sq` : norme au carré de chaque point train, forme `(N_train,)`.
- `X_test.dot(X_train.T)` : produit matriciel donnant tous les produits scalaires $\langle X_{test}[i], X_{train}[j] \rangle$, forme `(N_test, N_train)`.
- Le broadcasting additionne `test_sq` `(N_test, 1)` + `train_sq` `(N_train,)` → `(N_test, N_train)`.
- `np.sqrt(...)` applique la racine carrée élément par élément.

> Aucune boucle Python — tout est vectorisé, beaucoup plus rapide.

---

## 3. Classifieur KNN (`KNearestNeighborClassifier`)

### Concept mathématique

L'algorithme KNN est un **classifieur non-paramétrique** basé sur la règle de vote majoritaire :

Étant donné un point $x_{test}$, on trouve les $k$ points d'entraînement les plus proches (au sens de la distance euclidienne) et on prédit la classe la plus fréquente parmi eux :

$$\hat{y} = \arg\max_{c} \sum_{j \in \mathcal{N}_k(x)} \mathbf{1}[y_j = c]$$

où $\mathcal{N}_k(x)$ désigne l'ensemble des $k$ voisins les plus proches de $x$.

### Implémentation

#### `fit`

Mémorise simplement les données d'entraînement (apprentissage paresseux, _lazy learning_).

#### `predict_labels`

```python
closest_y = self.y_train[np.argsort(distances[i])[:self.k]]
y_pred[i] = np.bincount(closest_y.astype(int)).argmax()
```

- `np.argsort(distances[i])` : retourne les **indices triés** par distance croissante.
- `[:self.k]` : on garde les $k$ premiers indices (les $k$ plus proches voisins).
- `self.y_train[...]` : on récupère les étiquettes de ces $k$ voisins.
- `np.bincount(...)` : compte combien de fois chaque label apparaît parmi les $k$ voisins.
- `.argmax()` : retourne le label le plus fréquent — en cas d'égalité, le label avec l'indice le plus petit est retourné.

---

## 4. Choix de $k$ par Cross-Validation

### Concept mathématique

La **k-fold cross-validation** est une technique d'évaluation qui permet de choisir des hyperparamètres (ici $k$) sans biais lié à la taille du jeu de test.

**Principe :**

1. Diviser l'ensemble d'entraînement en $F$ plis (_folds_) de taille égale.
2. Pour chaque pli $f \in \{1, \ldots, F\}$ :
   - Utiliser le pli $f$ comme ensemble de validation.
   - Entraîner sur les $F-1$ plis restants.
   - Calculer le score (accuracy) sur la validation.
3. La performance pour un hyperparamètre donné est la **moyenne des $F$ scores**.
4. Choisir l'hyperparamètre qui maximise ce score moyen.

$$\text{CV}(k) = \frac{1}{F} \sum_{f=1}^{F} \text{Accuracy}(f, k)$$

### Implémentation

```python
X_train_folds = np.array_split(X_train3, num_folds)
y_train_folds = np.array_split(y_train3, num_folds)
```

- `np.array_split` découpe les données en `num_folds` parties (aussi équilibrées que possible).

```python
for k in k_choices:
    k_to_accuracies[k] = []
    for fold in range(num_folds):
        X_val = X_train_folds[fold]
        y_val = y_train_folds[fold]
        X_tr  = np.vstack([X_train_folds[i] for i in range(num_folds) if i != fold])
        y_tr  = np.hstack([y_train_folds[i] for i in range(num_folds) if i != fold])

        clf = KNearestNeighborClassifier(k=k)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        k_to_accuracies[k].append(acc)
```

- Pour chaque valeur de $k$ et chaque pli :
  - Le pli courant devient la **validation**, les autres sont concaténés en **train**.
  - `np.vstack` empile les tableaux de features verticalement.
  - `np.hstack` concatène les tableaux de labels.
  - On entraîne et évalue le modèle, puis on stocke l'accuracy.

---

## 5. Résumé des concepts

| Concept                 | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| Distance euclidienne    | $\|a - b\| = \sqrt{\sum_d (a_d - b_d)^2}$                            |
| Produit scalaire        | $\langle a, b\rangle = \sum_d a_d b_d = a \cdot b$                   |
| Broadcasting NumPy      | Extension automatique des dimensions pour les opérations vectorisées |
| Identité algébrique     | $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2\langle a,b\rangle$                |
| Vote majoritaire KNN    | Prédire la classe la plus fréquente parmi les $k$ voisins            |
| `np.argsort`            | Retourne les indices qui trieraient le tableau par ordre croissant   |
| `np.bincount`           | Compte les occurrences de chaque entier non-négatif                  |
| k-fold cross-validation | Évaluer un hyperparamètre en faisant tourner $F$ splits train/val    |
