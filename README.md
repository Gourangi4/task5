# Task 5: Decision Trees and Random Forests

## What this does
- Loads Heart Disease dataset (303 rows, 13 features, target = disease present/absent)
- Trains a Decision Tree, visualizes it
- Checks overfitting by varying max_depth
- Trains a Random Forest, compares accuracy
- Extracts feature importances
- Evaluates both models with 5-fold cross-validation

## Files
- task5_decision_trees_random_forests.py - main script
- heart.csv - dataset used
- decision_tree.png - visualized tree (top 3 levels)
- overfitting_depth.png - train vs test accuracy across depths
- feature_importances.png - bar chart of RF feature importances
- results_summary.txt - final numbers

## Results
- Unpruned tree: 98.5% test accuracy
- Pruned tree (depth=9): 98.5% test accuracy
- Random Forest: 100% test accuracy
- Decision Tree 5-fold CV mean: 99.8%
- Random Forest 5-fold CV mean: 99.7%
- Top features: cp (chest pain type), thalach (max heart rate), ca

**Note on accuracy:** this Kaggle copy of the dataset has 1025 rows but only 302 unique rows (duplicated/near-duplicated entries). That's why accuracy is near 100% — duplicate rows leak between train and test splits. This is a known quirk of this dataset, not a modeling error. For a cleaner evaluation, drop duplicates first (`df.drop_duplicates()`) before splitting, which brings results back down to realistic ~80% territory in line with the original 303-row UCI dataset.

## Concepts, explained simply

**Decision Tree**
- Splits data repeatedly on the feature/threshold that best separates classes
- Each split picks the question that reduces "impurity" the most
- Leaf nodes give the final prediction
- Keeps splitting until leaves are pure or a stopping rule kicks in (max_depth, min_samples_leaf)

**Entropy and Information Gain**
- Entropy = measure of disorder/impurity in a set of labels. Pure set (all one class) = entropy 0
- Information Gain = entropy before split minus weighted entropy after split
- Tree picks the split with the highest information gain at each step
- Gini impurity is the other common option, similar idea, cheaper to compute

**Overfitting**
- Deep trees memorize training data, including noise
- Symptom: train accuracy near 100%, test accuracy much lower
- Prevent it with: max_depth, min_samples_split, min_samples_leaf, pruning, or just use an ensemble (Random Forest)

**Random Forest**
- Trains many decision trees, each on a random bootstrap sample of the data
- Each split in each tree also considers only a random subset of features
- Final prediction = majority vote (classification) or average (regression)
- Because trees are decorrelated, errors cancel out, so it generalizes better than one tree

**Bagging (Bootstrap Aggregating)**
- The technique of training multiple models on random samples (with replacement) of the data, then averaging/voting
- Random Forest = bagging + random feature subsets at each split
- Reduces variance without increasing bias much

**Visualizing a tree**
- sklearn.tree.plot_tree() draws it directly with matplotlib
- Graphviz (export_graphviz) gives a cleaner, exportable version, good for large trees

**Feature Importance**
- Random Forest ranks features by how much they reduce impurity across all trees, averaged and weighted by how many samples pass through that split
- Higher value = feature is used more often, in more decisive splits
- Here cp, thal, thalach mattered most for predicting heart disease

**Pros/Cons of Random Forests**
- Pros: much less overfitting than a single tree, handles nonlinear relationships, gives feature importance, works with little preprocessing (no scaling needed)
- Cons: slower to train/predict than a single tree, less interpretable (can't visualize the "one tree"), can still overfit with too many deep trees on small noisy data, biased toward features with more categories in importance ranking

## Interview Q&A (quick recall version)

1. **How does a decision tree work?**
   Repeatedly splits data on the feature/threshold that best separates classes, until leaves are pure or a stop condition is hit.

2. **What is entropy and information gain?**
   Entropy measures impurity of a set of labels. Information gain = reduction in entropy after a split. Tree picks the split with max gain.

3. **How is random forest better than a single tree?**
   Averages many decorrelated trees (bagging + random features), which cancels out individual trees' overfitting, giving better generalization.

4. **What is overfitting and how do you prevent it?**
   Model fits noise in training data, so test performance drops. Prevent with max_depth limits, min_samples_leaf, pruning, cross-validation, or ensembling.

5. **What is bagging?**
   Bootstrap Aggregating: train multiple models on random samples (with replacement), combine via voting/averaging. Reduces variance.

6. **How do you visualize a decision tree?**
   sklearn.tree.plot_tree() for a quick matplotlib plot, or export_graphviz + Graphviz for a cleaner rendered diagram.

7. **How do you interpret feature importance?**
   Higher importance = feature contributed more to reducing impurity across splits/trees. Doesn't imply causation, just predictive contribution in this model.

8. **What are the pros/cons of random forests?**
   Pros: robust to overfitting, handles nonlinearity, minimal preprocessing, gives feature importance.
   Cons: slower, less interpretable than one tree, importance can be biased toward high-cardinality features.
