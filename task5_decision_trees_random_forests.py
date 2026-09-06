"""
Task 5: Decision Trees and Random Forests
Dataset: Heart Disease (UCI, via GitHub mirror)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------- 1. Load data ----------
df = pd.read_csv("heart.csv")
print("Shape:", df.shape)
print(df.head())

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 2. Train a Decision Tree and visualize it ----------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
print("\nDecision Tree (no depth limit) accuracy:", accuracy_score(y_test, dt_preds))

plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"],
          filled=True, max_depth=3, fontsize=8)
plt.title("Decision Tree (top 3 levels shown)")
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- 3. Analyze overfitting vs tree depth ----------
train_scores, test_scores, depths = [], [], range(1, 15)
for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, model.predict(X_train)))
    test_scores.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, marker="o", label="Train Accuracy")
plt.plot(depths, test_scores, marker="o", label="Test Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Check: Accuracy vs Tree Depth")
plt.legend()
plt.grid(True)
plt.savefig("overfitting_depth.png", dpi=150, bbox_inches="tight")
plt.close()

best_depth = depths[test_scores.index(max(test_scores))]
print(f"\nBest depth on test set: {best_depth} (test acc = {max(test_scores):.3f})")

dt_pruned = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_pruned.fit(X_train, y_train)
print("Pruned Decision Tree accuracy:", accuracy_score(y_test, dt_pruned.predict(X_test)))

# ---------- 4. Train a Random Forest and compare ----------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("\nRandom Forest accuracy:", accuracy_score(y_test, rf_preds))

# ---------- 5. Feature importances ----------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):")
print(importances)

plt.figure(figsize=(8, 6))
importances.plot(kind="barh")
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- 6. Cross-validation ----------
dt_cv = cross_val_score(DecisionTreeClassifier(max_depth=best_depth, random_state=42), X, y, cv=5)
rf_cv = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42), X, y, cv=5)

print(f"\nDecision Tree 5-fold CV: mean={dt_cv.mean():.3f}, scores={dt_cv.round(3)}")
print(f"Random Forest 5-fold CV: mean={rf_cv.mean():.3f}, scores={rf_cv.round(3)}")

# ---------- Summary ----------
summary = f"""
SUMMARY
-------
Unpruned Decision Tree test accuracy : {accuracy_score(y_test, dt_preds):.3f}
Pruned Decision Tree (depth={best_depth}) test accuracy : {accuracy_score(y_test, dt_pruned.predict(X_test)):.3f}
Random Forest test accuracy          : {accuracy_score(y_test, rf_preds):.3f}
Decision Tree 5-fold CV mean         : {dt_cv.mean():.3f}
Random Forest 5-fold CV mean        : {rf_cv.mean():.3f}
Top 3 important features            : {list(importances.index[:3])}
"""
print(summary)

with open("results_summary.txt", "w") as f:
    f.write(summary)
