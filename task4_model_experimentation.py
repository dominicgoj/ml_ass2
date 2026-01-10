
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from task3_baseline_model import flatten_clf_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os
from sklearn.neighbors import KNeighborsClassifier
import itertools
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    flat = flatten_clf_report(report)

    flat["accuracy"] = accuracy_score(y_test, y_pred)
    flat["macro_f1"] = f1_score(y_test, y_pred, average="macro")

    return flat

def split_data(df, id_col='id', label_col='label'):
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[label_col],
        test_size=0.2, random_state=42, stratify=df[label_col]
    )
    return X_train, X_test, y_train, y_test

def preprocess_linear(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def preprocess_tree(X_train, X_test):
    return X_train.values, X_test.values

def run_logistic_regression(X_train, X_test, y_train, y_test,
                            C=1.0, class_weight=None, max_iter=1000):

    X_tr, X_te = preprocess_linear(X_train, X_test)

    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter
    )

    results = evaluate_model(model, X_tr, X_te, y_train, y_test)
    results.update({
        "model": "LogisticRegression",
        "C": C,
        "class_weight": class_weight
    })
    return results

def run_knn_baseline(X_train, X_test, y_train, y_test, k=15):

    # kNN braucht Skalierung
    X_tr, X_te = preprocess_linear(X_train, X_test)

    model = KNeighborsClassifier(n_neighbors=k)

    results = evaluate_model(model, X_tr, X_te, y_train, y_test)
    results.update({
        "model": "kNN",
        "k": k
    })
    return results

def run_random_forest(X_train, X_test, y_train, y_test,
                      n_estimators=200, max_depth=None,
                      min_samples_leaf=1, class_weight=None):

    X_tr, X_te = preprocess_tree(X_train, X_test)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight=class_weight
    )

    results = evaluate_model(model, X_tr, X_te, y_train, y_test)
    results.update({
        "model": "RandomForest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight
    })
    return results

def run_gradient_boosting(X_train, X_test, y_train, y_test,
                          n_estimators=200, learning_rate=0.1, max_depth=3):

    X_tr, X_te = preprocess_tree(X_train, X_test)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
    )

    results = evaluate_model(model, X_tr, X_te, y_train, y_test)
    results.update({
        "model": "GradientBoosting",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        
    })
    return results

def run_xgboost_multiclass(X_train, X_test, y_train, y_test,
                           n_estimators=300,
                           learning_rate=0.1,
                           max_depth=3,
                           subsample=0.8,
                           colsample_bytree=0.8):

    X_tr, X_te = preprocess_tree(X_train, X_test)

    num_classes = len(set(y_train))

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    results = evaluate_model(model, X_tr, X_te, y_train, y_test)
    results.update({
        "model": "XGBoost",
        "task": "multiclass",
        "num_classes": num_classes,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree
    })

    return results

def run_different_algorithms():
    all_results = []
    # ===== Baseline kNN =====
    all_results.append(
        run_knn_baseline(X_train, X_test, y_train, y_test, k=15)
    )
    print(f"Finished base line model. K 15.")

    # ---- Logistic Regression experiments ----
    for C in [0.01, 0.1, 1.0]:
        all_results.append(
            run_logistic_regression(X_train, X_test, y_train, y_test,
                                    C=C, class_weight=None)
        )
        all_results.append(
            run_logistic_regression(X_train, X_test, y_train, y_test,
                                    C=C, class_weight="balanced")
        )
        print(f"Finished Logistic Regression. HP: C {C}")

    # ---- Random Forest experiments ----
    for depth in [None, 3, 5, 7]:
        for n_estimator in [100, 200, 400, 500]:
            all_results.append(
                run_random_forest(X_train, X_test, y_train, y_test,
                                n_estimators=n_estimator, max_depth=depth,
                                class_weight=None)
            )
            all_results.append(
                run_random_forest(X_train, X_test, y_train, y_test,
                                n_estimators=n_estimator, max_depth=depth,
                                class_weight="balanced")
            )
            print(f"Finished Random Forest. HP: Depth {depth}, n_estimator {n_estimator}")
            

    # ---- Gradient Boosting experiments ----
    for lr in [0.005, 0.01, 0.05, 0.1]:
        for md in [3, 5, 7, 9, 11]:
            for n_estimator in [100, 200, 400, 500, 600, 700]:
                all_results.append(
                    run_gradient_boosting(X_train, X_test, y_train, y_test,
                                        n_estimators=n_estimator, learning_rate=lr, max_depth=md)
                )
                print(f"Finished Gradient Boosting. HP: Learning rate {lr}, max_depth {md}, n_estimator {n_estimator}")
            

    # ---- Export ----
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("exports/tables/task4_all_experiments.csv", sep=";", index=False)

    print("Saved results to exports/tables/task4_all_experiments.csv")
    return


def test_xgboost_hyperparameters(X_train, X_test, y_train, y_test):
    results = []

    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500, 600],
        "learning_rate": [0.05, 0.1, 0.01],
        "max_depth": [2, 3, 4, 5, 6, 7, 8],
        "subsample": [0.8, 1.0, 0.6],
        "colsample_bytree": [0.8, 1.0, 0.6]
    }

    keys = param_grid.keys()
    values = param_grid.values()

    combinations = list(itertools.product(*values))
    total = len(combinations)

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        result = run_xgboost_multiclass(
            X_train, X_test, y_train, y_test,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"]
        )

        results.append(result)

        print(f"[{i+1:3d}/{total}] XGB params: {params}")

    return results

def decrease_feature_space(X_train, X_test, max_remove_features=4):
    feature_names = list(X_train.columns)
    results = []

    for k in range(1, max_remove_features + 1):
        for features_to_remove in itertools.combinations(feature_names, k):
            
            X_train_reduced = X_train.drop(columns=list(features_to_remove))
            X_test_reduced  = X_test.drop(columns=list(features_to_remove)) 

            results.append({
                "removed_features": features_to_remove,
                "X_train": X_train_reduced,
                "X_test": X_test_reduced
            })

    return results

def test_feature_space(X_train, X_test, y_train, y_test):
    results = []
    table_combinations = decrease_feature_space(X_train, X_test)
    for i, table_comb in enumerate(table_combinations):
        x_train = table_comb['X_train']
        x_test = table_comb['X_test']
        removed_features = table_comb['removed_features']
        result = run_random_forest(x_train, x_test, y_train, y_test, n_estimators=400)
        result["removed_features"] = ",".join(removed_features)
        result["num_features"] = x_train.shape[1]

        results.append(result)
        print(f"Finished {i}/{len(table_combinations)}")
    results_df = pd.DataFrame(results)
    results_df.to_csv("exports/tables/task4_randomforest_feature_space.csv", sep=";", index=False)

    print("Saved results to exports/tables/task4_randomforest_feature_space.csv")

def plot_confusion_matrix_best_model(X_train, X_test, y_train, y_test):
    X_tr, X_te = preprocess_tree(X_train, X_test)

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42
    )

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1, 2, 3]
    )

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – Random Forest (Task 4 Best Model)")
    plt.tight_layout()
    plt.savefig("exports/cm.png", dpi=200)

if __name__ == '__main__':
    os.makedirs("exports/tables", exist_ok=True)
    df = pd.read_csv("D.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    plot_confusion_matrix_best_model(X_train, X_test, y_train, y_test)

    
    