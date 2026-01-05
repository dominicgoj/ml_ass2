
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from baseline_model import flatten_clf_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os
from sklearn.neighbors import KNeighborsClassifier

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

def main():
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


if __name__ == '__main__':
    os.makedirs("exports/tables", exist_ok=True)
    df = pd.read_csv("D.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    main()

    
    