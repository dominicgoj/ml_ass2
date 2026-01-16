
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
from task3_baseline_model import split_data
import matplotlib.pyplot as plt

def expand_grid(param_grid):
    keys = param_grid.keys()
    values = param_grid.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def evaluate_model(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    report = classification_report(
        y_val,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    flat = flatten_clf_report(report)

    flat["accuracy"] = accuracy_score(y_val, y_pred)
    flat["macro_f1"] = f1_score(y_val, y_pred, average="macro")

    return flat


def preprocess_linear(X_train, X_val):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val)

def preprocess_tree(X_train, X_val):
    return X_train.values, X_val.values

def run_logistic_regression(X_train, X_val, y_train, y_val,
                            C=1.0, class_weight=None, max_iter=1000):

    X_tr, X_va = preprocess_linear(X_train, X_val)

    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter
    )

    results = evaluate_model(model, X_tr, X_va, y_train, y_val)
    results.update({
        "model": "LogisticRegression",
        "C": C,
        "class_weight": class_weight
    })
    return results

def run_knn_baseline(X_train, X_val, y_train, y_val, k=15):

    # kNN braucht Skalierung
    X_tr, X_va = preprocess_linear(X_train, X_val)

    model = KNeighborsClassifier(n_neighbors=k)

    results = evaluate_model(model, X_tr, X_va, y_train, y_val)
    results.update({
        "model": "kNN",
        "k": k
    })
    return results

def run_random_forest(X_train, X_val, y_train, y_val,
                      n_estimators=200, max_depth=None,
                      min_samples_leaf=1, class_weight=None):

    X_tr, X_va = preprocess_tree(X_train, X_val)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight=class_weight
    )

    results = evaluate_model(model, X_tr, X_va, y_train, y_val)
    results.update({
        "model": "RandomForest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight
    })
    return results

def run_gradient_boosting(X_train, X_val, y_train, y_val,
                          n_estimators=200, learning_rate=0.1, max_depth=3):

    X_tr, X_va = preprocess_tree(X_train, X_val)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
    )

    results = evaluate_model(model, X_tr, X_va, y_train, y_val)
    results.update({
        "model": "GradientBoosting",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        
    })
    return results

def run_xgboost_multiclass(X_train, X_val, y_train, y_val,
                           n_estimators=300,
                           learning_rate=0.1,
                           max_depth=3,
                           subsample=0.8,
                           colsample_bytree=0.8):

    X_tr, X_va = preprocess_tree(X_train, X_val)

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

    results = evaluate_model(model, X_tr, X_va, y_train, y_val)
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

def run_different_algorithms(X_train, X_val, y_train, y_val):
    all_results = []

    # ================= kNN Baseline =================
    all_results.append(
        run_knn_baseline(X_train, X_val, y_train, y_val, k=15)
    )
    print("Finished baseline kNN (k=15)")

    # ================= Logistic Regression =================
    lr_grid = {
        "C": [0.01, 0.1, 1.0],
        "class_weight": [None, "balanced"]
    }

    for params in expand_grid(lr_grid):
        all_results.append(
            run_logistic_regression(
                X_train, X_val, y_train, y_val,
                **params
            )
        )
        print(f"Finished Logistic Regression: {params}")

    # ================= Random Forest =================
    rf_grid = {
        "n_estimators": [100, 200, 400, 500],
        "max_depth": [None, 3, 5, 7],
        "class_weight": [None, "balanced"]
    }

    for params in expand_grid(rf_grid):
        all_results.append(
            run_random_forest(
                X_train, X_val, y_train, y_val,
                **params
            )
        )
        print(f"Finished Random Forest: {params}")

    # ================= Gradient Boosting =================
    gb_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 300, 500]
    }

    for params in expand_grid(gb_grid):
        all_results.append(
            run_gradient_boosting(
                X_train, X_val, y_train, y_val,
                **params
            )
        )
        print(f"Finished Gradient Boosting: {params}")

    # ================= XGBoost =================
    xgb_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 300, 500],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    for params in expand_grid(xgb_grid):
        all_results.append(
            run_xgboost_multiclass(
                X_train, X_val, y_train, y_val,
                **params
            )
        )
        print(f"Finished XGBoost: {params}")

    # ================= Export =================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        "exports/tables/task4_all_experiments.csv",
        sep=";",
        index=False
    )

    print("Saved results to exports/tables/task4_all_experiments.csv")
    return



def decrease_feature_space(X_train, X_val, max_remove_features=3):
    feature_names = list(X_train.columns)
    results = []

    for k in range(1, max_remove_features + 1):
        for features_to_remove in itertools.combinations(feature_names, k):
            
            X_train_reduced = X_train.drop(columns=list(features_to_remove))
            X_val_reduced  = X_val.drop(columns=list(features_to_remove)) 

            results.append({
                "removed_features": features_to_remove,
                "X_train": X_train_reduced,
                "X_val": X_val_reduced
            })

    return results

def test_feature_space(X_train, X_val, y_train, y_val):
    results = []
    table_combinations = decrease_feature_space(X_train, X_val)
    for i, table_comb in enumerate(table_combinations):
        x_train = table_comb['X_train']
        x_val = table_comb['X_val']
        removed_features = table_comb['removed_features']
        result = run_random_forest(x_train, x_val, y_train, y_val, n_estimators=400)
        result["removed_features"] = ",".join(removed_features)
        result["num_features"] = x_train.shape[1]

        results.append(result)
        print(f"Finished {i}/{len(table_combinations)}")
    results_df = pd.DataFrame(results)
    results_df.to_csv("exports/tables/task4_randomforest_feature_space.csv", sep=";", index=False)

    print("Saved results to exports/tables/task4_randomforest_feature_space.csv")

def reduce_matrix_dimensions(features_to_remove, X_data=[], return_as_pd_df=False):
    # Reduce feature space
    reduced_data = []
    feature_names = []
    for x_data in X_data:
        x_data_reduced = x_data.drop(columns=list(features_to_remove))
        feature_names = x_data_reduced.columns.tolist()
        if return_as_pd_df:
            reduced_data.append(x_data_reduced)
        else:
            reduced_data.append(x_data_reduced.values)
    
    return reduced_data, feature_names

def train_best_random_forest_reduced_features(
    X_train,
    X_val,
    y_train,
    y_val,
    features_to_remove=None,
    n_estimators=400,
    random_state=42
):
    """
    Trains the best Random Forest model on a reduced feature space.
    Returns the trained model and the remaining feature names.
    """
    if features_to_remove is not None:
        [X_tr, X_vl], feature_names = reduce_matrix_dimensions(features_to_remove=features_to_remove,
                                                            X_data=[X_train, X_val])
    else:
        X_tr, X_vl, feature_names = X_train, X_val, None

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state
    )

    results = evaluate_model(model, X_tr, X_vl, y_train, y_val)
    print(f"Macro-F1: {results['macro_f1']}")

    return results, model, feature_names



def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    feature_names=None,
    title="Confusion Matrix – Best Model",
    save_path="exports/cm.png"
):
    """
    Plots the confusion matrix for a trained model.
    Assumes the model is already trained.
    """
    # Apply same feature selection as during training
    if feature_names is not None:
        X_test_reduced = X_test[feature_names].values
    else:
        X_test_reduced = X_test

    y_pred = model.predict(X_test_reduced)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1, 2, 3]
    )

    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


if __name__ == '__main__':
    os.makedirs("exports/tables", exist_ok=True)
    df = pd.read_csv("D.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    """results = run_different_algorithms(X_train=X_train,
                             X_val=X_val,
                             y_train=y_train,
                             y_val=y_val)"""
    """test_feature_space(X_train=X_train,
                       X_val=X_val,
                       y_train=y_train,
                       y_val=y_val)"""
    results, best_model, feature_names = train_best_random_forest_reduced_features(
        X_train,
        X_val,
        y_train,
        y_val,
        features_to_remove=("feature_3", "feature_7", "feature_8")
    )
    """
    plot_confusion_matrix(
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    title="Confusion Matrix – Random Forest* (Reduced Feature Space)"
    )"""
    


    
    