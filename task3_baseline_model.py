from sklearn.model_selection import train_test_split
import pandas as pd
from task2_data_analysis import read_csv_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

def split_data(df, id_col='id', label_col='label'):
    feature_cols = [c for c in df.columns if c not in [id_col, label_col]]

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[label_col],
        test_size=0.2, random_state=42, stratify=df[label_col]
    )

    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    stratify=y_train,
    random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_standard_scaling(df)->(pd.DataFrame):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

def flatten_clf_report(report_dict):
    """
    Flatten the nested classification_report dict to a single-level dict:
    e.g. {'0': {'precision':..}} -> {'0_precision':..}
    keeps 'accuracy' as is (it's not a dict)
    """
    flat = {}
    for key, val in report_dict.items():
        if isinstance(val, dict):
            for metric, metric_val in val.items():
                # replace spaces in keys e.g. "macro avg" -> "macro_avg"
                k = f"{key}_{metric}".replace(" ", "_")
                flat[k] = metric_val
        else:
            # accuracy is returned as a scalar, keep as-is (key = 'accuracy')
            flat[key] = val
    return flat



def evaluate_knn(X_train, X_val, X_test, y_train, y_val, y_test):
    os.makedirs("exports/tables", exist_ok=True)
    best_k = None
    best_macro_f1 = -1

    results = {}
    table_results = []

    for k in [1, 3, 5, 7, 9, 11, 15, 17, 19]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        macro = f1_score(y_val, y_pred, average="macro")

        results[k] = macro
        if macro > best_macro_f1:
            best_macro_f1 = macro
            best_k = k
        print("=" * 50)
        print(f"K: {k}")
        print("Classification Report:")
        print(classification_report(y_val, y_pred, digits=3))
        print("Accuracy:", accuracy_score(y_val, y_pred))
        print("Macro F1:", f1_score(y_val, y_pred, average="macro"))

        out_dict = classification_report(y_val, y_pred, digits=3, output_dict=True)
        flat = flatten_clf_report(out_dict)
        flat['k'] = k
        flat['accuracy'] = accuracy_score(y_val, y_pred)
        flat['macro_f1'] = macro
        table_results.append(flat)

    # Train final model with best k
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    table = pd.DataFrame(table_results)
    table.to_csv("exports/tables/knn_evaluation.csv", sep=";", index=False)
    print("Best K:", best_k)
    print("=" * 50)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

    return best_k, y_pred, table_results

def task3():
    print(30*"*", "TASK3", 30*"*")
    df = read_csv_file("D.csv")
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = \
    preprocess_standard_scaling(df)

    evaluate_knn(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test
    )

if __name__ == '__main__':
    task3()
    