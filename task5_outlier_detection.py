import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from task4_model_experimentation import split_data


# =========================================================
# Random Forest Training + Evaluation
# =========================================================
def run_random_forest(X_train, X_test, y_train, y_test,
                      n_estimators=400, predict=True):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    if predict:
        y_pred = model.predict(X_test)

        return {
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "accuracy": accuracy_score(y_test, y_pred),
            "model": model
        }
    else:
        return model


# =========================================================
# Outlier Detection NUR auf Trainingsdaten
# =========================================================
def get_outlier_mask(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Takes ONLY the training split and returns Din_train
    """

    df_out = pd.read_csv("D_out.csv")

    X_train = df_train.drop(columns=['label'])
    X_out = df_out.drop(columns=["id"])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_out_scaled = scaler.transform(X_out)

    gmm = GaussianMixture(
        n_components=8,
        covariance_type="full",
        random_state=42
    )
    gmm.fit(X_train_scaled)

    log_p_train = gmm.score_samples(X_train_scaled)
    log_p_out = gmm.score_samples(X_out_scaled)

    tau = np.percentile(log_p_out, 80)

    is_outlier_train = log_p_train <= tau
    print("Outlier-Anteil im TRAIN:", is_outlier_train.mean())

    return is_outlier_train, gmm, tau, scaler

def train_model_gmm_tau(df_train:pd.DataFrame, X_train, X_test, y_train, y_test):
    boolean_mask, gmm, tau, scaler = get_outlier_mask(df_train.copy())
    X_train_in = X_train.loc[~boolean_mask]
    y_train_in = y_train.loc[~boolean_mask]
    results_in = run_random_forest(
        X_train_in, X_test, y_train_in, y_test
    )
    return results_in, gmm, tau

def task5_main():

    df = pd.read_csv("D.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    df_train = pd.concat([X_train, y_train], axis=1)

    results_D = run_random_forest(
        X_train, X_test, y_train, y_test
    )
    
    results_in, _, _, _ = train_model_gmm_tau(df_train, X_train, X_test, y_train, y_test)

    model = results_in['model']

    y_pred = model.predict(X_test)
    macro_f1_d_in_model = f1_score(y_test, y_pred, average="macro")

    print("\n=== FINAL RESULTS ===")
    print(f"Macro F1 (D) original model:   {results_D['macro_f1']:.4f}")
    print(f"Macro F1 (Din) outlier trained model: {results_in['macro_f1']:.4f}")
    print(f"Macro F1 (D) outlier trained model: {macro_f1_d_in_model:.4f}")


if __name__ == "__main__":
    task5_main()
