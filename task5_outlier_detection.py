import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from task4_model_experimentation import split_data, train_best_random_forest_reduced_features, reduce_matrix_dimensions


# =========================================================
# Random Forest Training + Evaluation
# =========================================================
def train_model(X_train, y_train, n_estimators=400):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model

def train_and_eval_on_val(X_train, X_val, y_train, y_val,
                      n_estimators=400, predict=True):

    model = train_model(X_train, y_train, n_estimators)
    if predict:
        y_pred = model.predict(X_val)

        return {
            "macro_f1": f1_score(y_val, y_pred, average="macro"),
            "accuracy": accuracy_score(y_val, y_pred),
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
    X_out = df_out.drop(columns=["id"])
    [X_out_reduced], _ = reduce_matrix_dimensions(features_to_remove=('feature_3', 'feature_7', 'feature_8'),
                                              X_data=[X_out], return_as_pd_df=True)
    X_train = df_train.drop(columns=['label'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_out_scaled = scaler.transform(X_out_reduced)

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

def train_model_gmm_tau(df_train:pd.DataFrame, X_train, X_val, y_train, y_val):
    boolean_mask, gmm, tau, scaler = get_outlier_mask(df_train.copy())
    X_train_in = X_train.loc[~boolean_mask]
    y_train_in = y_train.loc[~boolean_mask]
    
    results_in = train_and_eval_on_val(
        X_train_in, X_val, y_train_in, y_val
    )
    return results_in, gmm, tau

def task5_main():

    df = pd.read_csv("D.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    ### reduce X in feature dimensions
    [X_tr_red, X_val_red, X_test_red], feature_names = reduce_matrix_dimensions(features_to_remove=('feature_3',
                                                                                       'feature_7',
                                                                                       'feature_8'),
                                                                                       X_data=[X_train, X_val, X_test])
    X_tr_red_df = pd.DataFrame(
        X_tr_red,
        columns=feature_names,
        index=X_train.index
    )
    X_val_red_df = pd.DataFrame(
        X_val_red,
        columns=feature_names,
        index=X_val.index
    )
    df_train = pd.concat([X_tr_red_df, y_train], axis=1)

    results_D, _, _ = train_best_random_forest_reduced_features(X_train=X_tr_red,
                                              X_val=X_val_red,
                                              y_train=y_train,
                                              y_val=y_val,
                                              features_to_remove=None)
    
    results_D_in, _, _ = train_model_gmm_tau(df_train, X_tr_red_df, X_val_red_df, y_train, y_val)

    model = results_D_in['model']

    y_pred = model.predict(X_test_red)
    macro_f1_d_in_model = f1_score(y_test, y_pred, average="macro")

    print("\n=== FINAL RESULTS ===")
    print(f"Macro F1 (D) original model:   {results_D['macro_f1']:.4f}")
    print(f"Macro F1 (Din) outlier trained model: {results_D_in['macro_f1']:.4f}")
    print(f"Macro F1 (D_test | model trained on Din): {macro_f1_d_in_model:.4f}")


if __name__ == "__main__":
    task5_main()
