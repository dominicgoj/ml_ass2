from task5_outlier_detection import train_model_gmm_tau, get_outlier_mask, run_random_forest
from task4_model_experimentation import split_data
import pandas as pd

def train_model():
    df = pd.read_csv("D.csv")
    X_train, y_train = df.drop(columns=['label', 'id']), df['label']
    df_train = pd.concat([X_train, y_train], axis=1)
    boolean_mask, gmm, tau, scaler = get_outlier_mask(df_train.copy())
    X_train_in = X_train.loc[~boolean_mask]
    y_train_in = y_train.loc[~boolean_mask]
    model = run_random_forest(
        X_train=X_train_in, X_test=None, y_train=y_train_in, y_test=None,
        predict=False
    )
    return model, gmm, tau, scaler

if __name__ == '__main__':
    train_model()