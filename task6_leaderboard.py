from task5_outlier_detection import train_model_gmm_tau, get_outlier_mask, train_model, reduce_matrix_dimensions, train_best_random_forest_reduced_features
from task4_model_experimentation import split_data
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pandas as pd

def task6():
    print(30*"*", "TASK6", 30*"*")
    df = pd.read_csv("D.csv")
    X_train, y_train = df.drop(columns=['label', 'id']), df['label']
    [X_tr_red], feature_names = reduce_matrix_dimensions(features_to_remove=('feature_3',
                                                                                       'feature_7',
                                                                                       'feature_8'),
                                                                                       X_data=[X_train],
                                                                                       return_as_pd_df=True)
    

    df_train = pd.concat([X_tr_red, y_train], axis=1)


    boolean_mask, gmm, tau, scaler = get_outlier_mask(df_train.copy())
    X_train_in = X_tr_red.loc[~boolean_mask]
    y_train_in = y_train.loc[~boolean_mask]
    model = train_model(
        X_train_in, y_train_in
    )
    return model, gmm, tau, scaler, feature_names

if __name__ == '__main__':
    task6()