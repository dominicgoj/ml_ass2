from baseline_model import preprocess_with_clusters, split_data
from data_analysis import read_csv_file
from sklearn.metrics import classification_report, accuracy_score, f1_score
from baseline_model import flatten_clf_report
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

EXPORT_FOLDER = "exports/tables/"
SEED = 42

def evaluate_model(model, name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    train_report = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)
    test_report  = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

    train_macro_f1 = train_report["macro avg"]["f1-score"]
    test_macro_f1  = test_report["macro avg"]["f1-score"]
    return name, train_macro_f1, test_macro_f1, train_report, test_report, y_train_pred, y_test_pred, model

def model_exp(normalization=True, csv_out="task5_model_exp.csv"):
    os.makedirs(EXPORT_FOLDER, exist_ok=True)
    df = read_csv_file("D.csv")
    if normalization:
        X_train, X_test, y_train, y_test = preprocess_with_clusters(df)
    else:
        X_train, X_test, y_train, y_test = split_data(df)
    
    experiments = [
        ("KNN_k7_baseline", KNeighborsClassifier(n_neighbors=7)),
        ("LR_C1", LogisticRegression(C=1, random_state=SEED)),
        ("LR_C01", LogisticRegression(C=0.1, random_state=SEED)),
        ("SVM_rbf", SVC(kernel="rbf", C=1, random_state=SEED)),
        ("SVM_linear", SVC(kernel="linear", C=1, random_state=SEED)),
        ("DT_depth5", DecisionTreeClassifier(max_depth=5, random_state=SEED)),
        ("DT_depth20", DecisionTreeClassifier(max_depth=20, random_state=SEED)),
        ("DT_depth40", DecisionTreeClassifier(max_depth=40, random_state=SEED)),
        ("RF_100", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ("RF_300", RandomForestClassifier(n_estimators=300, random_state=SEED)),
        ("RF_500", RandomForestClassifier(n_estimators=500, random_state=SEED)),
        ("RF_750", RandomForestClassifier(n_estimators=750, random_state=SEED)),
        ("RF_1000", RandomForestClassifier(n_estimators=1000, random_state=SEED)),
        ("GB_lr_0.1", GradientBoostingClassifier(learning_rate=0.1, random_state=SEED)),
        ("GB_lr_0.05", GradientBoostingClassifier(learning_rate=0.05, random_state=SEED)),
        ("GB_lr_0.1_depth_3", GradientBoostingClassifier(learning_rate=0.1, max_depth=3, random_state=SEED)),
        ("GB_lr_0.05_depth_3", GradientBoostingClassifier(learning_rate=0.05, max_depth=3, random_state=SEED)),
        ("GB_depth_3", GradientBoostingClassifier(max_depth=3, random_state=SEED)),
        ("GB_depth_6", GradientBoostingClassifier(max_depth=6, random_state=SEED)),
    ]

    table_results = []

    for exp in experiments:
        name, train_macro_f1, test_macro_f1, train_report, test_report, y_train_pred, y_test_pred, _ = evaluate_model(exp[1], exp[0], X_train, X_test, y_train, y_test)
        #flat = flatten_clf_report(report_dict=report)
        results_dict = {}
        results_dict['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        results_dict['test_accuracy'] = accuracy_score(y_test, y_test_pred)
        results_dict['train_macro_f1'] = train_macro_f1
        results_dict['test_macro_f1'] = test_macro_f1
        results_dict['name'] = name
        table_results.append(results_dict)

    table = pd.DataFrame(table_results)
    table.to_csv(os.path.join(EXPORT_FOLDER, csv_out), sep=";")

def get_least_important_features_to_remove(df, ft_importances:list):
    features = [col for col in df.columns if col not in ('id', 'label')]
    table = pd.DataFrame(columns=['feature', 'importance'])
    table['feature'] = features
    table['importance'] = ft_importances
    table = table.sort_values(by='importance', ascending=True)
    
    remove_features_combinations = []
    remove_max_features = 5
    features_to_remove = table["feature"].tolist()[:remove_max_features]
    current = []
    for f in features_to_remove:
        current = current + [f]
        remove_features_combinations.append(current.copy())
    return remove_features_combinations

def generate_dfs_with_features_removed(df:pd.DataFrame, features_to_drop:list[list]):
    df_list = []
    df_copy = df.copy()
    for ft in features_to_drop:
        edited_df = df_copy.copy()
        edited_df = edited_df.drop(columns=ft)
        df_list.append(edited_df)
    return df_list

def random_forest_exp():
    df = read_csv_file("D.csv")
    X_train, X_test, y_train, y_test = preprocess_with_clusters(df)
    model = RandomForestClassifier(n_estimators=500, random_state=SEED, class_weight="balanced")
    name, train_macro_f1, test_macro_f1, train_report, test_report, y_train_pred, y_test_pred, model = evaluate_model(model, 'RF_500', X_train, X_test, y_train, y_test)
    ft_importances = model.feature_importances_
    remove_features_combinations = get_least_important_features_to_remove(df=df, ft_importances=ft_importances)
    df_alterations = generate_dfs_with_features_removed(df, remove_features_combinations)
    table_results = []
    df_alterations.append(df)
    for i, df_alteration in enumerate(df_alterations):
        model = RandomForestClassifier(n_estimators=500, random_state=SEED, class_weight="balanced")
        X_train, X_test, y_train, y_test = preprocess_with_clusters(df_alteration)
        name, train_macro_f1, test_macro_f1, train_report, test_report, y_train_pred, y_test_pred, model = \
            evaluate_model(model=model, name=f'Comb {i}', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        results_dict = {}
        results_dict['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        results_dict['test_accuracy'] = accuracy_score(y_test, y_test_pred)
        results_dict['train_macro_f1'] = train_macro_f1
        results_dict['test_macro_f1'] = test_macro_f1
        results_dict['name'] = name
        results_dict['ft_removal_comb'] = str(i)
        table_results.append(results_dict)
    
    table = pd.DataFrame(table_results)
    table.to_csv(os.path.join(EXPORT_FOLDER, 'rft_500_feature_comb_removal.csv'), sep=";")
    
if __name__ == '__main__':
    #model_exp(normalization=True, csv_out="task5_model_norm_data.csv")
    #model_exp(normalization=False, csv_out="task5_model_unnorm_data.csv")

    random_forest_exp()