import pandas as pd
import numpy as np
from task6_leaderboard import task6
from task5_outlier_detection import task5
from task4_model_experimentation import task4
from task3_baseline_model import task3
from task2_data_analysis import task2

def predict(X_test):
    # TODO replace this with your model's predictions
    # For now, we will just return random predictions
    model, gmm, tau, scaler, used_feature_names = task6()
    X_test = X_test[used_feature_names]
    X_leaderboard_scaled = scaler.transform(X_test)
    log_p = gmm.score_samples(X_leaderboard_scaled)
    outliers = (log_p <= tau).astype(int)
    labels = model.predict(X_test)
    return labels, outliers


def generate_submission(test_data):
    label_predictions, outlier_predictions = predict(test_data)
    # IMPORTANT: stick to this format for the submission, 
    # otherwise your submission will result in an error
    submission_df = pd.DataFrame({ 
        "id": test_data["id"],
        "label": label_predictions,
        "outlier": outlier_predictions
    })
    return submission_df


def main():
    task2()
    task3()
    task4()
    task5()
    df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
    submission_df = generate_submission(df_leaderboard)
    # IMPORTANT: The submission file must be named "submission_leaderboard_GroupName.csv",
    # replace GroupName with a group name of your choice. If you do not provide a group name, 
    # your submission will fail!
    submission_df.to_csv("submission_leaderboard_GOKU.csv", index=False)
    
    # For the final leaderboard, change the file name to "submission_final_GroupName.csv"
    """df_final = pd.read_csv("D_test_final.csv")
    submission_df = generate_submission(df_final)
    submission_df.to_csv("submission_final_GroupName.csv", index=False)"""

if __name__ == "__main__":
    main()