import pandas as pd
import numpy as np
import joblib
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RANDOM_STATE = 42
DATA_PATH = "data/student_burnout_dropout_dataset_2_link 1.csv"

FEATURES = [
    "Age", "Year_of_Study", "Attendance_Percent", "Study_Hours_Per_Day",
    "Previous_GPA", "Backlogs", "Sleep_Hours", "Screen_Time_Hours",
    "Exercise_Freq_Per_Week", "Social_Activity_Score", "Financial_Stress_Score",
    "Family_Support_Score", "Stress_Level", "Anxiety_Score",
    "Motivation_Score", "Peer_Pressure_Score"
]

def run_foundation():
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X_raw = df[FEATURES].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=None, random_state=RANDOM_STATE)
    pca_full = pca.fit_transform(X_scaled)
    pca_2d = pca_full[:, :2]

    joblib.dump(imputer, "models/imputer.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(pca, "models/pca.joblib")

    pd.DataFrame(X_scaled, columns=FEATURES).to_csv("data/X_scaled.csv", index=False)
    pd.DataFrame(pca_2d, columns=["PC1", "PC2"]).to_csv("data/pca_2d.csv", index=False)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_variance)
    components = [f"PC{i+1}" for i in range(len(explained_variance))]
    
    df_var = pd.DataFrame({
        "component": components,
        "explained_variance": explained_variance,
        "cumulative": cumulative
    })
    df_var.to_csv("data/pca_varianza.csv", index=False)
    
    with open("data/feature_names.json", "w") as f:
        json.dump(FEATURES, f)

if __name__ == "__main__":
    run_foundation()
