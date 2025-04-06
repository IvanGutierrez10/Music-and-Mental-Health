import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(data):
    """
    Clean the data by removing handle missing values, normalize numerical features,
    and encode categorial features.
    
    Parameters:
    data (DataFrame): Data to be cleaned.
    
    Returns:
    DataFrame: Cleaned data.
    """
    if data is None:
        print("No data to clean.")
        return None
    
    df_clean = data.copy()

    # Drop unnecessary columns
    df_clean.drop(columns=["Timestamp", "Permissions"], inplace=True)

    # Handle missing values
    categorical_cols = ["Primary streaming service", "While working", "Instrumentalist",
    "Composer", "Foreign languages", "Music effects"]

    for col in categorical_cols:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    df_clean["BPM"].fillna(df_clean["BPM"].mean(), inplace=True)
    df_clean["Age"].fillna(df_clean["Age"].median(), inplace=True)

    # Encode frequency and binary features
    frequency_map = {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Very frequently": 3
    }

    frequency_cols = [col for col in df_clean.columns if col.startswith("Frequency")]
    for col in frequency_cols:
        df_clean[col] = df_clean[col].map(frequency_map)

    # Normalize numerical features
    numerical_cols = ["Age", "Hours per day", "BPM", "Anxiety",
    "Depression", "Insomnia", "OCD"]+frequency_cols

    scaler = MinMaxScaler()
    df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

    # Encode categorical features
    binary_map = {"Yes": 1, "No": 0}
    binary_cols = ["Instrumentalist", "Composer", "Exploratory", "Foreign languages"]

    for col in binary_cols:
        df_clean[col] = df_clean[col].map(binary_map)

    encoding_cols = ["Primary streaming service", "While working", "Music effects", "Fav genre", "Primary streaming service"]
    df_clean = pd.get_dummies(df_clean, columns=encoding_cols, drop_first=True)

    # Convert all boolean columns to integers (True -> 1, False -> 0)
    bool_cols = df_clean.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df_clean[col] = df_clean[col].astype(int)

    return df_clean

def main():

    warnings.filterwarnings("ignore", category=FutureWarning)

    # Load data
    file_path = "data/raw_dataset/mxmh_survey_results.csv"
    data = load_data(file_path)

    # Clean data
    cleaned_data = clean_data(data)

    if cleaned_data is not None:
        print("Data cleaning completed successfully.")
        cleaned_data.to_csv("data/clean_dataset/cleaned_data.csv", index=False)
        print("Cleaned data saved to cleaned_data.csv")
    else:
        print("Data cleaning failed.")

if __name__ == "__main__":
    main()

