import pandas as pd
from predict_category import PredictCategory

# Load the test data
df = pd.read_csv('../Data/test2.csv')

# Define the label mapping with Arabic category names
label_mapping = {
    0: "التعليم",
    1: "المالية",
    2: "الحكومة",
    3: "الصحة",
    4: "أخرى",
    5: "العروض",
    6: "الخدمات أو المتاجر",
    7: "الاتصالات",
    8: "السفر"
}

# Initialize the PredictCategory class
predictor = PredictCategory(model_path="../Models/fine-tuned-modernbert_v4", label_mapping=label_mapping)

# Predict categories for the DataFrame
df_with_predictions = predictor.predict_df(df)

# Display the DataFrame with predictions
print(df_with_predictions.head())
print(df_with_predictions["Predicted Category"].value_counts())