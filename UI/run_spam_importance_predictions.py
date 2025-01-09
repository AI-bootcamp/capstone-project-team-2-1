import pandas as pd
from predict_spam_importance import PredictSpamImportance

# Load the test data
df = pd.read_csv('../Data/test2.csv')

# Initialize the PredictSpamImportance class
predictor = PredictSpamImportance(
    model_path='../Models/random_forest_multi_output_model.pkl',
    vectorizer_path='../Models/tfidf_vectorizer.pkl'
)

# Predict spam and importance for the DataFrame
df_with_predictions = predictor.predict_df(df)

# Display the DataFrame with predictions
print(df_with_predictions.head())