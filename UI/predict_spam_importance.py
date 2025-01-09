import pandas as pd
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

class PredictSpamImportance:
    def __init__(self, model_path, vectorizer_path):
        """
        Initialize the PredictSpamImportance class.

        Args:
            model_path (str): Path to the saved RandomForest model.
            vectorizer_path (str): Path to the saved TF-IDF vectorizer.
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text):
        """
        Predict whether a message is spam and important.

        Args:
            text (str): Input text to classify.

        Returns:
            dict: Dictionary containing 'is_spam' and 'is_important' predictions.
        """
        # Transform the text using the TF-IDF vectorizer
        text_vectorized = self.vectorizer.transform([text])

        # Predict using the model
        prediction = self.model.predict(text_vectorized)

        # Return the results as a dictionary
        return {
            'is_spam': bool(prediction[0][0]),
            'is_important': bool(prediction[0][1])
        }

    def predict_df(self, df, text_column="Message Content"):
        """
        Predict spam and importance for a DataFrame containing message content.

        Args:
            df (pd.DataFrame): DataFrame containing the text data.
            text_column (str): Name of the column containing the text to classify.

        Returns:
            pd.DataFrame: DataFrame with additional columns 'is_spam' and 'is_important'.
        """
        # Transform the text column using the TF-IDF vectorizer
        text_vectorized = self.vectorizer.transform(df[text_column])

        # Predict using the model
        predictions = self.model.predict(text_vectorized)

        # Add predictions to the DataFrame
        df['is_spam'] = predictions[:, 0]
        df['is_important'] = predictions[:, 1]

        return df