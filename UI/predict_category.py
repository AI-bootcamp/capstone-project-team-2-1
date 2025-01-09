import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class PredictCategory:
    def __init__(self, model_path, label_mapping):
        """
        Initialize the PredictCategory class.

        Args:
            model_path (str): Path to the fine-tuned model.
            label_mapping (dict): Mapping of label IDs to category names.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.label_mapping = label_mapping

    def predict(self, text):
        """
        Predict the category for a single text input.

        Args:
            text (str): Input text to classify.

        Returns:
            str: Predicted category label.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Perform inference
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Get the predicted label ID
        predicted_id = torch.argmax(logits, dim=1).item()

        # Map the label ID to the category name
        return self.label_mapping[predicted_id]

    def predict_df(self, df, text_column="Message Content"):
        """
        Predict categories for a DataFrame containing message content.

        Args:
            df (pd.DataFrame): DataFrame containing the text data.
            text_column (str): Name of the column containing the text to classify.

        Returns:
            pd.DataFrame: DataFrame with an additional column 'Predicted Category'.
        """
        # Apply the predict method to each row in the DataFrame
        df["Predicted Category"] = df[text_column].apply(self.predict)
        return df