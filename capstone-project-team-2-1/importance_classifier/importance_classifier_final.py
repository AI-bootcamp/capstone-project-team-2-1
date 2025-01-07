# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.normalize import normalize_unicode
import pandas as pd
import numpy as np
import joblib
import re
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

class MessageImportanceClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.important_patterns = [
            r"\bراتب\b", r"\bتم تحويل\b", r"\bخصم\b", r"\b(دولار|ريال|درهم)\b",
            r"\bجدول\b", r"\bschedule\b", r"\bموعد\b", r"\bdate\b", r"\bmeeting\b",
            r"\btest result\b", r"\bappointment\b", r"\bflight\b", r"\bticket\b",
            r"\bفاتورة\b", r"\bbill\b", r"\bpayment\b", r"\burgent\b", r"\bعاجل\b",
            r"\bdeadline\b", r"\bموعد نهائي\b", r"\bemergency\b", r"\bطوارئ\b"
        ]
        
        self.less_important_patterns = [
            r"\bرصيد\b", r"\bbalance\b", r"\bتنويه\b", r"\bnote\b", r"\btips\b",
            r"\bإشعار\b", r"\bnotice\b", r"\bإعلان\b", r"\bannouncement\b",
            r"\bدعوة\b", r"\binvite\b", r"\bاشعار\b", r"\bnotification\b",
            r"\bجدول\b", r"\bschedule\b", r"\bمعلومة\b", r"\binfo\b",
            r"\bتذكير\b", r"\breminder\b", r"\bتنويه\b", r"\bnote\b",
            r"\bتذكير طبي\b", r"\bhealth tip\b", r"\bرسالة عامة\b",
            r"\bgeneral message\b", r"\bتنبيه\b", r"\bspam\b",
            r"\bunidentified\b", r"\bunknown\b"
        ]

    def preprocess_message(self, message):
        if not isinstance(message, str):
            return ""
            
        # Normalize Arabic text
        message = normalize_unicode(message)
        
        # Tokenize using camel_tools
        tokens = simple_word_tokenize(message)
        
        # Remove special characters and numbers
        tokens = [re.sub(r'[^\w\s\u0600-\u06FF]', '', token) for token in tokens]
        
        # Remove empty tokens and normalize whitespace
        tokens = [token.strip() for token in tokens if token.strip()]
        
        # Convert English text to lowercase while preserving Arabic
        tokens = [token.lower() if not any(c in '\u0600-\u06FF' for c in token) else token 
                 for token in tokens]
        
        return ' '.join(tokens)

    def label_importance(self, message):
        if not isinstance(message, str):
            return "Less Important"
            
        message = self.preprocess_message(message)
        
        # Check for important patterns
        for pattern in self.important_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "Important"
        
        # Check for less important patterns
        for pattern in self.less_important_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "Less Important"
        
        return "Less Important"

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()

    def train(self, data_path, test_size=0.2, random_state=42):
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data = pd.read_csv(data_path)
        data['Message Content'] = data['Message Content'].fillna('').astype(str)
        data['Importance'] = data['Message Content'].apply(self.label_importance)
        data['Processed_Message'] = data['Message Content'].apply(self.preprocess_message)

        # Feature extraction
        print("Extracting features...")
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(data['Processed_Message'])
        y = data['Importance']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Apply SMOTE for balance
        print("Balancing dataset with SMOTE...")
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Define base classifiers with hyperparameter grids
        classifiers = {
            'nb': (MultinomialNB(), {
                'alpha': [0.1, 0.5, 1.0]
            }),
            'rf': (RandomForestClassifier(random_state=random_state), {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None]
            }),
            'knn': (KNeighborsClassifier(), {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }),
            'svc': (SVC(probability=True, random_state=random_state), {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear']
            })
        }

        # Train and optimize each classifier
        print("Training and optimizing individual classifiers...")
        best_classifiers = []
        for name, (clf, param_grid) in classifiers.items():
            print(f"Optimizing {name}...")
            grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_train_balanced, y_train_balanced)
            best_classifiers.append((name, grid_search.best_estimator_))
            print(f"Best parameters for {name}: {grid_search.best_params_}")

        # Create and train voting classifier
        print("Training ensemble classifier...")
        self.model = VotingClassifier(estimators=best_classifiers, voting='soft')
        self.model.fit(X_train_balanced, y_train_balanced)

        # Evaluate the model
        print("\nEvaluating model performance...")
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print("\nCross-validation scores:", cv_scores)
        print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

        return self

    def save_model(self, model_path='importance_classifier.pkl', vectorizer_path='vectorizer.pkl'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")

    def load_model(self, model_path='importance_classifier.pkl', vectorizer_path='vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        return self

    def predict(self, messages):
        if isinstance(messages, str):
            messages = [messages]
        
        processed_messages = [self.preprocess_message(msg) for msg in messages]
        X = self.vectorizer.transform(processed_messages)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for msg, pred, prob in zip(messages, predictions, probabilities):
            results.append({
                'message': msg,
                'prediction': pred,
                'confidence': max(prob)
            })
        
        return results

def main():
    # Initialize classifier
    classifier = MessageImportanceClassifier()
    
    # Train the model
    classifier.train('importance_classifier\Data_with_Improved_Importance.csv')
    
    # Save the model
    classifier.save_model()
    
    # Example predictions
    test_messages = [
        "Your salary has been transferred",
        "Don't forget your appointment tomorrow",
        "Check out our latest offers!",
        "تم تحويل الراتب الشهري",
        "عروض حصرية لفترة محدودة"
    ]
    
    results = classifier.predict(test_messages)
    print("\nExample Predictions:")
    for result in results:
        print(f"\nMessage: {result['message']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()