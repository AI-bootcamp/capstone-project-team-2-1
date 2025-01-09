from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("../Models/fine-tuned-modernbert")
model = AutoModelForSequenceClassification.from_pretrained("../Models/fine-tuned-modernbert")

# Class labels (English and Arabic)
class_labels = {
    0: {'en': 'Apps', 'ar': 'تطبيقات'},
    1: {'en': 'Commercial', 'ar': 'تجاري'},
    2: {'en': 'Conferences and Events', 'ar': 'المؤتمرات والفعاليات'},
    3: {'en': 'Education', 'ar': 'تعليم'},
    4: {'en': 'Financial', 'ar': 'مالي'},
    5: {'en': 'Governmental', 'ar': 'حكومي'},
    6: {'en': 'Health', 'ar': 'صحة'},
    7: {'en': 'Personal', 'ar': 'شخصي'},
    8: {'en': 'Promotional', 'ar': 'ترويجي'},
    9: {'en': 'Services', 'ar': 'خدمات'},
    10: {'en': 'Stores', 'ar': 'متاجر'},
    11: {'en': 'Telecommunications', 'ar': 'اتصالات'},
    12: {'en': 'Travel', 'ar': 'سفر'}
}

# Emojis for each category
emojis = {
    'Apps': '📱', 'Commercial': '💼', 'Conferences and Events': '🎪',
    'Education': '🎓', 'Financial': '💰', 'Governmental': '🏛️',
    'Health': '🏥', 'Personal': '👤', 'Promotional': '📢',
    'Services': '🛠️', 'Stores': '🏪', 'Telecommunications': '📞',
    'Travel': '✈️'
}

# Function to predict category
def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probs, dim=1).item()
    return pred_class

# Load the dataset
df = pd.read_csv('../Data/test_data.csv')  # Ensure this file has 'Message Content' and 'Sender' columns

# Predict categories for all messages
df['Predicted Category'] = df['Message Content'].apply(lambda x: class_labels[predict_category(x)]['en'])

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_categories')
def get_categories():
    # Count messages per category
    category_counts = df['Predicted Category'].value_counts().to_dict()
    # Prepare data for frontend
    categories_data = []
    for category, count in category_counts.items():
        categories_data.append({
            'en': category,
            'ar': class_labels[[k for k, v in class_labels.items() if v['en'] == category][0]]['ar'],
            'count': count,
            'emoji': emojis[category]
        })
    return jsonify({'categories': categories_data})

@app.route('/get_messages/<category>')
def get_messages(category):
    # Filter messages by the selected category
    messages = df[df['Predicted Category'] == category][['Message Content', 'Sender']].to_dict('records')
    return jsonify({'messages': messages})

if __name__ == '__main__':
    app.run(debug=True)