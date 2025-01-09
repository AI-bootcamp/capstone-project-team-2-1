from flask import Flask, render_template, request
import pandas as pd
import datetime
from predict_category import PredictCategory
from predict_spam_importance import PredictSpamImportance
from financial_analysis import FinancialAnalysis
from date_analysis import DateAnalysis

app = Flask(__name__)

# Load data
df = pd.read_csv('../Data/test2.csv')

# Initialize predictors
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
category_predictor = PredictCategory(model_path="../Models/fine-tuned-modernbert_v4", label_mapping=label_mapping)
spam_predictor = PredictSpamImportance(
    model_path='../Models/random_forest_multi_output_model.pkl',
    vectorizer_path='../Models/tfidf_vectorizer.pkl'
)
financial_analysis = FinancialAnalysis(df)
date_analysis = DateAnalysis(df)

# Add predictions to the DataFrame
df = category_predictor.predict_df(df)
df = spam_predictor.predict_df(df)

# Routes
@app.route('/')
def home():
    return render_template('messages.html', messages=df.to_dict('records'))

@app.route('/messages', methods=['GET'])
def messages():
    # Handle filter for important messages
    filter_important = request.args.get('filter_important', 'all')

    # Filter messages based on importance
    if filter_important == 'important':
        filtered_messages = [msg for msg in df.to_dict('records') if msg['is_important']]
    else:
        filtered_messages = df.to_dict('records')

    return render_template(
        'messages.html',
        messages=filtered_messages,
        filter_important=filter_important
    )

@app.route('/spam')
def spam():
    spam_messages = df[df['is_spam'] == True].to_dict('records')
    return render_template('spam.html', spam_messages=spam_messages)

@app.route('/transactions', methods=['GET', 'POST'])
def transactions():
    # Default start and end dates
    end_date = datetime.date.today()
    
    # Handle user input for time period
    time_period = request.args.get('time_period', 'last_month')  # Default to last month
    if time_period == 'last_day':
        start_date = end_date - datetime.timedelta(days=1)
    elif time_period == 'last_week':
        start_date = end_date - datetime.timedelta(weeks=1)
    else:  # last_month
        start_date = end_date - datetime.timedelta(days=30)  # Approximate last month

    # Handle user input for unique sender filter
    unique_sender = request.args.get('unique_sender', None)

    # Filter data for transactions
    filtered_df = financial_analysis.filter_data(None, start_date, end_date)
    
    # Apply unique sender filter if selected
    if unique_sender and "Predicted Category" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Predicted Category"] == "المالية"]
        filtered_df = filtered_df[filtered_df["Sender"] == unique_sender]

    # Calculate summary and generate charts
    income, expense, savings = financial_analysis.calculate_summary(filtered_df)
    bar_fig = financial_analysis.generate_bar_chart(income, expense, savings)
    pie_fig = financial_analysis.generate_pie_chart(income, expense, savings)
    financial_analysis_text = financial_analysis.generate_financial_analysis(income, expense, savings)

    # Get unique senders for the dropdown
    unique_senders = []
    if "Sender" in filtered_df.columns:
        unique_senders = filtered_df[filtered_df["Predicted Category"] == "المالية"]["Sender"].unique().tolist()

    return render_template(
        'transactions.html',
        bar_fig=bar_fig.to_html(),
        pie_fig=pie_fig.to_html(),
        analysis_text=financial_analysis_text,
        time_period=time_period,
        unique_senders=unique_senders,
        selected_sender=unique_sender
    )

@app.route('/dates')
def dates():
    # Get upcoming and past dates
    upcoming_dates = date_analysis.get_upcoming_dates()
    past_dates = date_analysis.get_past_dates()

    # Replace NaN in 'date_time' with None
    upcoming_dates['date_time'] = upcoming_dates['date_time'].where(pd.notna(upcoming_dates['date_time']), None)
    past_dates['date_time'] = past_dates['date_time'].where(pd.notna(past_dates['date_time']), None)

    # Convert to dictionary for the template
    upcoming_dates = upcoming_dates.to_dict('records')
    past_dates = past_dates.to_dict('records')

    return render_template('dates.html', upcoming_dates=upcoming_dates, past_dates=past_dates)

if __name__ == '__main__':
    app.run(debug=True)