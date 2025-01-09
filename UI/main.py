import pandas as pd
from IPython.display import HTML, display
import datetime
from date_analysis import DateAnalysis
from financial_analysis import FinancialAnalysis
from predict_category import PredictCategory
from predict_spam_importance import PredictSpamImportance

# Load the data
df = pd.read_csv('../Data/test2.csv')

# Initialize the DateAnalysis class
date_analysis = DateAnalysis(df)

# Generate and display the summary
summary_text = date_analysis.get_summary()
HTML(summary_text)  # Use HTML to display the summary

# Display upcoming dates
upcoming_dates = date_analysis.get_upcoming_dates()
print("\nUpcoming Dates:")
display(upcoming_dates)

# Display past dates
past_dates = date_analysis.get_past_dates()
print("\nPast Dates:")
display(past_dates)

# Generate and print notifications
notifications = date_analysis.generate_notifications()
print("\nNotifications for Upcoming Dates:")
for notification in notifications:
    print(notification)

# Initialize the FinancialAnalysis class
financial_analysis = FinancialAnalysis(df)

# Define the bank, start date, and end date (you can customize these)
selected_bank = None  # Set to None for all banks, or specify a bank name
start_date = datetime.date(2024, 10, 27)  # Custom start date
end_date = datetime.date(2025, 1, 4)  # Custom end date

# Filter data
filtered_df = financial_analysis.filter_data(selected_bank, start_date, end_date)

# Calculate summary
income, expense, savings = financial_analysis.calculate_summary(filtered_df)

# Generate charts and analysis
bar_fig = financial_analysis.generate_bar_chart(income, expense, savings)
pie_fig = financial_analysis.generate_pie_chart(income, expense, savings)
financial_analysis_text = financial_analysis.generate_financial_analysis(income, expense, savings)

# Display results
bar_fig.show()
pie_fig.show()
print(financial_analysis_text)

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
predictor_category = PredictCategory(model_path="../Models/fine-tuned-modernbert_v4", label_mapping=label_mapping)

# Predict categories for the DataFrame
df_with_category_predictions = predictor_category.predict_df(df)

# Display the DataFrame with predictions
print("\nDataFrame with Category Predictions:")
print(df_with_category_predictions.head())

# Initialize the PredictSpamImportance class
predictor_spam_importance = PredictSpamImportance(
    model_path='../Models/random_forest_multi_output_model.pkl',
    vectorizer_path='../Models/tfidf_vectorizer.pkl'
)

# Predict spam and importance for the DataFrame
df_with_spam_importance_predictions = predictor_spam_importance.predict_df(df)

# Display the DataFrame with predictions
print("\nDataFrame with Spam and Importance Predictions:")
print(df_with_spam_importance_predictions.head())