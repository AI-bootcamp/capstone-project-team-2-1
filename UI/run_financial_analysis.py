import pandas as pd
import datetime
from financial_analysis import FinancialAnalysis

# Load the data
df5 = pd.read_csv('../Data/test2.csv')

# Initialize the FinancialAnalysis class
financial_analysis = FinancialAnalysis(df5)

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