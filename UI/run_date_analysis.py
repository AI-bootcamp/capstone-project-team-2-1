import pandas as pd
from IPython.display import HTML , display
from date_analysis import DateAnalysis

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