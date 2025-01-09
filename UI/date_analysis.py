import pandas as pd
from datetime import datetime, timedelta

class DateAnalysis:
    def __init__(self, df):
        """
        Initialize the DateAnalysis class.

        Args:
            df (pd.DataFrame): The DataFrame containing date-related data.
        """
        self.df = df
        # Ensure 'date' is in datetime format
        self.df['parsed_date'] = self.df['date'].apply(self.parse_date)
        # Replace "Tomorrow" with "غداً"
        self.df['date'] = self.df['date'].replace('Tomorrow', 'غداً')
        # Categorize dates into upcoming and past
        today = datetime.today()
        self.df['date_category'] = self.df['parsed_date'].apply(
            lambda x: 'Upcoming' if x and x > today else ('Past' if x and x <= today else 'Unknown')
        )

    def parse_date(self, date_str):
        """
        Convert date strings to datetime objects.

        Args:
            date_str (str): The date string to parse.

        Returns:
            datetime: The parsed datetime object, or None if invalid.
        """
        if date_str == 'Tomorrow' or date_str == 'غداً':
            return datetime.today() + timedelta(days=1)
        elif pd.isna(date_str):
            return None
        else:
            try:
                return datetime.strptime(date_str, '%d/%m/%Y')
            except ValueError:
                return None

    def get_upcoming_dates(self):
        """
        Get a DataFrame of upcoming dates.

        Returns:
            pd.DataFrame: A DataFrame containing upcoming dates with Sender and Message Content.
        """
        return self.df[self.df['date_category'] == 'Upcoming'][['Sender', 'Message Content', 'date_time', 'date', 'parsed_date']]

    def get_past_dates(self):
        """
        Get a DataFrame of past dates.

        Returns:
            pd.DataFrame: A DataFrame containing past dates with Sender and Message Content.
        """
        return self.df[self.df['date_category'] == 'Past'][['Sender', 'Message Content', 'date_time', 'date', 'parsed_date']]