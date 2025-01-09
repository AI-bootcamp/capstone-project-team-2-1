import pandas as pd
import plotly.graph_objects as go
import datetime

class FinancialAnalysis:
    def __init__(self, df):
        """
        Initialize the FinancialAnalysis class.

        Args:
            df (pd.DataFrame): The DataFrame containing financial data.
        """
        self.df = df
        # Ensure 'transaction_date' is in datetime format
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'], format='%d-%m-%Y', errors='coerce')

    def filter_data(self, selected_bank, start_date, end_date):
        """
        Filter the DataFrame based on the selected bank and date range.

        Args:
            selected_bank (str): The selected bank (or None for all banks).
            start_date (datetime.date): The start date of the range.
            end_date (datetime.date): The end date of the range.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        # Filter by selected bank
        if selected_bank:
            filtered_df = self.df[self.df['Sender'] == selected_bank]
        else:
            filtered_df = self.df

        # Filter by date range
        filtered_df = filtered_df[
            (filtered_df['transaction_date'].dt.date >= start_date) &
            (filtered_df['transaction_date'].dt.date <= end_date)
        ]

        return filtered_df

    def calculate_summary(self, filtered_df):
        """
        Calculate income, expenses, and savings from the filtered DataFrame.

        Args:
            filtered_df (pd.DataFrame): The filtered DataFrame.

        Returns:
            tuple: A tuple containing income, expenses, and savings.
        """
        summary = filtered_df.groupby('transaction_type')['transaction_amount'].sum()
        income = summary.get('Income', 0)
        expense = summary.get('Expense', 0)
        savings = income - expense
        return income, expense, savings

    def generate_bar_chart(self, income, expense, savings):
        """
        Generate a bar chart for income, expenses, and savings.

        Args:
            income (float): Total income.
            expense (float): Total expenses.
            savings (float): Net savings.

        Returns:
            go.Figure: The bar chart figure.
        """
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=['الدخل', 'المصروفات', 'المدخرات'],
            y=[income, expense, max(savings, 0)],  # Ensure savings are non-negative
            marker_color=['blue', 'orange', 'green']
        ))
        bar_fig.update_layout(template='plotly_white', height=320, width=520)
        return bar_fig

    def generate_pie_chart(self, income, expense, savings):
        """
        Generate a pie chart for income, expenses, and savings.

        Args:
            income (float): Total income.
            expense (float): Total expenses.
            savings (float): Net savings.

        Returns:
            go.Figure: The pie chart figure.
        """
        labels = ['الدخل', 'المصروفات', 'المدخرات']
        values = [income, expense, max(savings, 0)]  # Ensure savings are non-negative
        filtered_labels = [label for label, value in zip(labels, values) if value > 0]
        filtered_values = [value for value in values if value > 0]

        pie_fig = go.Figure()
        pie_fig.add_trace(go.Pie(
            labels=filtered_labels,
            values=filtered_values,
            textinfo='none',
            hoverinfo='label+value',
            marker=dict(colors=['blue', 'orange', 'green'])
        ))
        pie_fig.update_layout(template='plotly_white', height=320, width=520)
        return pie_fig
    
    def generate_financial_analysis(self, income, expense, savings):
        """
        Generate financial analysis text with formatted numbers.
    
        Args:
            income (float): Total income.
            expense (float): Total expenses.
            savings (float): Net savings.
    
        Returns:
            str: The financial analysis text.
        """
        # Format numbers to 2 decimal places
        income_formatted = f"{float(income):.2f}"
        expense_formatted = f"{float(expense):.2f}"
        savings_formatted = f"{float(savings):.2f}"
    
        financial_analysis = f"""
        <b>التحليل المالي</b><br>
        إجمالي الدخل: {income_formatted} ريال سعودي<br>
        إجمالي المصروفات: {expense_formatted} ريال سعودي<br>
        صافي المدخرات: {max(float(savings_formatted), 0)} ريال سعودي<br>
        """
        if savings > 0:
            financial_analysis += "الملاحظة الرئيسية: صافي مدخراتك إيجابي، مما يشير إلى إدارة مالية جيدة. فكر في زيادة الاستثمارات للنمو المستقبلي."
        elif savings < 0:
            financial_analysis += "الملاحظة الرئيسية: صافي مدخراتك سلبي، مما يشير إلى زيادة المصروفات. راجع ميزانيتك للتحكم في النفقات."
        else:
            financial_analysis += "الملاحظة الرئيسية: صافي مدخراتك صفر، حاول تحسين التوازن بين الدخل والمصروفات."
    
        return financial_analysis