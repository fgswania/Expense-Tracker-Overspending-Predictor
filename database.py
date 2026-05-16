"""
database.py - Handles all SQLite database operations
This file creates and manages the expenses database.
Think of this as your data storage layer.
"""

import sqlite3
import pandas as pd
from datetime import datetime

# Name of our database file - SQLite stores everything in a single .db file
DB_NAME = "expenses.db"


def get_connection():
    """
    Create and return a connection to the SQLite database.
    SQLite is a lightweight database that saves data in a local file.
    """
    conn = sqlite3.connect(DB_NAME)
    return conn


def initialize_database():
    """
    Create the expenses table if it doesn't already exist.
    This runs once when the app starts up.
    
    Table columns:
    - id: auto-incremented unique ID for each expense
    - date: when the expense occurred
    - category: type of expense (Food, Transport, etc.)
    - amount: how much was spent
    - description: optional note about the expense
    - month_budget: the user's budget for that month
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            description TEXT,
            month_budget REAL DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()


def add_expense(date, category, amount, description, month_budget):
    """
    Insert a new expense record into the database.
    
    Parameters:
    - date: date string (e.g. "2024-01-15")
    - category: category name (e.g. "Food")
    - amount: expense amount (e.g. 250.0)
    - description: short note (e.g. "Lunch at work")
    - month_budget: user's monthly budget
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO expenses (date, category, amount, description, month_budget)
        VALUES (?, ?, ?, ?, ?)
    """, (date, category, amount, description, month_budget))

    conn.commit()
    conn.close()


def get_all_expenses():
    """
    Retrieve all expense records from the database.
    Returns a Pandas DataFrame for easy data manipulation.
    """
    conn = get_connection()
    
    # Use pandas to read SQL query result directly into a DataFrame
    df = pd.read_sql_query("""
        SELECT * FROM expenses ORDER BY date DESC
    """, conn)
    
    conn.close()
    return df


def get_expenses_by_month(year, month):
    """
    Get expenses for a specific month.
    Useful for calculating monthly totals and predictions.
    
    Parameters:
    - year: e.g. 2024
    - month: e.g. 1 (January), 12 (December)
    """
    conn = get_connection()
    
    # Format month to 2 digits: month 1 → "01", month 12 → "12"
    month_str = f"{year}-{month:02d}"
    
    df = pd.read_sql_query("""
        SELECT * FROM expenses
        WHERE strftime('%Y-%m', date) = ?
        ORDER BY date ASC
    """, conn, params=(month_str,))
    
    conn.close()
    return df


def get_monthly_summary():
    """
    Get a summary of total spending grouped by month.
    Used to train the ML model and show trends.
    """
    conn = get_connection()
    
    df = pd.read_sql_query("""
        SELECT 
            strftime('%Y-%m', date) AS month,
            SUM(amount) AS total_spent,
            MAX(month_budget) AS budget,
            COUNT(*) AS num_transactions
        FROM expenses
        GROUP BY strftime('%Y-%m', date)
        ORDER BY month ASC
    """, conn)
    
    conn.close()
    return df


def delete_expense(expense_id):
    """
    Delete a specific expense by its ID.
    
    Parameters:
    - expense_id: the unique ID of the expense to remove
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
    
    conn.commit()
    conn.close()
