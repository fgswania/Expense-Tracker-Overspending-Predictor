"""
model.py - Machine Learning model for predicting overspending
This file trains a Random Forest model to predict if a user
will overspend based on their spending patterns.

Concepts used:
- Feature Engineering: Creating useful input columns for the model
- Random Forest: An ensemble of decision trees (like a group vote)
- Train/Test Split: Dividing data to evaluate model performance
- Prediction: Using the trained model on new data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")  # Suppress sklearn warnings for clean output


def prepare_features(df_monthly, df_expenses):
    """
    Convert raw expense data into features the ML model can learn from.
    
    Feature Engineering - we create these input columns:
    1. day_of_month   → How far into the month are we?
    2. daily_avg      → Average spending per day so far
    3. projected_total → Estimated total if spending continues at this rate
    4. budget         → The monthly budget set by the user
    5. category_count → How many different categories were used
    
    Target (what we're predicting):
    - overspent: 1 if total_spent > budget, else 0
    """
    
    features = []
    labels = []

    for _, row in df_monthly.iterrows():
        month = row["month"]
        budget = row["budget"] if row["budget"] > 0 else 1  # Avoid divide by zero
        total_spent = row["total_spent"]
        
        # Get expenses for this specific month to calculate day-based features
        month_expenses = df_expenses[df_expenses["date"].str.startswith(month)]
        
        if len(month_expenses) == 0:
            continue
        
        # Parse dates and find the last day with an expense
        month_expenses = month_expenses.copy()
        month_expenses["date"] = pd.to_datetime(month_expenses["date"])
        last_day = month_expenses["date"].dt.day.max()
        
        # Feature 1: Which day of the month is the last recorded expense?
        day_of_month = last_day
        
        # Feature 2: Average daily spending so far
        daily_avg = total_spent / day_of_month if day_of_month > 0 else 0
        
        # Feature 3: Project total for full 30 days based on current rate
        projected_total = daily_avg * 30
        
        # Feature 4: Budget ratio (how much of budget has been used?)
        budget_ratio = total_spent / budget
        
        # Feature 5: Number of unique categories used
        category_count = month_expenses["category"].nunique()
        
        features.append([
            day_of_month,
            daily_avg,
            projected_total,
            budget,
            budget_ratio,
            category_count
        ])
        
        # Label: Did the user actually overspend this month?
        labels.append(1 if total_spent > budget else 0)

    if not features:
        return None, None

    X = pd.DataFrame(features, columns=[
        "day_of_month", "daily_avg", "projected_total",
        "budget", "budget_ratio", "category_count"
    ])
    y = np.array(labels)
    
    return X, y


def train_model(df_monthly, df_expenses):
    """
    Train a Random Forest classifier on historical spending data.
    
    Random Forest works by:
    1. Creating many decision trees (like many advisors)
    2. Each tree votes on whether you'll overspend
    3. The majority vote wins
    
    Returns the trained model, or None if not enough data.
    """
    
    X, y = prepare_features(df_monthly, df_expenses)
    
    # Need at least 5 months of data to train meaningfully
    if X is None or len(X) < 5:
        return None, "Not enough historical data to train model (need 5+ months)"
    
    # Split data: 80% for training, 20% for testing
    # This lets us evaluate how well the model generalizes
    if len(X) > 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
    
    # Initialize Random Forest with 100 decision trees
    model = RandomForestClassifier(
        n_estimators=100,   # Number of trees in the forest
        max_depth=5,         # Limit tree depth to prevent overfitting
        random_state=42      # Ensures reproducible results
    )
    
    # Train the model on historical data
    model.fit(X_train, y_train)
    
    # Evaluate model accuracy
    accuracy = model.score(X_test, y_test)
    
    return model, f"Model trained with {len(X)} months of data. Accuracy: {accuracy:.0%}"


def predict_overspending(model, current_day, daily_avg, projected_total, budget, category_count):
    """
    Use the trained model to predict if the user will overspend this month.
    
    Parameters:
    - model: trained RandomForest model
    - current_day: today's day of month (e.g. 15)
    - daily_avg: average spending per day so far
    - projected_total: estimated end-of-month total
    - budget: user's monthly budget
    - category_count: number of categories used
    
    Returns:
    - prediction: 1 (will overspend) or 0 (within budget)
    - probability: confidence score 0.0 to 1.0
    """
    
    if model is None:
        # If no model, use a simple rule-based fallback
        budget_ratio = (daily_avg * 30) / budget if budget > 0 else 0
        will_overspend = 1 if budget_ratio > 1.0 else 0
        confidence = min(abs(budget_ratio - 1.0) + 0.5, 1.0)
        return will_overspend, confidence
    
    budget_ratio = (daily_avg * 30) / budget if budget > 0 else 0
    
    # Build the feature row (same order as during training)
    features = pd.DataFrame([[
        current_day,
        daily_avg,
        projected_total,
        budget,
        budget_ratio,
        category_count
    ]], columns=[
        "day_of_month", "daily_avg", "projected_total",
        "budget", "budget_ratio", "category_count"
    ])
    
    # Get prediction (0 or 1)
    prediction = model.predict(features)[0]
    
    # Get probability scores for each class [prob_0, prob_1]
    # Handle edge case: if model only saw one class during training
    proba = model.predict_proba(features)[0]
    if len(proba) == 1:
        # Only one class seen; use budget_ratio heuristic as fallback
        probability = min(budget_ratio, 1.0) if prediction == 1 else 1.0 - min(budget_ratio, 1.0)
    else:
        probability = proba[1]  # Probability of class 1 (overspending)
    
    return prediction, probability


def predict_end_of_month_spending(df_current_month):
    """
    Use Linear Regression to predict total spending by end of month.
    
    Linear Regression finds the best straight line through your data.
    Here: X = day of month, y = cumulative spending
    
    Returns predicted total spending for day 30.
    """
    
    if df_current_month.empty or len(df_current_month) < 2:
        return None
    
    df = df_current_month.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["day"] = df["date"].dt.day
    
    # Calculate cumulative spending day by day
    df["cumulative"] = df["amount"].cumsum()
    
    # Prepare X (day numbers) and y (cumulative spending)
    X = df["day"].values.reshape(-1, 1)
    y = df["cumulative"].values
    
    # Train simple linear regression
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Predict spending on day 30 (end of month)
    predicted_day_30 = lr.predict([[30]])[0]
    
    return max(predicted_day_30, df["cumulative"].max())  # Never predict less than current
