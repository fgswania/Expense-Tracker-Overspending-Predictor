"""
seed_data.py - Populate the database with realistic sample data
Run this ONCE before launching the app to see charts and AI predictions work.

Usage:
    python seed_data.py

This generates 6 months of sample expenses so the ML model
has enough historical data to train on (needs 5+ months).
"""

import random
from datetime import datetime, timedelta
from database import initialize_database, add_expense

# ─────────────────────────────────────
# Configuration
# ─────────────────────────────────────
MONTHLY_BUDGET = 50000.0  # Rupees per month

# Category templates: (category_name, min_amount, max_amount, avg_per_month)
CATEGORY_TEMPLATES = [
    ("Food & Dining",   150,  2000, 12),   # ~12 transactions/month
    ("Transport",        80,   800,  8),
    ("Shopping",        500,  5000,  4),
    ("Entertainment",   200,  1500,  3),
    ("Healthcare",      500,  3000,  2),
    ("Education",      1000,  8000,  1),
    ("Utilities",      2000,  8000,  1),
    ("Personal Care",   100,   800,  3),
]

FOOD_DESCRIPTIONS   = ["Lunch", "Dinner", "Breakfast", "Groceries", "Coffee", "Pizza", "Biryani", "Snacks"]
TRANSPORT_DESC      = ["Uber", "Rickshaw", "Bus fare", "Petrol", "Parking"]
SHOPPING_DESC       = ["Clothes", "Electronics", "Books", "Home items", "Accessories"]
ENTERTAINMENT_DESC  = ["Cinema ticket", "Netflix", "Game", "Event ticket"]
HEALTHCARE_DESC     = ["Doctor visit", "Medicines", "Lab test", "Pharmacy"]
EDUCATION_DESC      = ["Course fee", "Books", "Stationery", "Workshop"]
UTILITIES_DESC      = ["Electricity bill", "Internet bill", "Gas bill", "Water bill"]
PERSONAL_DESC       = ["Haircut", "Salon", "Skincare", "Gym"]

CATEGORY_DESCRIPTIONS = {
    "Food & Dining":  FOOD_DESCRIPTIONS,
    "Transport":      TRANSPORT_DESC,
    "Shopping":       SHOPPING_DESC,
    "Entertainment":  ENTERTAINMENT_DESC,
    "Healthcare":     HEALTHCARE_DESC,
    "Education":      EDUCATION_DESC,
    "Utilities":      UTILITIES_DESC,
    "Personal Care":  PERSONAL_DESC,
}

def generate_expenses_for_month(year, month, budget, overspend_chance=0.3):
    """
    Generate realistic expense records for a given month.
    
    Parameters:
    - year, month: target month
    - budget: monthly budget
    - overspend_chance: probability (0-1) the user overspends this month
    """
    import calendar
    num_days = calendar.monthrange(year, month)[1]
    
    # Randomly decide if this month is an overspending month
    is_overspend_month = random.random() < overspend_chance
    spending_multiplier = random.uniform(1.1, 1.4) if is_overspend_month else random.uniform(0.6, 0.95)
    
    expenses = []
    
    for category, min_amt, max_amt, avg_count in CATEGORY_TEMPLATES:
        # Vary the number of transactions per category
        num_transactions = max(1, int(avg_count * random.uniform(0.7, 1.3)))
        
        for _ in range(num_transactions):
            # Random day within the month
            day = random.randint(1, num_days)
            expense_date = f"{year}-{month:02d}-{day:02d}"
            
            # Random amount within category range, scaled by spending multiplier
            amount = round(random.uniform(min_amt, max_amt) * spending_multiplier, 0)
            
            # Random description
            descriptions = CATEGORY_DESCRIPTIONS.get(category, ["Expense"])
            description = random.choice(descriptions)
            
            expenses.append((expense_date, category, amount, description, budget))
    
    return expenses


def seed_database():
    """Generate 6 months of sample data and insert into database."""
    
    initialize_database()
    
    print("🌱 Seeding database with 6 months of sample expenses...")
    print("=" * 50)
    
    now = datetime.now()
    total_inserted = 0
    
    # Generate data for the past 6 months (including current)
    for months_ago in range(5, -1, -1):
        # Calculate target month
        target_date = now - timedelta(days=months_ago * 30)
        year = target_date.year
        month = target_date.month
        
        # Generate expenses
        expenses = generate_expenses_for_month(year, month, MONTHLY_BUDGET)
        
        month_total = sum(e[2] for e in expenses)
        label = "OVERSPENT ⚠️" if month_total > MONTHLY_BUDGET else "within budget ✅"
        
        print(f"📅 {year}-{month:02d}: {len(expenses)} expenses, ₨{month_total:,.0f} ({label})")
        
        # Insert into database
        for date_str, category, amount, description, budget in expenses:
            add_expense(date_str, category, amount, description, budget)
        
        total_inserted += len(expenses)
    
    print("=" * 50)
    print(f"✅ Done! Inserted {total_inserted} expense records.")
    print("\n🚀 Now run the app with:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    seed_database()
