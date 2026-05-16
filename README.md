# 💰 Smart Expense Tracker & Overspending Predictor

A beginner-friendly machine learning web application built with Python and Streamlit.
Track your daily expenses and let AI predict if you're at risk of overspending!

---

## 📁 Project Structure

```
smart_expense_tracker/
│
├── app.py            ← Main Streamlit UI (run this)
├── database.py       ← SQLite database operations
├── model.py          ← ML model (Random Forest + Linear Regression)
├── charts.py         ← Matplotlib chart functions
├── seed_data.py      ← Generate sample data for demo
├── requirements.txt  ← Python package dependencies
└── README.md         ← This file
```

---

## 🚀 How to Run

### Step 1 — Install Python
Make sure Python 3.8+ is installed. Download from: https://python.org

### Step 2 — Install Dependencies
Open your terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Add Sample Data
To see the ML prediction and charts working right away, seed the database:
```bash
python seed_data.py
```
This adds 6 months of realistic expense data so the AI model has enough to train on.

### Step 4 — Launch the App
```bash
streamlit run app.py
```
A browser window will open automatically at **http://localhost:8501**

---

## 🧠 Machine Learning Concepts Used

| Concept | Used For |
|---|---|
| **Random Forest Classifier** | Predicting if user will overspend |
| **Linear Regression** | Projecting end-of-month spending |
| **Feature Engineering** | Converting raw data into model inputs |
| **Train/Test Split** | Evaluating model accuracy |

### Features the model learns from:
1. `day_of_month` — How far into the month are we?
2. `daily_avg` — Average daily spending so far
3. `projected_total` — Estimated 30-day total
4. `budget` — User's monthly budget
5. `budget_ratio` — % of budget used
6. `category_count` — How many spending categories used

---

## 📊 App Pages

| Page | Description |
|---|---|
| **Add Expense** | Log daily expenses with category, amount, date |
| **Dashboard** | Monthly overview with budget progress and charts |
| **Analytics** | Category breakdown, bar charts, monthly trends |
| **AI Prediction** | ML-powered overspending probability and alerts |
| **History** | Browse, filter, and export all expenses |

---

## 💡 Tips for Students

- **SQLite** is a file-based database. All data is stored in `expenses.db` (created automatically).
- The **ML model needs 5+ months** of data to train. Use `seed_data.py` to demo it.
- Each Python file has a clear purpose — this is called **separation of concerns**.
- All functions have **docstrings** explaining what they do and why.
- The UI layer (`app.py`) never talks to the database directly — it uses `database.py` functions.

---

## 🛠️ Tech Stack

- **Streamlit** — Python web UI framework (no HTML/JS needed!)
- **Pandas** — Data manipulation and analysis
- **Scikit-learn** — Machine learning models
- **Matplotlib** — Charts and visualizations
- **SQLite** — Lightweight local database

---

Made for educational purposes. Happy learning! 🎓
