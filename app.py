"""
app.py - Main Streamlit Application
Smart Expense Tracker & Overspending Predictor

Run this file with: streamlit run app.py

This is the UI layer. It uses Streamlit to create an interactive
web app without needing HTML, CSS, or JavaScript knowledge.

File structure:
    app.py       ← You are here (UI layer)
    database.py  ← Data storage (SQLite)
    model.py     ← Machine Learning predictions
    charts.py    ← Matplotlib visualizations
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
import calendar

# Import our custom modules
from database import (
    initialize_database, add_expense, get_all_expenses,
    get_expenses_by_month, get_monthly_summary, delete_expense
)
from model import train_model, predict_overspending, predict_end_of_month_spending
from charts import (
    plot_expense_pie_chart, plot_daily_spending_line,
    plot_category_bar_chart, plot_monthly_overview
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION - must be first Streamlit command
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS - Modern dark UI styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0F1117 0%, #141824 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141824 0%, #1a2035 100%);
        border-right: 1px solid #2D3454;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1E2130 0%, #242840 100%);
        border: 1px solid #2D3454;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #4ECDC4;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2D3454;
    }

    /* Alert boxes */
    .alert-danger {
        background: linear-gradient(135deg, #FF4757, #c0392b);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255, 71, 87, 0.4);
        animation: pulse 2s infinite;
    }
    .alert-warning {
        background: linear-gradient(135deg, #FFA502, #e67e22);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        margin: 10px 0;
    }
    .alert-success {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        margin: 10px 0;
    }

    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(255,71,87,0.5); }
        70%  { box-shadow: 0 0 0 10px rgba(255,71,87,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,71,87,0); }
    }

    /* Progress bar container */
    .budget-bar-container {
        background: #2D3454;
        border-radius: 8px;
        height: 14px;
        margin: 8px 0;
        overflow: hidden;
    }
    .budget-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }

    /* Table styling */
    .stDataFrame {
        border: 1px solid #2D3454;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4ECDC4, #2cb5ad);
        color: #0F1117;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-family: 'Sora', sans-serif;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(78,205,196,0.4);
    }

    /* App title */
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: #8892B0;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# INITIALIZE DATABASE (runs once on app startup)
# ─────────────────────────────────────────────────────────────
initialize_database()

# ─────────────────────────────────────────────────────────────
# EXPENSE CATEGORIES
# ─────────────────────────────────────────────────────────────
CATEGORIES = [
    "Food & Dining", "Transport", "Shopping", "Entertainment",
    "Healthcare", "Education", "Utilities", "Rent/Housing",
    "Personal Care", "Other"
]

# ─────────────────────────────────────────────────────────────
# SIDEBAR - Navigation & Settings
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-title">💰 Smart Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">AI-powered expense analysis</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Navigation menu
    page = st.radio(
        "📌 Navigate",
        options=["➕ Add Expense", "📊 Dashboard", "📈 Analytics", "🤖 AI Prediction", "🗂️ History"],
        index=0
    )
    
    st.divider()
    
    # Monthly budget setting
    st.markdown("### 🎯 Monthly Budget")
    monthly_budget = st.number_input(
        "Set your budget (₨)",
        min_value=100.0,
        max_value=1_000_000.0,
        value=50000.0,
        step=500.0,
        help="Set your total spending budget for the current month"
    )
    
    # Show current month info
    now = datetime.now()
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    days_passed = now.day
    days_remaining = days_in_month - days_passed
    
    st.markdown(f"""
    <div style="background:#1E2130; padding:12px; border-radius:10px; border:1px solid #2D3454; font-size:0.85rem; color:#AAAAAA; margin-top:8px;">
        📅 <b style="color:white;">{now.strftime('%B %Y')}</b><br>
        Day {days_passed} of {days_in_month}<br>
        {days_remaining} days remaining
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA (fetched once, used across all pages)
# ─────────────────────────────────────────────────────────────
all_expenses = get_all_expenses()
current_month_df = get_expenses_by_month(now.year, now.month)
monthly_summary = get_monthly_summary()

# Calculate current month stats
total_spent_this_month = current_month_df["amount"].sum() if not current_month_df.empty else 0
remaining_budget = monthly_budget - total_spent_this_month
budget_used_pct = min((total_spent_this_month / monthly_budget * 100), 100) if monthly_budget > 0 else 0

# ─────────────────────────────────────────────────────────────
# PAGE 1: ADD EXPENSE
# ─────────────────────────────────────────────────────────────
if page == "➕ Add Expense":
    st.markdown('<div class="app-title">➕ Add New Expense</div>', unsafe_allow_html=True)
    st.markdown("Track every rupee you spend to get accurate predictions.")
    
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("expense_form", clear_on_submit=True):
            st.markdown('<div class="section-header">Expense Details</div>', unsafe_allow_html=True)
            
            # Row 1: Date and Category
            r1_col1, r1_col2 = st.columns(2)
            with r1_col1:
                expense_date = st.date_input(
                    "📅 Date",
                    value=date.today(),
                    help="When did you spend this?"
                )
            with r1_col2:
                category = st.selectbox(
                    "🏷️ Category",
                    CATEGORIES,
                    help="What type of expense is this?"
                )
            
            # Row 2: Amount
            amount = st.number_input(
                "💵 Amount (₨)",
                min_value=1.0,
                max_value=500_000.0,
                value=100.0,
                step=10.0,
                help="How much did you spend?"
            )
            
            # Row 3: Description
            description = st.text_input(
                "📝 Description (optional)",
                placeholder="e.g., Lunch with friends, Uber ride...",
                max_chars=100
            )
            
            # Submit button
            submitted = st.form_submit_button(
                "✅ Add Expense",
                use_container_width=True
            )
        
        # Handle form submission
        if submitted:
            add_expense(
                date=str(expense_date),
                category=category,
                amount=amount,
                description=description,
                month_budget=monthly_budget
            )
            st.success(f"✅ Expense of ₨{amount:,.0f} added under **{category}**!")
            st.balloons()
            st.rerun()  # Refresh app to show updated data
    
    with col2:
        st.markdown('<div class="section-header">This Month So Far</div>', unsafe_allow_html=True)
        
        # Quick stats
        st.metric("Total Spent", f"₨{total_spent_this_month:,.0f}")
        st.metric("Budget Remaining", f"₨{remaining_budget:,.0f}",
                  delta=f"{remaining_budget:+,.0f}", delta_color="normal")
        
        # Budget progress bar
        bar_color = "#FF6B6B" if budget_used_pct > 90 else "#FFA502" if budget_used_pct > 70 else "#4ECDC4"
        st.markdown(f"""
        <div style="margin-top:10px;">
            <div style="color:#AAAAAA; font-size:0.85rem; margin-bottom:4px;">
                Budget Used: {budget_used_pct:.1f}%
            </div>
            <div class="budget-bar-container">
                <div class="budget-bar-fill" style="width:{budget_used_pct}%; background:{bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show budget warning if needed
        if budget_used_pct >= 100:
            st.markdown('<div class="alert-danger">🚨 Over Budget!</div>', unsafe_allow_html=True)
        elif budget_used_pct >= 80:
            st.markdown('<div class="alert-warning">⚠️ 80% of budget used</div>', unsafe_allow_html=True)
        
        # Recent expenses (last 5)
        if not current_month_df.empty:
            st.markdown('<div class="section-header">Recent Expenses</div>', unsafe_allow_html=True)
            recent = current_month_df.tail(5)[["date", "category", "amount"]].copy()
            recent["amount"] = recent["amount"].apply(lambda x: f"₨{x:,.0f}")
            st.dataframe(recent, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 2: DASHBOARD
# ─────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.markdown('<div class="app-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f"Overview for **{now.strftime('%B %Y')}**")
    
    st.divider()
    
    # ── Key Metrics Row ──
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric(
            "💸 Total Spent",
            f"₨{total_spent_this_month:,.0f}",
            help="Total spending this month"
        )
    with m2:
        st.metric(
            "🎯 Monthly Budget",
            f"₨{monthly_budget:,.0f}",
            help="Your set budget for this month"
        )
    with m3:
        color = "inverse" if remaining_budget < 0 else "normal"
        st.metric(
            "💰 Remaining",
            f"₨{abs(remaining_budget):,.0f}",
            delta=f"{'Over' if remaining_budget < 0 else 'Under'} budget",
            delta_color=color
        )
    with m4:
        num_transactions = len(current_month_df)
        st.metric(
            "📋 Transactions",
            num_transactions,
            help="Number of expense entries this month"
        )
    
    # ── Budget Alerts ──
    if budget_used_pct >= 100:
        st.markdown(f'<div class="alert-danger">🚨 OVERSPENT! You have exceeded your budget by ₨{abs(remaining_budget):,.0f}. Review your expenses immediately!</div>', unsafe_allow_html=True)
    elif budget_used_pct >= 90:
        st.markdown(f'<div class="alert-warning">⚠️ Critical: {budget_used_pct:.1f}% of your monthly budget has been used. Only ₨{remaining_budget:,.0f} left!</div>', unsafe_allow_html=True)
    elif budget_used_pct >= 70:
        st.markdown(f'<div class="alert-warning">⚠️ Heads up: {budget_used_pct:.1f}% of budget used. Spend wisely for the rest of the month.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-success">✅ On track! You have used {budget_used_pct:.1f}% of your budget. Keep it up!</div>', unsafe_allow_html=True)
    
    # ── Budget Progress Bar ──
    bar_color = "#FF6B6B" if budget_used_pct > 90 else "#FFA502" if budget_used_pct > 70 else "#4ECDC4"
    st.markdown(f"""
    <div style="margin:16px 0;">
        <div style="display:flex; justify-content:space-between; color:#AAAAAA; font-size:0.85rem; margin-bottom:6px;">
            <span>Budget Progress</span>
            <span>{budget_used_pct:.1f}% used</span>
        </div>
        <div class="budget-bar-container" style="height:18px;">
            <div class="budget-bar-fill" style="width:{budget_used_pct}%; background:{bar_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Charts ──
    if not current_month_df.empty:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown('<div class="section-header">Spending by Category</div>', unsafe_allow_html=True)
            pie = plot_expense_pie_chart(current_month_df, "")
            if pie:
                st.pyplot(pie, use_container_width=True)
        
        with chart_col2:
            st.markdown('<div class="section-header">Daily Cumulative Spending</div>', unsafe_allow_html=True)
            line = plot_daily_spending_line(current_month_df, monthly_budget, "")
            if line:
                st.pyplot(line, use_container_width=True)
    else:
        st.info("📭 No expenses recorded this month yet. Add your first expense to see charts!")

# ─────────────────────────────────────────────────────────────
# PAGE 3: ANALYTICS
# ─────────────────────────────────────────────────────────────
elif page == "📈 Analytics":
    st.markdown('<div class="app-title">📈 Analytics</div>', unsafe_allow_html=True)
    st.markdown("Deep dive into your spending patterns.")
    
    st.divider()
    
    # Time range filter
    view_option = st.radio(
        "View Period",
        ["This Month", "All Time"],
        horizontal=True
    )
    
    df_view = current_month_df if view_option == "This Month" else all_expenses
    view_label = now.strftime('%B %Y') if view_option == "This Month" else "All Time"
    
    if not df_view.empty:
        # Category breakdown table
        st.markdown('<div class="section-header">Category Breakdown</div>', unsafe_allow_html=True)
        
        cat_summary = df_view.groupby("category").agg(
            Total=("amount", "sum"),
            Count=("amount", "count"),
            Average=("amount", "mean"),
            Highest=("amount", "max")
        ).reset_index().sort_values("Total", ascending=False)
        
        # Format currency columns
        for col in ["Total", "Average", "Highest"]:
            cat_summary[col] = cat_summary[col].apply(lambda x: f"₨{x:,.0f}")
        
        st.dataframe(cat_summary, hide_index=True, use_container_width=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Category Comparison</div>', unsafe_allow_html=True)
            bar_chart = plot_category_bar_chart(df_view, "")
            if bar_chart:
                st.pyplot(bar_chart, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header">Spending Distribution</div>', unsafe_allow_html=True)
            pie_chart = plot_expense_pie_chart(df_view, "")
            if pie_chart:
                st.pyplot(pie_chart, use_container_width=True)
        
        # Monthly trend (only if all-time view and multiple months)
        if view_option == "All Time" and not monthly_summary.empty and len(monthly_summary) > 1:
            st.markdown('<div class="section-header">Monthly Spending Trend</div>', unsafe_allow_html=True)
            monthly_chart = plot_monthly_overview(monthly_summary, "")
            if monthly_chart:
                st.pyplot(monthly_chart, use_container_width=True)
    else:
        st.info("📭 No data for the selected period. Start adding expenses!")

# ─────────────────────────────────────────────────────────────
# PAGE 4: AI PREDICTION
# ─────────────────────────────────────────────────────────────
elif page == "🤖 AI Prediction":
    st.markdown('<div class="app-title">🤖 AI Overspending Predictor</div>', unsafe_allow_html=True)
    st.markdown("Machine learning model that predicts if you'll exceed your budget.")
    
    st.divider()
    
    # ── How the ML Model Works (Explainer) ──
    with st.expander("ℹ️ How does the AI Prediction work?"):
        st.markdown("""
        **Random Forest Classifier** — Here's how it works in simple terms:
        
        1. 📚 **Training**: The model learns from your past months of spending data
        2. 🌳 **Decision Trees**: It builds 100 decision trees, each analyzing different patterns
        3. 🗳️ **Voting**: All trees vote — majority decides if you'll overspend
        4. 📊 **Features used**: Day of month, daily average, projected total, budget ratio, category count
        5. 🎯 **Output**: Overspend probability (0% = safe, 100% = definitely overspending)
        
        **Fallback rule** (when not enough data): If projected 30-day spending > budget → Overspending alert
        
        > 💡 The model improves with more data. Add expenses daily for better predictions!
        """)
    
    # ── Train Model ──
    st.markdown('<div class="section-header">Model Training</div>', unsafe_allow_html=True)
    
    if st.button("🔁 Train / Refresh Model", use_container_width=False):
        with st.spinner("Training Random Forest model on your data..."):
            model, status_msg = train_model(monthly_summary, all_expenses)
            st.session_state["model"] = model
            st.session_state["model_status"] = status_msg
    
    # Auto-train if not done yet
    if "model" not in st.session_state:
        model, status_msg = train_model(monthly_summary, all_expenses)
        st.session_state["model"] = model
        st.session_state["model_status"] = status_msg
    
    model = st.session_state.get("model")
    model_status = st.session_state.get("model_status", "")
    
    if model:
        st.success(f"✅ {model_status}")
    else:
        st.info(f"ℹ️ {model_status}. Using rule-based prediction as fallback.")
    
    # ── Prediction ──
    st.markdown('<div class="section-header">Current Month Prediction</div>', unsafe_allow_html=True)
    
    if not current_month_df.empty:
        # Calculate prediction inputs
        days_passed_float = max(now.day, 1)
        daily_avg = total_spent_this_month / days_passed_float
        projected_total = daily_avg * calendar.monthrange(now.year, now.month)[1]
        category_count = current_month_df["category"].nunique()
        
        # Get ML prediction
        will_overspend, probability = predict_overspending(
            model,
            current_day=days_passed_float,
            daily_avg=daily_avg,
            projected_total=projected_total,
            budget=monthly_budget,
            category_count=category_count
        )
        
        # Get linear regression end-of-month estimate
        lr_prediction = predict_end_of_month_spending(current_month_df)
        
        # Display prediction results
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            # Overspend probability meter
            prob_pct = probability * 100
            prob_color = "#FF6B6B" if prob_pct > 70 else "#FFA502" if prob_pct > 40 else "#4ECDC4"
            
            st.markdown(f"""
            <div style="background:#1E2130; border:1px solid #2D3454; border-radius:14px; padding:20px; text-align:center;">
                <div style="font-size:0.9rem; color:#AAAAAA; margin-bottom:8px;">Overspend Probability</div>
                <div style="font-size:3rem; font-weight:700; color:{prob_color};">{prob_pct:.0f}%</div>
                <div style="color:{prob_color}; font-size:1rem; font-weight:600; margin-top:6px;">
                    {"🔴 HIGH RISK" if prob_pct > 70 else "🟡 MEDIUM RISK" if prob_pct > 40 else "🟢 LOW RISK"}
                </div>
                <div class="budget-bar-container" style="margin-top:12px; height:10px;">
                    <div class="budget-bar-fill" style="width:{prob_pct}%; background:{prob_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_col2:
            # Spending projection
            proj_diff = lr_prediction - monthly_budget if lr_prediction else projected_total - monthly_budget
            proj_total = lr_prediction if lr_prediction else projected_total
            
            st.markdown(f"""
            <div style="background:#1E2130; border:1px solid #2D3454; border-radius:14px; padding:20px;">
                <div style="font-size:0.9rem; color:#AAAAAA; margin-bottom:12px;">📊 Month-End Projection</div>
                <div style="margin-bottom:10px;">
                    <span style="color:#AAAAAA; font-size:0.85rem;">Current Daily Avg:</span>
                    <span style="color:white; font-weight:600; float:right;">₨{daily_avg:,.0f}/day</span>
                </div>
                <div style="margin-bottom:10px;">
                    <span style="color:#AAAAAA; font-size:0.85rem;">Projected Total:</span>
                    <span style="color:{'#FF6B6B' if proj_total > monthly_budget else '#4ECDC4'}; font-weight:700; float:right;">₨{proj_total:,.0f}</span>
                </div>
                <div style="margin-bottom:10px;">
                    <span style="color:#AAAAAA; font-size:0.85rem;">Your Budget:</span>
                    <span style="color:white; font-weight:600; float:right;">₨{monthly_budget:,.0f}</span>
                </div>
                <div style="border-top:1px solid #2D3454; padding-top:10px; margin-top:4px;">
                    <span style="color:#AAAAAA; font-size:0.85rem;">Expected Surplus/Deficit:</span>
                    <span style="color:{'#FF6B6B' if proj_diff > 0 else '#4ECDC4'}; font-weight:700; float:right;">
                        {'–' if proj_diff > 0 else '+'} ₨{abs(proj_diff):,.0f}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Big alert based on prediction
        st.markdown("<br>", unsafe_allow_html=True)
        if will_overspend == 1 or prob_pct > 70:
            st.markdown(f"""
            <div class="alert-danger">
                🚨 <b>AI WARNING:</b> Based on your current spending pattern, you are <b>{prob_pct:.0f}% likely</b> to overspend this month.<br>
                💡 Tip: Reduce daily spending by ₨{(proj_total - monthly_budget) / max(days_remaining, 1):,.0f}/day to stay within budget.
            </div>
            """, unsafe_allow_html=True)
        elif prob_pct > 40:
            st.markdown(f"""
            <div class="alert-warning">
                ⚠️ <b>CAUTION:</b> Moderate overspending risk detected ({prob_pct:.0f}%). Monitor your spending carefully for the rest of the month.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-success">
                ✅ <b>GREAT JOB!</b> AI predicts you will stay within your budget this month. Keep up the discipline!
            </div>
            """, unsafe_allow_html=True)
        
        # Spending projection line chart
        st.markdown('<div class="section-header">Spending Trend This Month</div>', unsafe_allow_html=True)
        line_chart = plot_daily_spending_line(current_month_df, monthly_budget)
        if line_chart:
            st.pyplot(line_chart, use_container_width=True)
    
    else:
        st.info("📭 No expenses recorded this month. Add some expenses to get a prediction!")

# ─────────────────────────────────────────────────────────────
# PAGE 5: HISTORY
# ─────────────────────────────────────────────────────────────
elif page == "🗂️ History":
    st.markdown('<div class="app-title">🗂️ Expense History</div>', unsafe_allow_html=True)
    st.markdown("View and manage all your recorded expenses.")
    
    st.divider()
    
    if not all_expenses.empty:
        # Filter controls
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Category filter
            all_cats = ["All Categories"] + sorted(all_expenses["category"].unique().tolist())
            selected_cat = st.selectbox("Filter by Category", all_cats)
        
        with filter_col2:
            # Month filter
            if "date" in all_expenses.columns:
                all_expenses["month"] = pd.to_datetime(all_expenses["date"]).dt.strftime("%Y-%m")
                all_months = ["All Months"] + sorted(all_expenses["month"].unique().tolist(), reverse=True)
                selected_month = st.selectbox("Filter by Month", all_months)
            else:
                selected_month = "All Months"
        
        with filter_col3:
            sort_by = st.selectbox("Sort By", ["Date (Newest)", "Date (Oldest)", "Amount (High)", "Amount (Low)"])
        
        # Apply filters
        filtered = all_expenses.copy()
        
        if selected_cat != "All Categories":
            filtered = filtered[filtered["category"] == selected_cat]
        
        if selected_month != "All Months":
            filtered = filtered[filtered["month"] == selected_month]
        
        # Apply sort
        sort_map = {
            "Date (Newest)": ("date", False),
            "Date (Oldest)": ("date", True),
            "Amount (High)": ("amount", False),
            "Amount (Low)": ("amount", True),
        }
        sort_col, sort_asc = sort_map[sort_by]
        filtered = filtered.sort_values(sort_col, ascending=sort_asc)
        
        # Summary stats for filtered view
        st.markdown(f"**{len(filtered)} records** | Total: **₨{filtered['amount'].sum():,.0f}**")
        
        # Display table
        display_df = filtered[["id", "date", "category", "amount", "description"]].copy()
        display_df["amount"] = display_df["amount"].apply(lambda x: f"₨{x:,.2f}")
        display_df.columns = ["ID", "Date", "Category", "Amount", "Description"]
        
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)
        
        # Delete expense
        st.markdown('<div class="section-header">Delete an Expense</div>', unsafe_allow_html=True)
        del_col1, del_col2 = st.columns([2, 1])
        
        with del_col1:
            delete_id = st.number_input(
                "Enter Expense ID to delete",
                min_value=1,
                step=1,
                help="Find the ID in the table above"
            )
        with del_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Delete", type="secondary"):
                delete_expense(int(delete_id))
                st.success(f"Expense #{delete_id} deleted.")
                st.rerun()
        
        # Export CSV
        st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)
        csv_data = filtered.to_csv(index=False)
        st.download_button(
            "⬇️ Download as CSV",
            data=csv_data,
            file_name=f"expenses_{now.strftime('%Y_%m')}.csv",
            mime="text/csv"
        )
    
    else:
        st.info("📭 No expense records found. Start adding expenses using the 'Add Expense' tab!")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#4A5568; font-size:0.8rem; padding:10px 0;">
    💰 Smart Expense Tracker &amp; Overspending Predictor &nbsp;|&nbsp; 
    Built with Python, Streamlit &amp; Scikit-learn &nbsp;|&nbsp;
    For educational purposes
</div>
""", unsafe_allow_html=True)
