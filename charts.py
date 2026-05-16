"""
charts.py - All chart and visualization functions
This file creates Matplotlib charts for displaying spending data.
Each function returns a Matplotlib figure that Streamlit can display.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from datetime import datetime

# Define a consistent color palette for categories
CATEGORY_COLORS = {
    "Food & Dining":     "#FF6B6B",
    "Transport":         "#4ECDC4",
    "Shopping":          "#45B7D1",
    "Entertainment":     "#96CEB4",
    "Healthcare":        "#FFEAA7",
    "Education":         "#DDA0DD",
    "Utilities":         "#98D8C8",
    "Rent/Housing":      "#F7DC6F",
    "Personal Care":     "#BB8FCE",
    "Other":             "#AEB6BF",
}

# Default colors for categories not in our dict
FALLBACK_COLORS = [
    "#FF9999", "#66B2FF", "#99FF99", "#FFCC99",
    "#FF99CC", "#99CCFF", "#FFD700", "#98FB98"
]


def get_color(category, index=0):
    """Return a color for a given category."""
    return CATEGORY_COLORS.get(category, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def plot_expense_pie_chart(df, title="Spending by Category"):
    """
    Create a pie chart showing the distribution of expenses by category.
    
    Parameters:
    - df: DataFrame with 'category' and 'amount' columns
    - title: chart title
    
    Returns: matplotlib Figure
    """
    if df.empty:
        return None
    
    # Group spending by category and sum the amounts
    category_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    
    # Assign colors to each category
    colors = [get_color(cat, i) for i, cat in enumerate(category_totals.index)]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0F1117")  # Dark background to match Streamlit dark theme
    ax.set_facecolor("#0F1117")
    
    # Draw pie chart with explosion effect on the largest slice
    explode = [0.05] * len(category_totals)
    
    wedges, texts, autotexts = ax.pie(
        category_totals.values,
        labels=None,           # We'll use a legend instead
        autopct="%1.1f%%",
        colors=colors,
        explode=explode,
        startangle=90,
        pctdistance=0.82,
        wedgeprops=dict(width=0.6, edgecolor="#0F1117", linewidth=2)  # Donut style
    )
    
    # Style percentage text
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(8)
        autotext.set_fontweight("bold")
    
    # Add legend with category names and amounts
    legend_labels = [f"{cat}  ₨{amt:,.0f}" for cat, amt in category_totals.items()]
    legend = ax.legend(
        wedges, legend_labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        title_fontsize=9,
        facecolor="#1E2130",
        edgecolor="#3D4462",
        labelcolor="white"
    )
    legend.get_title().set_color("white")
    
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    
    return fig


def plot_daily_spending_line(df, budget, title="Daily Spending Trend"):
    """
    Create a line chart showing daily cumulative spending vs budget.
    
    This helps users see if their spending is on track or accelerating.
    
    Parameters:
    - df: DataFrame with 'date' and 'amount' columns
    - budget: monthly budget amount
    - title: chart title
    
    Returns: matplotlib Figure
    """
    if df.empty:
        return None
    
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # Group by date and sum (multiple expenses on same day)
    daily = df.groupby("date")["amount"].sum().reset_index()
    daily["cumulative"] = daily["amount"].cumsum()
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1E2130")
    
    # Plot cumulative spending line
    ax.plot(
        daily["date"], daily["cumulative"],
        color="#4ECDC4", linewidth=2.5, marker="o",
        markersize=5, markerfacecolor="#FF6B6B",
        label="Cumulative Spending", zorder=3
    )
    
    # Shade area under spending line
    ax.fill_between(daily["date"], daily["cumulative"], alpha=0.15, color="#4ECDC4")
    
    # Add horizontal budget line
    ax.axhline(
        y=budget, color="#FF6B6B", linestyle="--",
        linewidth=2, label=f"Monthly Budget (₨{budget:,.0f})", zorder=2
    )
    
    # Color background red if spending is over budget
    if daily["cumulative"].max() > budget:
        ax.axhspan(budget, daily["cumulative"].max() * 1.05,
                   alpha=0.08, color="red")
    
    # Styling
    ax.set_xlabel("Date", color="#AAAAAA", fontsize=10)
    ax.set_ylabel("Cumulative Spending (₨)", color="#AAAAAA", fontsize=10)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#AAAAAA", labelsize=8)
    ax.spines[:].set_color("#3D4462")
    plt.xticks(rotation=30)
    
    legend = ax.legend(facecolor="#1E2130", edgecolor="#3D4462", labelcolor="white", fontsize=9)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₨{x:,.0f}"))
    plt.tight_layout()
    
    return fig


def plot_category_bar_chart(df, title="Spending per Category"):
    """
    Create a horizontal bar chart showing spending per category.
    Great for quick comparison between categories.
    
    Parameters:
    - df: DataFrame with 'category' and 'amount' columns
    - title: chart title
    
    Returns: matplotlib Figure
    """
    if df.empty:
        return None
    
    category_totals = df.groupby("category")["amount"].sum().sort_values()
    colors = [get_color(cat, i) for i, cat in enumerate(category_totals.index)]
    
    fig, ax = plt.subplots(figsize=(8, max(3, len(category_totals) * 0.7)))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1E2130")
    
    bars = ax.barh(
        category_totals.index,
        category_totals.values,
        color=colors, edgecolor="#0F1117",
        linewidth=0.5, height=0.6
    )
    
    # Add value labels at end of each bar
    for bar, value in zip(bars, category_totals.values):
        ax.text(
            bar.get_width() + category_totals.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"₨{value:,.0f}",
            va="center", ha="left",
            color="white", fontsize=9
        )
    
    ax.set_xlabel("Amount (₨)", color="#AAAAAA", fontsize=10)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#AAAAAA", labelsize=9)
    ax.spines[:].set_color("#3D4462")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₨{x:,.0f}"))
    
    plt.tight_layout()
    return fig


def plot_monthly_overview(df_monthly, title="Monthly Spending vs Budget"):
    """
    Bar chart comparing monthly spending against the budget over time.
    Helps users see spending trends across months.
    
    Parameters:
    - df_monthly: DataFrame with 'month', 'total_spent', 'budget' columns
    - title: chart title
    
    Returns: matplotlib Figure
    """
    if df_monthly.empty or len(df_monthly) < 2:
        return None
    
    months = df_monthly["month"].tolist()
    spent = df_monthly["total_spent"].tolist()
    budgets = df_monthly["budget"].tolist()
    
    x = np.arange(len(months))
    width = 0.35  # Bar width
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1E2130")
    
    # Draw side-by-side bars
    bars1 = ax.bar(x - width/2, spent, width, label="Spent",
                   color="#4ECDC4", alpha=0.85, edgecolor="#0F1117")
    bars2 = ax.bar(x + width/2, budgets, width, label="Budget",
                   color="#FF6B6B", alpha=0.7, edgecolor="#0F1117")
    
    # Add value labels on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + max(spent)*0.01,
                f"₨{h:,.0f}", ha="center", va="bottom", color="white", fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=30, color="#AAAAAA", fontsize=8)
    ax.set_ylabel("Amount (₨)", color="#AAAAAA", fontsize=10)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#AAAAAA")
    ax.spines[:].set_color("#3D4462")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₨{x:,.0f}"))
    
    legend = ax.legend(facecolor="#1E2130", edgecolor="#3D4462", labelcolor="white", fontsize=9)
    
    plt.tight_layout()
    return fig
