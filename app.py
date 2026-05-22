import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime
import calendar

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Home Profit Analyzer",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  THEME / CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2d40 50%, #0f2033 100%);
    color: #e8ead4;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 25, 35, 0.95) !important;
    border-right: 1px solid rgba(180, 160, 100, 0.2);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #c8cab4 !important;
}

/* Headings */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #d4b870 !important;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(180,160,100,0.25);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(6px);
}
.metric-card .metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8a9bab;
    margin-bottom: 0.3rem;
}
.metric-card .metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: #d4b870;
    line-height: 1.1;
}
.metric-card .metric-sub {
    font-size: 0.72rem;
    color: #6a7b8b;
    margin-top: 0.25rem;
}
.metric-positive { color: #5dbf8a !important; }
.metric-negative { color: #e06060 !important; }

/* Section divider */
.section-rule {
    border: none;
    border-top: 1px solid rgba(180, 160, 100, 0.2);
    margin: 1.5rem 0;
}

/* Hero header */
.hero-header {
    background: linear-gradient(90deg, rgba(212,184,112,0.12) 0%, rgba(93,191,138,0.08) 100%);
    border-left: 4px solid #d4b870;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
}

/* Info box */
.info-box {
    background: rgba(93,191,138,0.08);
    border: 1px solid rgba(93,191,138,0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #9ed4b4;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────
def fmt_dollar(v):
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v:,.0f}"
    return f"${v:.2f}"

def fmt_signed(v):
    s = fmt_dollar(abs(v))
    return f"+{s}" if v >= 0 else f"-{s}"

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def compute_amortization(loan_amount, annual_rate, term_years, extra_monthly=0):
    """Return DataFrame with monthly amortization schedule."""
    n = term_years * 12
    r = annual_rate / 100 / 12
    if r == 0:
        payment = loan_amount / n
    else:
        payment = loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1)

    rows = []
    balance = loan_amount
    for mo in range(1, n + 1):
        interest = balance * r
        principal = min(payment - interest + extra_monthly, balance)
        balance -= principal
        balance = max(balance, 0)
        rows.append({
            "month": mo,
            "payment": payment + extra_monthly,
            "interest": interest,
            "principal": principal,
            "balance": balance,
        })
        if balance == 0:
            break
    return pd.DataFrame(rows)


def build_yearly_cost_df(amort_df, prop_tax_yr, tax_growth_pct, home_ins_yr, ins_growth_pct, hoa_mo, pmi_pct, loan_amount,
                          term_years):
    """Aggregate monthly costs into yearly buckets, growing taxes and insurance over time."""
    rows = []
    pmi_threshold = loan_amount * 0.80  # PMI drops when LTV < 80%

    for year in range(1, term_years + 1):
        start = (year - 1) * 12 + 1
        end = year * 12
        subset = amort_df[(amort_df["month"] >= start) & (amort_df["month"] <= end)]
        if subset.empty:
            break
            
        # Grow costs based on the year index (Year 1 uses initial cost base)
        current_tax_yr = prop_tax_yr * ((1 + tax_growth_pct / 100) ** (year - 1))
        current_ins_yr = home_ins_yr * ((1 + ins_growth_pct / 100) ** (year - 1))

        # PMI: charged while balance > 80% of original loan
        pmi_months = sum(
            1 for _, row in subset.iterrows()
            if (row["balance"] + row["principal"]) > pmi_threshold
        )
        avg_balance = subset["balance"].mean()
        pmi_annual = (avg_balance * pmi_pct / 100 / 12) * pmi_months

        rows.append({
            "Year": year,
            "Principal": subset["principal"].sum(),
            "Interest": subset["interest"].sum(),
            "Property Tax": current_tax_yr,
            "Home Insurance": current_ins_yr,
            "HOA": hoa_mo * 12,
            "PMI": pmi_annual,
        })
    return pd.DataFrame(rows)


def monte_carlo_home_value(home_value, n_years, n_sims=1000, annual_mean=0.04, annual_std=0.08):
    """Simulate home value paths. Returns array shape (n_sims, n_years+1)."""
    monthly_mean = annual_mean / 12
    monthly_std = annual_std / (12 ** 0.5)
    n_months = n_years * 12
    log_drift = monthly_mean - 0.5 * monthly_std ** 2
    shocks = np.random.normal(log_drift, monthly_std, size=(n_sims, n_months))
    log_returns = np.cumsum(shocks, axis=1)
    paths_monthly = home_value * np.exp(log_returns)
    # Return yearly snapshots
    yearly_paths = np.hstack([
        np.full((n_sims, 1), home_value),
        paths_monthly[:, 11::12]
    ])
    return yearly_paths[:, :n_years + 1]


# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏡 Loan Details")

    home_value = st.number_input("Home Value ($)", min_value=10_000, max_value=10_000_000,
                                  value=350_000, step=1_000, format="%d")

    dp_mode = st.radio("Down Payment as", ["Dollar ($)", "Percent (%)"], horizontal=True)
    if dp_mode == "Dollar ($)":
        down_payment = st.number_input("Down Payment ($)", min_value=0, max_value=home_value,
                                        value=min(70_000, home_value), step=500, format="%d")
        dp_pct = down_payment / home_value * 100
    else:
        dp_pct = st.slider("Down Payment (%)", 0.0, 100.0, 20.0, 0.5)
        down_payment = home_value * dp_pct / 100

    loan_amount = home_value - down_payment
    st.markdown(f"<div style='color:#8a9bab;font-size:0.82rem;margin-top:-0.5rem;margin-bottom:0.8rem;'>"
                f"Loan Amount: <b style='color:#d4b870'>{fmt_dollar(loan_amount)}</b> &nbsp;|&nbsp; "
                f"Down: <b style='color:#d4b870'>{dp_pct:.1f}%</b></div>", unsafe_allow_html=True)

    interest_rate = st.number_input("Interest Rate (%)", min_value=0.01, max_value=20.0,
                                     value=6.75, step=0.05, format="%.2f")
    term_years = st.selectbox("Loan Term (years)", [10, 15, 20, 25, 30], index=4)
    loan_type = st.selectbox("Loan Type", ["Conventional", "FHA", "VA", "USDA"])

    start_month = st.selectbox("Start Month", MONTHS, index=0)
    start_year = st.number_input("Start Year", min_value=2020, max_value=2060, value=2025, step=1)

    st.markdown("---")
    st.markdown("### 💰 Initial Monthly Costs")
    prop_tax_yr = st.number_input("Property Tax ($/yr)", min_value=0, value=4_200, step=100, format="%d")
    pmi_pct = st.number_input("PMI (%)", min_value=0.0, max_value=5.0, value=0.5 if dp_pct < 20 else 0.0,
                               step=0.05, format="%.2f")
    home_ins_yr = st.number_input("Home Insurance ($/yr)", min_value=0, value=1_800, step=100, format="%d")
    hoa_mo = st.number_input("Monthly HOA ($)", min_value=0, value=0, step=10, format="%d")

    st.markdown("---")
    st.markdown("### 🔄 Cost Growth Assumptions")
    tax_growth_pct = st.slider("Property Tax Annual Increase (%)", 0.0, 10.0, 2.0, 0.1)
    ins_growth_pct = st.slider("Insurance Annual Increase (%)", 0.0, 10.0, 3.0, 0.1)

    st.markdown("---")
    st.markdown("### 📅 Sale Scenario")
    sale_year_offset = st.slider("Sell After (years)", 1, term_years, 7)
    sale_month_sel = st.selectbox("Sale Month", MONTHS, index=5)
    sale_month_abs = sale_year_offset * 12 + MONTHS.index(sale_month_sel)

    st.markdown("---")
    st.markdown("### 📈 Market Assumptions")
    mc_mean = st.slider("Avg Annual Appreciation (%)", 0.0, 10.0, 4.0, 0.1) / 100
    mc_std  = st.slider("Annual Volatility (%)", 1.0, 20.0, 8.0, 0.1) / 100
    n_sims  = st.select_slider("Monte Carlo Simulations", [200, 500, 1000, 2000], value=1000)

    selling_cost_pct = st.slider("Selling Costs (% of sale price)", 0.0, 10.0, 6.0, 0.25) / 100


# ──────────────────────────────────────────────
#  COMPUTATIONS
# ──────────────────────────────────────────────
amort_df = compute_amortization(loan_amount, interest_rate, term_years)
yearly_df = build_yearly_cost_df(amort_df, prop_tax_yr, tax_growth_pct, home_ins_yr, ins_growth_pct, hoa_mo, pmi_pct,
                                  loan_amount, term_years)

# Calculate cumulative costs up to exact sale month factoring in cost inflation
total_months = min(sale_month_abs, len(amort_df))
sale_subset = amort_df[amort_df["month"] <= total_months]
total_interest_paid = sale_subset["interest"].sum()
total_principal_paid = sale_subset["principal"].sum()
remaining_balance = sale_subset["balance"].iloc[-1] if not sale_subset.empty else loan_amount

total_prop_tax = 0
total_ins = 0
total_hoa = hoa_mo * total_months

# Loop month-by-month up to sale to capture compounding inflation on carrying costs accurately
for m in range(total_months):
    current_yr_idx = m // 12
    total_prop_tax += (prop_tax_yr * ((1 + tax_growth_pct / 100) ** current_yr_idx)) / 12
    total_ins += (home_ins_yr * ((1 + ins_growth_pct / 100) ** current_yr_idx)) / 12

# PMI cost up to sale
pmi_threshold = loan_amount * 0.80
pmi_rows = amort_df[amort_df["month"] <= total_months]
pmi_months_paid = sum(
    1 for _, row in pmi_rows.iterrows()
    if (row["balance"] + row["principal"]) > pmi_threshold
)
avg_bal_for_pmi = pmi_rows["balance"].mean() if not pmi_rows.empty else loan_amount
total_pmi = (avg_bal_for_pmi * pmi_pct / 100 / 12) * pmi_months_paid

total_non_equity = (total_interest_paid + total_prop_tax + total_ins + total_hoa + total_pmi)

np.random.seed(42)
mc_paths = monte_carlo_home_value(home_value, term_years, n_sims, mc_mean, mc_std)

# Sale values at sale_year_offset
sale_col = min(sale_year_offset, mc_paths.shape[1] - 1)
sale_values = mc_paths[:, sale_col]
selling_costs = sale_values * selling_cost_pct
net_proceeds = sale_values - selling_costs - remaining_balance

# Total cost basis: down payment + all non-equity costs accumulated over time
cost_basis = down_payment + total_non_equity
total_profit = net_proceeds - cost_basis

p10, p25, p50, p75, p90 = np.percentile(total_profit, [10, 25, 50, 75, 90])
prob_profit = (total_profit > 0).mean() * 100

# True Housing Cost Calculation (Total Outflows minus Total Recoverable Inflows over time)
# Equivalent to: (Cost Basis - Median Profit) / Total Months
# Which simplifies purely to: (Down Payment + Total Non-Equity Expenses - (Median Net Proceeds - Down Payment - Total Non-Equity Expenses)) / Months
net_housing_cost_total = cost_basis - p50
estimated_monthly_housing_cost = net_housing_cost_total / total_months


# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1 style="margin:0;font-size:2rem;">🏡 Home Profit Analyzer</h1>
  <p style="margin:0.3rem 0 0;color:#8a9bab;font-size:0.9rem;">
    Total-cost profitability analysis for future home buyers — amortization, dynamic carrying costs, and Monte Carlo sale projections.
  </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  SUMMARY METRICS & EXPANDIBLE BREAKDOWNS
# ──────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

monthly_payment_base = amort_df["payment"].iloc[0] if not amort_df.empty else 0
monthly_total = monthly_payment_base + prop_tax_yr/12 + home_ins_yr/12 + hoa_mo + (loan_amount * pmi_pct / 100 / 12 if dp_pct < 20 else 0)

profit_color = "metric-positive" if p50 >= 0 else "metric-negative"

cards_config = [
    {"label": "Initial Monthly Payment", "value": fmt_dollar(monthly_total), "sub": "P&I + initial costs"},
    {"label": "Total Cost Basis", "value": fmt_dollar(cost_basis), "sub": f"Paid over {sale_year_offset}yr {sale_month_sel}"},
    {"label": "Remaining Balance", "value": fmt_dollar(remaining_balance), "sub": "Loan balance at sale"},
    {"label": "Median Profit (50th %)", "value": fmt_signed(p50), "sub": "At median sale price"},
    {"label": "Prob. of Profit", "value": f"{prob_profit:.0f}%", "sub": "Chance you net profit"}
]

for col, config in zip([col1, col2, col3, col4, col5], cards_config):
    color_cls = profit_color if config["label"] in ["Median Profit (50th %)", "Prob. of Profit"] else ""
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{config["label"]}</div>
      <div class="metric-value {color_cls}">{config["value"]}</div>
      <div class="metric-sub">{config["sub"]}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")
exp_col1, exp_col2, exp_col3 = st.columns([1, 2, 2])
with exp_col2:
    with st.expander("🔍 What is Total Cost Basis?"):
        st.markdown(f"""
        This represents the total out-of-pocket structural cost required to secure, own, and dispose of the home up to month **{total_months}**:
        
        ```
        Cost Basis = Down Payment + Interest Paid + Taxes + Insurance + HOA + PMI
        ```
        - **Down Payment:** {fmt_dollar(down_payment)}
        - **Interest Paid:** {fmt_dollar(total_interest_paid)}
        - **Property Tax (Inflated):** {fmt_dollar(total_prop_tax)} (at {tax_growth_pct}% annual growth)
        - **Insurance (Inflated):** {fmt_dollar(total_ins)} (at {ins_growth_pct}% annual growth)
        - **HOA & PMI:** {fmt_dollar(total_hoa + total_pmi)}
        
        *Principal payments are omitted from the baseline cost basis because they are already accounted for when deducting the remaining loan balance from structural proceeds.*
        """)
with exp_col3:
    with st.expander("🔍 What is Remaining Balance?"):
        st.markdown(f"""
        The principal amount still owed to your
