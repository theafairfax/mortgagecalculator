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
    border-top: 1px solid rgba(180,160,100,0.2);
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
# Generate full yearly dataset reflecting variable growth rates
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
#  SUMMARY METRICS & EXPANSDIBLE BREAKDOWNS
# ──────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

monthly_payment_base = amort_df["payment"].iloc[0] if not amort_df.empty else 0
monthly_total = monthly_payment_base + prop_tax_yr/12 + home_ins_yr/12 + hoa_mo + (loan_amount * pmi_pct / 100 / 12 if dp_pct < 20 else 0)

profit_color = "metric-positive" if p50 >= 0 else "metric-negative"

# Card values configuration
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

# Interactive feedback feature: Detail explanations explicitly positioned below cards
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
        The principal amount still owed to your lending institution at the moment of your intended transaction in **Year {sale_year_offset}**.
        
        - **Original Loan Amount:** {fmt_dollar(loan_amount)}
        - **Principal Paid Off:** {fmt_dollar(total_principal_paid)}
        - **Remaining Payoff Balance:** {fmt_dollar(remaining_balance)}
        
        When selling, this amount is deducted from the gross sale price alongside your **{selling_cost_pct*100:.2f}%** transaction fees ({fmt_dollar(selling_costs.mean())} on average) to compute your raw net proceeds.
        """)

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  TAB LAYOUT
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Yearly Cost Breakdown",
    "📉 Amortization Schedule",
    "🎲 Monte Carlo Sale Analysis",
    "📋 Full Amortization Table",
])


# ──────────────── TAB 1: Yearly Costs ────────────────
with tab1:
    st.markdown("#### Annual Cost Breakdown by Category")
    st.markdown(f"""
    <div class="info-box">
    Stacked bars show each year's total carrying costs split into Principal (equity built),
    Interest, Taxes, Insurance, HOA, and PMI. The dashed line marks your planned sale at
    <b>Year {sale_year_offset}</b>. Notice taxes and insurance compounding over time based on growth assumptions.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    colors = {
        "Principal":       "#5dbf8a",
        "Interest":        "#e06060",
        "Property Tax":    "#d4b870",
        "Home Insurance":  "#7ab3d4",
        "HOA":             "#a07ad4",
        "PMI":             "#d4935a",
    }

    fig1 = go.Figure()
    for cat, clr in colors.items():
        if cat in yearly_df.columns:
            fig1.add_trace(go.Bar(
                name=cat,
                x=yearly_df["Year"],
                y=yearly_df[cat],
                marker_color=clr,
                hovertemplate=f"<b>{cat}</b><br>Year %{{x}}<br>%{{y:$,.0f}}<extra></extra>",
            ))

    fig1.add_vline(x=sale_year_offset, line_dash="dash", line_color="#d4b870",
                   annotation_text=f"Sale: Yr {sale_year_offset}", annotation_font_color="#d4b870")
    fig1.update_layout(
        barmode="stack",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8cab4"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Annual Cost ($)", gridcolor="rgba(255,255,255,0.05)", tickformat="$,.0f"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Pie chart of total cost split up to sale
    st.markdown("#### Cost Composition (Ownership Period)")
    cost_items = {
        "Principal (Equity Built)": total_principal_paid,
        "Interest":                  total_interest_paid,
        "Property Tax":              total_prop_tax,
        "Home Insurance":            total_ins,
        "HOA":                       total_hoa,
        "PMI":                       total_pmi,
    }
    cost_items = {k: v for k, v in cost_items.items() if v > 0}

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig_pie = go.Figure(go.Pie(
            labels=list(cost_items.keys()),
            values=list(cost_items.values()),
            marker_colors=["#5dbf8a","#e06060","#d4b870","#7ab3d4","#a07ad4","#d4935a"],
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:$,.0f}<extra></extra>",
        ))
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8cab4"), showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10), height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("**Cost Totals (to sale)**")
        for k, v in cost_items.items():
            pct = v / sum(cost_items.values()) * 100
            st.markdown(f"<div style='display:flex;justify-content:space-between;margin:0.3rem 0;"
                        f"font-size:0.85rem;'><span style='color:#c8cab4'>{k}</span>"
                        f"<span style='color:#d4b870'>{fmt_dollar(v)} <span style='color:#6a7b8b'>({pct:.1f}%)</span></span></div>",
                        unsafe_allow_html=True)
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        total_all = sum(cost_items.values())
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:0.9rem;'>"
                    f"<b style='color:#d4b870'>Total Spent</b>"
                    f"<b style='color:#d4b870'>{fmt_dollar(total_all)}</b></div>", unsafe_allow_html=True)
        equity = total_principal_paid
        non_eq = total_all - equity
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:0.82rem;margin-top:0.3rem;'>"
                    f"<span style='color:#8a9bab'>of which non-recoverable</span>"
                    f"<span style='color:#e06060'>{fmt_dollar(non_eq)}</span></div>", unsafe_allow_html=True)


# ──────────────── TAB 2: Amortization ────────────────
with tab2:
    st.markdown("#### Loan Balance & Running Cost Over Time")

    fig2 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Remaining Loan Balance", "Cumulative Interest vs Principal Paid"),
    )

    # Balance line
    fig2.add_trace(go.Scatter(
        x=amort_df["month"], y=amort_df["balance"],
        mode="lines", name="Remaining Balance",
        line=dict(color="#7ab3d4", width=2.5),
        fill="tozeroy", fillcolor="rgba(122,179,212,0.1)",
        hovertemplate="Month %{x}<br>Balance: %{y:$,.0f}<extra></extra>",
    ), row=1, col=1)

    # Cumulative interest vs principal
    fig2.add_trace(go.Scatter(
        x=amort_df["month"], y=amort_df["interest"].cumsum(),
        mode="lines", name="Cumulative Interest",
        line=dict(color="#e06060", width=2),
        fill="tozeroy", fillcolor="rgba(224,96,96,0.1)",
        hovertemplate="Month %{x}<br>Cum. Interest: %{y:$,.0f}<extra></extra>",
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=amort_df["month"], y=amort_df["principal"].cumsum(),
        mode="lines", name="Cumulative Principal",
        line=dict(color="#5dbf8a", width=2),
        fill="tozeroy", fillcolor="rgba(93,191,138,0.1)",
        hovertemplate="Month %{x}<br>Cum. Principal: %{y:$,.0f}<extra></extra>",
    ), row=2, col=1)

    # Sale marker
    for r in [1, 2]:
        fig2.add_vline(x=total_months, line_dash="dot", line_color="#d4b870",
                       annotation_text="Sale", annotation_font_color="#d4b870", row=r, col=1)

    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8cab4"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10), height=520,
    )
    fig2.update_yaxes(gridcolor="rgba(255,255,255,0.05)", tickformat="$,.0f")
    fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)", title_text="Month", row=2, col=1)
    st.plotly_chart(fig2, use_container_width=True)


# ──────────────── TAB 3: Monte Carlo ────────────────
with tab3:
    st.markdown("#### Monte Carlo Home Value Simulation")
    st.markdown(f"""
    <div class="info-box">
    {n_sims:,} simulated price paths using a log-normal model with {mc_mean*100:.1f}% annual appreciation
    and {mc_std*100:.1f}% volatility. The shaded band shows the 25th–75th percentile range.
    At sale (Year {sale_year_offset}), after paying off remaining balance ({fmt_dollar(remaining_balance)})
    and selling costs ({selling_cost_pct*100:.1f}%), profit is calculated vs your total cost basis.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    years_x = np.arange(0, term_years + 1)

    # Sub-sample paths for display
    display_n = min(200, n_sims)
    rng_idx = np.random.choice(n_sims, display_n, replace=False)

    fig_mc = go.Figure()

    # Fan of paths
    for i in rng_idx:
        fig_mc.add_trace(go.Scatter(
            x=years_x, y=mc_paths[i],
            mode="lines", line=dict(color="rgba(122,179,212,0.07)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Percentile bands
    p25_path = np.percentile(mc_paths, 25, axis=0)
    p75_path = np.percentile(mc_paths, 75, axis=0)
    p10_path = np.percentile(mc_paths, 10, axis=0)
    p90_path = np.percentile(mc_paths, 90, axis=0)
    p50_path = np.percentile(mc_paths, 50, axis=0)

    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([years_x, years_x[::-1]]),
        y=np.concatenate([p90_path, p10_path[::-1]]),
        fill="toself", fillcolor="rgba(93,191,138,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="10–90th %ile", hoverinfo="skip",
    ))
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([years_x, years_x[::-1]]),
        y=np.concatenate([p75_path, p25_path[::-1]]),
        fill="toself", fillcolor="rgba(93,191,138,0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75th %ile", hoverinfo="skip",
    ))
    fig_mc.add_trace(go.Scatter(
        x=years_x, y=p50_path,
        mode="lines", line=dict(color="#5dbf8a", width=3),
        name="Median Path",
        hovertemplate="Year %{x}<br>Median Value: %{y:$,.0f}<extra></extra>",
    ))

    # Sale marker
    fig_mc.add_vline(x=sale_year_offset, line_dash="dash", line_color="#d4b870",
                     annotation_text=f"Planned Sale (Yr {sale_year_offset})",
                     annotation_font_color="#d4b870")

    # Purchase price line
    fig_mc.add_hline(y=home_value, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                     annotation_text=f"Purchase: {fmt_dollar(home_value)}",
                     annotation_font_color="rgba(255,255,255,0.4)")

    fig_mc.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8cab4"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Years After Purchase", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Home Value ($)", gridcolor="rgba(255,255,255,0.05)", tickformat="$,.0f"),
        margin=dict(l=10, r=10, t=10, b=10), height=420,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # ── Profit Distribution ──
    st.markdown("#### Total Profit / Deficit Distribution at Sale")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=total_profit,
        nbinsx=60,
        marker=dict(
            color=np.where(total_profit >= 0, "#5dbf8a", "#e06060"),
            line=dict(color="rgba(0,0,0,0.3)", width=0.5),
        ),
        name="Simulation Outcomes",
        hovertemplate="Profit: %{x:$,.0f}<br>Count: %{y}<extra></extra>",
    ))

    # Percentile markers
    for pv, label, clr in [
        (p10, "P10", "#e06060"), (p25, "P25", "#d4935a"),
        (p50, "P50", "#d4b870"), (p75, "P75", "#7ab3d4"), (p90, "P90", "#5dbf8a"),
    ]:
        fig_hist.add_vline(x=pv, line_dash="dot", line_color=clr,
                           annotation_text=f"{label}: {fmt_signed(pv)}",
                           annotation_font_color=clr, annotation_font_size=11)

    fig_hist.add_vline(x=0, line_color="white", line_width=1.5)

    fig_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8cab4"),
        xaxis=dict(title="Total Profit / Deficit ($)", gridcolor="rgba(255,255,255,0.05)", tickformat="$,.0f"),
        yaxis=dict(title="Number of Simulations", gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10), height=340,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Profit percentile table ──
    st.markdown("#### Profit Scenario Summary")
    pctiles = [5, 10, 25, 50, 75, 90, 95]
    pctile_vals = np.percentile(total_profit, pctiles)
    sale_vals_at_p = np.percentile(sale_values, pctiles)

    tbl_data = []
    for pct, prof, sv in zip(pctiles, pctile_vals, sale_vals_at_p):
        tbl_data.append({
            "Scenario (Percentile)": f"P{pct}",
            "Sale Price": fmt_dollar(sv),
            "Selling Costs": fmt_dollar(sv * selling_cost_pct),
            "Remaining Balance": fmt_dollar(remaining_balance),
            "Net Proceeds": fmt_dollar(sv - sv * selling_cost_pct - remaining_balance),
            "Total Cost Basis": fmt_dollar(cost_basis),
            "Total Profit / Deficit": fmt_signed(prof),
        })

    tbl_df = pd.DataFrame(tbl_data)

    def color_profit(val):
        try:
            if val.startswith("+"):
                return "color: #5dbf8a"
            elif val.startswith("-"):
                return "color: #e06060"
        except:
            pass
        return ""

    styled = tbl_df.style.map(color_profit, subset=["Total Profit / Deficit"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ──────────────── TAB 4: Full Table ────────────────
with tab4:
    st.markdown("#### Complete Monthly Amortization Schedule")

    # Build display-ready table
    display_df = amort_df.copy()
    start_mo_idx = MONTHS.index(start_month)
    dates = []
    yr, mo = start_year, start_mo_idx
    for _ in range(len(display_df)):
        dates.append(f"{MONTHS[mo]} {yr}")
        mo += 1
        if mo == 12:
            mo = 0
            yr += 1
    display_df.insert(0, "Date", dates)
    display_df["Cumulative Interest"] = display_df["interest"].cumsum()
    display_df["Cumulative Principal"] = display_df["principal"].cumsum()

    display_df = display_df.rename(columns={
        "month": "Month", "payment": "Payment", "interest": "Interest",
        "principal": "Principal", "balance": "Balance",
    })

    dollar_cols = ["Payment","Interest","Principal","Balance","Cumulative Interest","Cumulative Principal"]
    fmt_dict = {c: "${:,.2f}" for c in dollar_cols}

    st.dataframe(
        display_df[["Month","Date","Payment","Interest","Principal",
                     "Cumulative Interest","Cumulative Principal","Balance"]]
        .style.format(fmt_dict),
        use_container_width=True,
        height=480,
    )

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="amortization_schedule.csv",
        mime="text/csv",
    )
