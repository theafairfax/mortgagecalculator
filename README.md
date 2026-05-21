# 🏡 Home Profit Analyzer

A total-cost profitability analysis tool for future home buyers, built with Streamlit.

## Features

- **Full mortgage calculator** — all standard inputs (home value, down payment, rate, term, loan type)
- **Carrying cost breakdown** — property tax, PMI, home insurance, HOA included monthly and annually
- **Amortization schedule** — full month-by-month table with CSV download
- **Interactive charts** — yearly stacked cost bars, balance curve, interest vs principal area chart
- **Monte Carlo sale analysis** — simulate 200–2,000 price paths at user-defined appreciation/volatility
- **Profit/deficit distribution** — histogram + percentile table (P5–P95) for every sale scenario
- **Sale date selector** — pick any year+month offset to see costs and projected profit at that point

## Deploy to Streamlit Community Cloud

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, and file `app.py`
4. Click **Deploy** — Streamlit will install `requirements.txt` automatically

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
.
├── app.py            # Main application
├── requirements.txt  # Python dependencies
└── README.md
```

## How Profit Is Calculated

```
Profit = Net Proceeds − Cost Basis

Net Proceeds  = Sale Price − Selling Costs (%) − Remaining Loan Balance
Cost Basis    = Down Payment + Interest Paid + Property Tax + Insurance + HOA + PMI
```

Principal payments are **not** counted in the cost basis because they directly reduce the
remaining balance deducted from proceeds — they are accounted for exactly once.

## Monte Carlo Model

Home values are simulated using geometric Brownian motion (log-normal returns):

- **Drift**: user-set annual appreciation rate (default 4%)  
- **Volatility**: user-set annual standard deviation (default 8%)  
- Monthly log-returns are drawn independently and compounded  
- The distribution of sale prices at the chosen sale year drives the profit histogram
