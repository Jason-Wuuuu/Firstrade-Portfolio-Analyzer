# Firstrade Portfolio Analysis

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Files](#files)
- [Setup](#setup)
- [Usage](#usage)
- [Features](#features)
- [Visualizations](#visualizations)
- [Example Output](#example-output)
- [Note](#note)
- [Appendix: Output File Structures](#appendix-output-file-structures)

## Introduction

This project offers a comprehensive suite of Python scripts and a Jupyter notebook for in-depth analysis and visualization of your Firstrade investment portfolio. By meticulously reconstructing your complete transaction history from the Firstrade CSV file, it enables:

- Precise tracking of your portfolio's evolution from the very first trade
- Detailed performance analysis over any time period
- Accurate calculation of returns, including the compounding effects of dividends and reinvestments
- Proper handling of corporate actions like stock splits
- Insights into how different transaction types have impacted your overall performance

## Quick Start

1. Install requirements: `pip install -r requirements.txt`
2. Place your Firstrade CSV file as `FT_History.csv` in the project directory
3. Run `python ft_history_processor.py`
4. Run `python transaction_history_processor.py`
5. Open `portfolio_analysis_dashboard.ipynb` in Jupyter

## Files

1. **portfolio_analysis_dashboard.ipynb**:
   This Jupyter notebook provides an interactive dashboard for analyzing your portfolio. It visualizes portfolio composition, performance metrics, and historical trends using interactive Plotly charts.

2. **transaction_history_processor.py**:
   This Python script processes the transaction history from Firstrade, calculates portfolio states, and provides methods for viewing and saving portfolio data.

3. **ft_history_processor.py**:
   This script handles the initial processing of the Firstrade CSV history file, converting it into a structured JSON format for further analysis.

## Setup

1. Install required packages:

   ```
   pip install -r requirements.txt
   ```

2. Download your transaction history from Firstrade as a CSV file and rename it to `FT_History.csv`. Place this file in the same directory as the scripts.

## Usage

1. Run the `ft_history_processor.py` script to convert the CSV file to a JSON format:

   ```
   python ft_history_processor.py
   ```

2. Run the `transaction_history_processor.py` script to process the JSON data and calculate portfolio states:

   ```
   python transaction_history_processor.py
   ```

3. Open the `portfolio_analysis_dashboard.ipynb` notebook in Jupyter Lab or Jupyter Notebook to interact with the dashboard and analyze your portfolio.

## Features

- Transaction history processing from CSV to JSON
- Portfolio state calculation and historical tracking
- Integration with yfinance for fetching market data
- Interactive Plotly visualizations of portfolio composition and performance
- Analysis of individual stock performance within the portfolio
- Comparison of portfolio performance against S&P 500 benchmark
- Generation of HTML reports for portfolio analysis

## Visualizations

The dashboard features interactive Plotly charts, allowing for:

- Stock proportions by value
- Sector proportions
- Portfolio performance vs S&P 500
- Zooming and panning on all charts
- Hovering over data points for detailed information
- Comparing portfolio performance against S&P 500 benchmark
- Adjustable date ranges for focused analysis
- Sector breakdown with detailed stock information on hover

Sample visualizations:

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="./outputs/output1.png" alt="Portfolio Performance vs S&P 500" style="max-width: 100%; height: auto; margin-bottom: 20px;">
  <img src="./outputs/output2.png" alt="Individual Stock Performance Comparison" style="max-width: 100%; height: auto; margin-bottom: 20px;">
  <img src="./outputs/output3.png" alt="Individual Stock Performance Comparison" style="max-width: 100%; height: auto;">
</div>

These visualizations are examples of the interactive charts generated by the Jupyter notebook. They provide insights into overall portfolio performance and individual stock contributions.

## Example Output

The notebook generates various visualizations and tables, including:

- Stock proportions pie chart
- Sector proportions pie chart
- Portfolio performance vs S&P 500 line chart
- Detailed portfolio holdings table
- Closed positions table
- Portfolio summary statistics

| Symbol | Quantity | Unit Cost | Market Price | Total Cost | Market Value | Unrealized G/L   | Daily Gain     |
| ------ | -------- | --------- | ------------ | ---------- | ------------ | ---------------- | -------------- |
| AAPL   | 10.00    | $150.00   | $175.00      | $1,500.00  | $1,750.00    | $250.00 (16.67%) | $25.00 (1.43%) |
| GOOGL  | 5.00     | $2,000.00 | $2,100.00    | $10,000.00 | $10,500.00   | $500.00 (5.00%)  | $50.00 (0.48%) |
| MSFT   | 15.00    | $200.00   | $220.00      | $3,000.00  | $3,300.00    | $300.00 (10.00%) | $30.00 (0.91%) |

### Portfolio Summary:

- Cash: $5,000.00
- Total Market Value: $20,550.00
- Invested Value: $19,500.00
- Unrealized Gain/Loss: $1,050.00 (5.38%)
- Daily Gain: $105.00 (0.51%)

### Closed Positions

| Symbol | Quantity | Sell Price | Cost Basis | Realized Gain    |
| ------ | -------- | ---------- | ---------- | ---------------- |
| TSLA   | 2.00     | $800.00    | $700.00    | $200.00 (14.29%) |
| AMZN   | 1.00     | $3,200.00  | $3,000.00  | $200.00 (6.67%)  |

Total Realized Gains: $400.00 (10.81%)

## Note

Ensure that your Firstrade history file (`FT_History.csv`) is up to date for the most accurate analysis. The scripts and notebook use this file as the primary data source for all calculations and visualizations.

## Appendix: Output File Structures

The scripts generate two main output files:

- `transaction_history.json`: Processed transaction data
- `portfolio_history.json`: Calculated portfolio states with market data

### transaction_history.json Structure

The `transaction_history.json` file contains a detailed record of all transactions processed from the Firstrade CSV file. Here's an overview of its structure:

```json
{
  "2023-05-15": {
    "buy": {
      "AAPL": [
        {
          "quantity": 10,
          "price": 150.0,
          "amount": 1500.0
        }
      ]
    },
    "sell": {
      "GOOGL": [
        {
          "quantity": 5,
          "price": 2500.0,
          "amount": 12500.0
        }
      ]
    },
    "reinvestment": {
      "SPY": [
        {
          "quantity": 1.5,
          "price": 400.0,
          "amount": 600.0
        }
      ]
    },
    "split": {
      "TSLA": {
        "quantity": 2
      }
    },
    "deposit": 5000.0,
    "interest": 10.5,
    "dividend": 100.0
  },
  "2023-05-16": {
    // Similar structure for another date
  }
  // ... more dates
}
```

### portfolio_history.json Structure

The `portfolio_history.json` file contains a detailed record of your portfolio's state over time. Here's an overview of its structure:

```json
{
  "version": "1.0",
  "timestamp": "2023-05-15T12:34:56.789012",
  "sectors": {
    "AAPL": "Technology",
    "GOOGL": "Communication Services",
    "SPY": "ETF"
    // ... other symbols and their sectors
  },
  "portfolios": {
    "2023-05-15": {
      "summary": {
        "total_market_value": "100000.00",
        "cash": "5000.00",
        "invested_value": "95000.00",
        "total_cost_basis": "90000.00",
        "unrealized_gain_loss": "5000.00",
        "unrealized_gain_loss_percentage": "5.56",
        "daily_gain": "500.00",
        "daily_return": "0.0050",
        "total_deposits": "95000.00"
      },
      "holdings": {
        "AAPL": {
          "quantity": "100.00",
          "total_cost": "15000.00",
          "unit_cost": "150.00",
          "market_price": "160.00",
          "market_value": "16000.00",
          "unrealized_gain_loss": "1000.00",
          "unrealized_gain_loss_percentage": "6.67",
          "daily_gain": "200.00",
          "daily_return": "0.0125"
        }
        // ... other holdings
      },
      "closed_positions": {
        "GOOGL": [
          {
            "quantity": "10.00",
            "sell_price": "2500.00",
            "cost_basis": "2400.00",
            "realized_gain": "1000.00"
          }
        ]
        // ... other closed positions
      }
    }
    // ... portfolio data for other dates
  }
}
```
