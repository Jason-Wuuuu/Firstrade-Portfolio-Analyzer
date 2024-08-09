# Firstrade Portfolio Analysis

This project contains Jupyter notebooks for analyzing and visualizing your Firstrade investment portfolio.

## Notebooks

1. **Firstrade_History.ipynb**: 
   This notebook recreates all transactions from your Firstrade account history. It allows you to view your portfolio composition on any given day. The notebook processes the transaction data, calculates realized gains, and provides a detailed view of your investment history.

2. **Firstrade_Performance.ipynb**: 
   This notebook focuses on analyzing the performance of your current portfolio. It calculates total costs, market values, and profit/loss metrics. Additionally, it generates visualizations of your portfolio's performance, including individual stock charts and overall portfolio returns.

## Setup

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Download your transaction history from Firstrade and rename it to `FT_History.csv`. Place this file in the same directory as the notebooks.

## Usage

1. Open the desired notebook in Jupyter Lab or Jupyter Notebook.
2. Run the cells in order to process your data and generate insights.
3. Customize the date ranges or specific analyses as needed within the notebooks.

## Features

- Transaction history processing
- Portfolio composition at any given date
- Realized and unrealized gain calculations
- Performance visualizations
- Comparison with market indices (e.g., S&P 500)

## Note

Ensure that your Firstrade history file (`FT_History.csv`) is up to date for the most accurate analysis. The notebooks use this file as the primary data source for all calculations and visualizations.
