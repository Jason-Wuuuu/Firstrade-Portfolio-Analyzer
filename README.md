# Firstrade Portfolio Analysis

This project offers a comprehensive suite of Python scripts and a Jupyter notebook for in-depth analysis and visualization of your Firstrade investment portfolio. By meticulously reconstructing your complete transaction history from the Firstrade CSV file, it enables:

- Precise tracking of your portfolio's evolution from the very first trade
- Detailed performance analysis over any time period
- Accurate calculation of returns, including the compounding effects of dividends and reinvestments
- Proper handling of corporate actions like stock splits
- Insights into how different transaction types have impacted your overall performance

## Files

1. **portfolio_analysis_dashboard.ipynb**:
   This Jupyter notebook provides an interactive dashboard for analyzing your portfolio. It visualizes portfolio composition, performance metrics, and historical trends using interactive Plotly charts.

2. **transaction_history_processor.py**:
   This Python script processes the transaction history from Firstrade, calculates portfolio states, and provides methods for viewing and saving portfolio data.

3. **ft_history_processor.py**:
   This script handles the initial processing of the Firstrade CSV history file, converting it into a structured JSON format for further analysis.

## Interactive Visualizations

The dashboard now features interactive Plotly charts, allowing for:

- Zooming and panning on all charts
- Hovering over data points for detailed information
- Comparing portfolio performance against S&P 500 benchmark
- Adjustable date ranges for focused analysis
- Sector breakdown with detailed stock information on hover

## Outputs

<div style="display: flex; justify-content: space-between;">
  <img src="./outputs/output1.png" alt="Portfolio Performance" style="max-height: 400px; width: auto;">
  <img src="./outputs/output2.png" alt="Stock Comparison" style="max-height: 400px; width: auto;">
</div>

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

## Outputs

The scripts generate two main output files:

- `transaction_history.json`: Processed transaction data
- `portfolio_history.json`: Calculated portfolio states with market data

The Jupyter notebook provides various interactive charts and tables for comprehensive portfolio analysis.

## Note

Ensure that your Firstrade history file (`FT_History.csv`) is up to date for the most accurate analysis. The scripts and notebook use this file as the primary data source for all calculations and visualizations.
