import json
import datetime
from decimal import Decimal
from collections import defaultdict
import yfinance as yf
import logging
import os
import time
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class PortfolioHistory:
    """
    Manages and processes the portfolio transaction history, calculates portfolio states,
    and provides methods for viewing and saving portfolio data.
    """

    def __init__(self):
        self.transaction_history = {}
        self.portfolio_state = {}
        self.filled_portfolio_state = {}
        self.historical_data = None
        self.sectors = {}

    def process_transaction_history(self, input_file='transaction_history.json', save_output=False, output_file='portfolio_history.json'):
        """
        Processes the transaction history from start to finish, including loading transactions,
        calculating portfolio states, filling missing dates, fetching historical data,
        and updating the portfolio with market prices.

        :param input_file: Path to the input JSON file containing transaction history
        :param save_output: Boolean flag to determine if output should be saved
        :param output_file: Path to save the output JSON file if save_output is True
        :return: The filled portfolio state dictionary
        """
        try:
            self.load_transactions(input_file)
            if not self.transaction_history:
                raise ValueError(
                    "No transactions were loaded from the JSON file.")

            self.calculate_portfolio_state()
            if not self.portfolio_state:
                raise ValueError("No portfolio state was calculated.")

            self.fill_missing_dates()
            self.fetch_historical_data()
            self.update_portfolio_with_market_prices()

            if save_output:
                self.save_to_json(output_file)

            logging.info(f"Successfully processed transaction history with {
                         len(self.transaction_history)} dates.")
            return self.filled_portfolio_state
        except Exception as e:
            logging.error(f"Error processing transaction history: {str(e)}")
            raise

    def load_transactions(self, file_path):
        """
        Loads transactions from a JSON file.

        :param file_path: Path to the JSON file containing transaction history
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        try:
            with open(file_path, 'r') as file:
                self.transaction_history = json.load(file)

            if not self.transaction_history:
                raise ValueError("The JSON file is empty")

            logging.info(f"Successfully loaded transactions from {file_path}")
        except json.JSONDecodeError:
            raise ValueError(
                "Unable to parse the JSON file. Please check the file format.")
        except Exception as e:
            logging.error(f"Error loading transactions from JSON: {str(e)}")
            raise

    def calculate_portfolio_state(self):
        """
        Calculates the portfolio state based on the loaded transaction history,
        processing deposits, interests, dividends, splits, buys, sells, and reinvestments.
        """
        portfolio_state = {}
        cash = Decimal('0')
        holdings = defaultdict(
            lambda: {'quantity': Decimal('0'), 'total_cost': Decimal('0')})

        for date, transactions in self.transaction_history.items():
            # Update cash
            cash += Decimal(str(transactions['deposit'])) + Decimal(
                str(transactions['interest'])) + Decimal(str(transactions['dividend']))

            # Process splits
            for symbol, split_data in transactions['split'].items():
                if symbol in holdings:
                    old_quantity = holdings[symbol]['quantity']
                    new_quantity = old_quantity + \
                        Decimal(str(split_data['quantity']))
                    holdings[symbol]['quantity'] = new_quantity
                    # total_cost remains unchanged after a split

            # Process buys
            for symbol, buys in transactions['buy'].items():
                for buy in buys:
                    quantity = Decimal(str(buy['quantity']))
                    cost = abs(Decimal(str(buy['amount'])))
                    holdings[symbol]['quantity'] += quantity
                    holdings[symbol]['total_cost'] += cost
                    cash -= cost

            # Process sells
            for symbol, sells in transactions['sell'].items():
                for sell in sells:
                    sell_quantity = abs(Decimal(str(sell['quantity'])))
                    sell_amount = Decimal(str(sell['amount']))
                    if holdings[symbol]['quantity'] > 0:
                        current_quantity = holdings[symbol]['quantity']
                        current_total_cost = holdings[symbol]['total_cost']
                        avg_cost_per_share = current_total_cost / current_quantity

                        cost_basis_sold = avg_cost_per_share * sell_quantity

                        holdings[symbol]['quantity'] -= sell_quantity
                        holdings[symbol]['total_cost'] -= cost_basis_sold

                        cash += sell_amount

            # Process reinvestments
            for symbol, reinvests in transactions['reinvestment'].items():
                for reinvest in reinvests:
                    quantity = Decimal(str(reinvest['quantity']))
                    cost = abs(Decimal(str(reinvest['amount'])))
                    holdings[symbol]['quantity'] += quantity
                    holdings[symbol]['total_cost'] += cost

            # Create the portfolio state for this date
            current_holdings = {}
            for symbol, data in holdings.items():
                if data['quantity'] > 0:
                    current_holdings[symbol] = {
                        'quantity': float(data['quantity']),
                        'total_cost': float(data['total_cost']),
                        'unit_cost': float(data['total_cost'] / data['quantity']) if data['quantity'] > 0 else 0
                    }

            portfolio_state[date] = {
                'cash': float(cash),
                'holdings': current_holdings
            }

        self.portfolio_state = portfolio_state

    def fill_missing_dates(self):
        """
        Fills in missing dates in the portfolio state with the last known state,
        ensuring a continuous daily record from the first transaction to the current date.
        """
        dates = sorted(self.portfolio_state.keys())
        start_date = datetime.datetime.strptime(dates[0], '%Y-%m-%d').date()
        end_date = datetime.date.today()

        filled_portfolio_state = {}
        last_state = None

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            if date_str in self.portfolio_state:
                filled_portfolio_state[date_str] = self.portfolio_state[date_str]
                last_state = self.portfolio_state[date_str]
            elif last_state is not None:
                # Copy only quantity, total_cost, and unit_cost
                filled_portfolio_state[date_str] = {
                    'cash': last_state['cash'],
                    'holdings': {
                        symbol: {
                            'quantity': data['quantity'],
                            'total_cost': data['total_cost'],
                            'unit_cost': data['unit_cost']
                        }
                        for symbol, data in last_state['holdings'].items()
                    }
                }

            current_date += datetime.timedelta(days=1)

        self.filled_portfolio_state = filled_portfolio_state

    def fetch_historical_data(self):
        """
        Fetches historical price data and sector information for all symbols in the portfolio using yfinance.
        """
        try:
            first_date = min(self.filled_portfolio_state.keys())
            last_date = max(self.filled_portfolio_state.keys())
            start_date = (datetime.datetime.strptime(
                first_date, '%Y-%m-%d') - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
            end_date = (datetime.datetime.strptime(
                last_date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

            symbols_list = self.get_unique_symbols()
            if not symbols_list:
                raise ValueError("No symbols found in the portfolio.")

            # Split symbols into batches of 10
            symbol_batches = [symbols_list[i:i+10]
                              for i in range(0, len(symbols_list), 10)]

            all_data = []
            sectors = {}
            for batch in symbol_batches:
                batch_data = yf.download(
                    batch, start=start_date, end=end_date, group_by='ticker')
                all_data.append(batch_data)

                # Fetch sector information for each symbol in the batch
                for symbol in batch:
                    ticker = yf.Ticker(symbol)
                    # Change 'Unknown' to 'ETF'
                    sector = ticker.info.get('sector', 'ETF')
                    sectors[symbol] = sector

                time.sleep(.5)  # Add delay between batches

            self.historical_data = pd.concat(all_data, axis=1)
            if self.historical_data.empty:
                raise ValueError("Failed to fetch historical data.")

            self.historical_data = self.historical_data.loc[:, (slice(
                None), 'Adj Close')]
            self.historical_data.columns = self.historical_data.columns.droplevel(
                1)

            # Store sector information
            self.sectors = sectors

            logging.info(f"Successfully fetched historical data and sector information for {
                         len(symbols_list)} symbols.")
        except Exception as e:
            logging.error(
                f"Error fetching historical data and sector information: {str(e)}")
            raise

    def get_unique_symbols(self):
        """
        Returns a sorted list of unique symbols present in the portfolio history.

        :return: List of unique symbols
        """
        unique_symbols = set()
        for date_data in self.filled_portfolio_state.values():
            unique_symbols.update(date_data['holdings'].keys())
        return sorted(unique_symbols)

    def update_portfolio_with_market_prices(self):
        """
        Updates the portfolio state with market prices from the fetched historical data,
        adjusting for splits and removing dates where market data is not available.
        """
        dates_to_remove = []
        split_adjustments = {}

        # First, collect all split information
        for date, transactions in self.transaction_history.items():
            for symbol, split_data in transactions['split'].items():
                additional_quantity = Decimal(str(split_data['quantity']))
                if additional_quantity != 0:
                    # Find the holding quantity just before the split
                    split_date = datetime.datetime.strptime(date, '%Y-%m-%d')
                    pre_split_date = (
                        split_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                    if pre_split_date in self.filled_portfolio_state:
                        original_quantity = Decimal(
                            str(self.filled_portfolio_state[pre_split_date]['holdings'][symbol]['quantity']))
                        new_quantity = original_quantity + additional_quantity
                        split_ratio = new_quantity / original_quantity
                        print(f"Split detected for {symbol} on {
                              date}. Ratio: {split_ratio:.4f}")
                        split_adjustments[symbol] = {
                            'date': split_date,
                            'ratio': split_ratio
                        }

        for date, state in self.filled_portfolio_state.items():
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
            if date_obj not in self.historical_data.index:
                dates_to_remove.append(date)
                continue

            for symbol in state['holdings']:
                if symbol in self.historical_data.columns:
                    market_price = Decimal(
                        str(self.historical_data.at[date_obj, symbol]))

                    # Adjust market price for splits
                    if symbol in split_adjustments and date_obj < split_adjustments[symbol]['date']:
                        # Multiply by ratio to adjust pre-split prices
                        market_price *= split_adjustments[symbol]['ratio']

                    state['holdings'][symbol]['market_price'] = float(
                        market_price)
                else:
                    state['holdings'][symbol]['market_price'] = None

        for date in dates_to_remove:
            del self.filled_portfolio_state[date]

    def view_portfolio_on_date(self, date):
        """
        Returns the portfolio state for a specific date.

        :param date: Date string in 'YYYY-MM-DD' format
        :return: Portfolio state on the given date or a message if no data is available
        """
        try:
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'")

        if date_str in self.filled_portfolio_state:
            return {date_str: self.filled_portfolio_state[date_str]}
        else:
            return "No portfolio data available for this date."

    def pretty_print_portfolio(self, portfolio_data):
        """
        Prints a formatted JSON representation of the portfolio data.

        :param portfolio_data: Portfolio data to be printed
        """
        def default_serializer(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        print(json.dumps(portfolio_data, indent=4, default=default_serializer))

    def save_to_json(self, file_path):
        """
        Saves the filled portfolio state and sectors to a JSON file.

        :param file_path: Path to save the JSON file
        """
        try:
            output_data = {
                "sectors": self.sectors,
                "portfolios": self.filled_portfolio_state
            }

            with open(file_path, 'w') as file:
                json.dump(output_data, file, indent=2, default=str)
            logging.info(f"Portfolio history and sectors saved to {file_path}")
        except IOError as e:
            logging.error(f"Unable to write to file {file_path}: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        portfolio_history = PortfolioHistory()

        portfolio_history.process_transaction_history(
            save_output=True, output_file='portfolio_history.json')

        # Example usage of new functions
        date_to_view = max(portfolio_history.filled_portfolio_state.keys())
        portfolio_on_date = portfolio_history.view_portfolio_on_date(
            date_to_view)
        portfolio_history.pretty_print_portfolio(portfolio_on_date)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
