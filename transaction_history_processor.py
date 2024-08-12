import json
import datetime
from decimal import Decimal, getcontext
from collections import defaultdict
import yfinance as yf
import logging
import os
import time
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass

# Set decimal precision
getcontext().prec = 10

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class Holding:
    """
    Represents a single holding in the portfolio.

    Attributes:
        quantity (Decimal): The number of shares held.
        total_cost (Decimal): The total cost of the holding.
    """
    quantity: Decimal
    total_cost: Decimal


class DecimalEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for Decimal and datetime objects.
    """

    def default(self, obj):
        """
        Convert Decimal and datetime objects to JSON-serializable formats.

        Args:
            obj: The object to be serialized.

        Returns:
            A JSON-serializable representation of the object.
        """
        if isinstance(obj, Decimal):
            return str(obj)  # Store as string instead of float
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super(DecimalEncoder, self).default(obj)


def decimal_decoder(dct):
    for k, v in dct.items():
        if isinstance(v, str):
            try:
                dct[k] = Decimal(v)
            except:
                pass
    return dct


class PortfolioHistory:
    """
    Manages and processes the history of a portfolio, including transactions and market data.
    """

    def __init__(self):
        """Initialize the PortfolioHistory object with essential data structures."""
        self.transaction_history: Dict[str, Any] = {}
        self.portfolio_state: Dict[str, Any] = {}
        self.historical_data: pd.DataFrame = None
        self.sectors: Dict[str, str] = {}
        self.split_adjusted_prices: Dict[str,
                                         Dict[datetime.datetime, Decimal]] = {}

    def process_transaction_history(self, input_file: str = 'transaction_history.json',
                                    save_output: bool = False,
                                    output_file: str = 'portfolio_history.json') -> Dict[str, Any]:
        """
        Process the transaction history from a JSON file and calculate portfolio states.

        Args:
            input_file (str): Path to the input JSON file containing transaction history.
            save_output (bool): Whether to save the processed data to a JSON file.
            output_file (str): Path to the output JSON file if saving is enabled.

        Returns:
            Dict[str, Any]: The processed portfolio state history.
        """
        try:
            self.load_transactions(input_file)
            self.calculate_portfolio_state()
            self.fill_missing_dates()
            self.fetch_historical_data()
            self.update_portfolio_with_market_prices()

            if save_output:
                self.save_to_json(output_file)

            logging.info(f"Successfully processed transaction history with {
                         len(self.transaction_history)} dates.")
            return self.portfolio_state
        except Exception as e:
            logging.error(f"Error processing transaction history: {str(e)}")
            raise

    def load_transactions(self, file_path: str):
        """
        Load transaction data from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing transaction data.

        Raises:
            FileNotFoundError: If the specified file is not found.
            ValueError: If the JSON file is empty or cannot be parsed.
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
        Calculate the portfolio state for each date based on the transaction history.
        Updates the portfolio_state attribute with the calculated states.
        """
        portfolio_state = {}
        cash = Decimal('0')
        holdings: Dict[str, Holding] = defaultdict(
            lambda: Holding(Decimal('0'), Decimal('0')))
        cumulative_realized_gains = Decimal('0')
        cumulative_deposits = Decimal('0')
        closed_positions = {}

        for date, transactions in self.transaction_history.items():
            cash = self._update_cash(cash, transactions)
            holdings = self._process_splits(holdings, transactions['split'])
            cash, holdings = self._process_buys(
                cash, holdings, transactions['buy'])
            cash, holdings, realized_gains, new_closed_positions = self._process_sells(
                cash, holdings, transactions['sell'])
            holdings = self._process_reinvestments(
                holdings, transactions['reinvestment'])

            cumulative_realized_gains += realized_gains
            cumulative_deposits += Decimal(str(transactions['deposit']))
            closed_positions.update(new_closed_positions)

            portfolio_state[date] = self._create_portfolio_state(
                cash, holdings, cumulative_realized_gains, cumulative_deposits, closed_positions)

        self.portfolio_state = portfolio_state

    def _update_cash(self, cash: Decimal, transactions: Dict[str, Any]) -> Decimal:
        """
        Update the cash balance based on deposits, interest, and dividends.

        Args:
            cash (Decimal): Current cash balance.
            transactions (Dict[str, Any]): Transaction data for the current date.

        Returns:
            Decimal: Updated cash balance.
        """
        return cash + sum(Decimal(str(transactions[key])) for key in ['deposit', 'interest', 'dividend'])

    def _process_splits(self, holdings: Dict[str, Holding], splits: Dict[str, Any]) -> Dict[str, Holding]:
        """
        Process stock splits and update holdings accordingly.

        Args:
            holdings (Dict[str, Holding]): Current holdings.
            splits (Dict[str, Any]): Split data for the current date.

        Returns:
            Dict[str, Holding]: Updated holdings after processing splits.
        """
        for symbol, split_data in splits.items():
            if symbol in holdings:
                holdings[symbol].quantity += Decimal(
                    str(split_data['quantity']))
        return holdings

    def _process_buys(self, cash: Decimal, holdings: Dict[str, Holding], buys: Dict[str, List[Dict[str, Any]]]) -> tuple:
        """
        Process buy transactions and update cash and holdings.

        Args:
            cash (Decimal): Current cash balance.
            holdings (Dict[str, Holding]): Current holdings.
            buys (Dict[str, List[Dict[str, Any]]]): Buy transaction data.

        Returns:
            tuple: Updated cash balance and holdings.
        """
        for symbol, buy_list in buys.items():
            for buy in buy_list:
                quantity = Decimal(str(buy['quantity']))
                cost = Decimal(str(buy['amount']))
                holdings[symbol].quantity += quantity
                holdings[symbol].total_cost += cost
                cash -= cost
        return cash, holdings

    def _process_sells(self, cash: Decimal, holdings: Dict[str, Holding],
                       sells: Dict[str, List[Dict[str, Any]]]) -> tuple:
        """
        Process sell transactions and update cash, holdings, and realized gains.

        Args:
            cash (Decimal): Current cash balance.
            holdings (Dict[str, Holding]): Current holdings.
            sells (Dict[str, List[Dict[str, Any]]]): Sell transaction data.

        Returns:
            tuple: Updated cash balance, holdings, realized gains, and new closed positions.
        """
        realized_gains = Decimal('0')
        new_closed_positions = {}
        for symbol, sell_list in sells.items():
            for sell in sell_list:
                sell_quantity = Decimal(str(sell['quantity']))
                sell_amount = Decimal(str(sell['amount']))
                sell_price = Decimal(str(sell['price']))

                if holdings[symbol].quantity >= sell_quantity:
                    avg_cost_per_share = holdings[symbol].total_cost / \
                        holdings[symbol].quantity
                    cost_basis_sold = avg_cost_per_share * sell_quantity
                    realized_gain = sell_amount - cost_basis_sold
                    realized_gains += realized_gain

                    holdings[symbol].quantity -= sell_quantity
                    holdings[symbol].total_cost -= cost_basis_sold
                    cash += sell_amount

                    if symbol not in new_closed_positions:
                        new_closed_positions[symbol] = []
                    new_closed_positions[symbol].append({
                        'quantity': sell_quantity,
                        'sell_price': sell_price,
                        'cost_basis': cost_basis_sold,
                        'realized_gain': realized_gain
                    })
                else:
                    # Handle error: trying to sell more shares than owned
                    logging.error(f"Attempted to sell {sell_quantity} shares of {
                                  symbol}, but only {holdings[symbol].quantity} owned.")

        return cash, holdings, realized_gains, new_closed_positions

    def _process_reinvestments(self, holdings: Dict[str, Holding], reinvestments: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Holding]:
        """
        Process reinvestment transactions and update holdings.

        Args:
            holdings (Dict[str, Holding]): Current holdings.
            reinvestments (Dict[str, List[Dict[str, Any]]]): Reinvestment transaction data.

        Returns:
            Dict[str, Holding]: Updated holdings after processing reinvestments.
        """
        for symbol, reinvest_list in reinvestments.items():
            for reinvest in reinvest_list:
                quantity = Decimal(str(reinvest['quantity']))
                cost = Decimal(str(reinvest['amount']))
                holdings[symbol].quantity += quantity
                holdings[symbol].total_cost += cost
        return holdings

    def _create_portfolio_state(self, cash: Decimal, holdings: Dict[str, Holding],
                                cumulative_realized_gains: Decimal, cumulative_deposits: Decimal,
                                closed_positions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Create a portfolio state snapshot for a given point in time.

        Args:
            cash (Decimal): Current cash balance.
            holdings (Dict[str, Holding]): Current holdings.
            cumulative_realized_gains (Decimal): Current cumulative realized gains.
            cumulative_deposits (Decimal): Total deposits made.
            closed_positions (Dict[str, List[Dict[str, Any]]]): Closed positions data.

        Returns:
            Dict[str, Any]: A snapshot of the portfolio state.
        """
        current_holdings = {
            symbol: {
                'quantity': data.quantity,
                'total_cost': data.total_cost,
                'unit_cost': data.total_cost / data.quantity if data.quantity > 0 else 0
            }
            for symbol, data in holdings.items() if data.quantity > 0
        }

        return {
            'summary': {
                'cash': cash,
                'total_deposits': cumulative_deposits,
                'realized_gains': cumulative_realized_gains,
            },
            'holdings': current_holdings,
            'closed_positions': closed_positions
        }

    def fill_missing_dates(self):
        """
        Fill in missing dates in the portfolio state history with the last known state.
        Updates the portfolio_state attribute.
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
                filled_portfolio_state[date_str] = self._copy_last_state(
                    last_state)

            current_date += datetime.timedelta(days=1)

        self.portfolio_state = filled_portfolio_state

    def _copy_last_state(self, last_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a copy of the last known portfolio state.

        Args:
            last_state (Dict[str, Any]): The last known portfolio state.

        Returns:
            Dict[str, Any]: A copy of the last known state.
        """
        return {
            'summary': {
                'cash': last_state['summary']['cash'],
                'total_deposits': last_state['summary']['total_deposits'],
                'realized_gains': last_state['summary']['realized_gains'],
            },
            'holdings': {
                symbol: {
                    'quantity': data['quantity'],
                    'total_cost': data['total_cost'],
                    'unit_cost': data['unit_cost']
                }
                for symbol, data in last_state['holdings'].items()
            },
            'closed_positions': last_state['closed_positions']
        }

    def fetch_historical_data(self):
        """
        Fetch historical price data for all symbols in the portfolio.
        Updates the historical_data and sectors attributes.
        """
        try:
            first_date = min(self.portfolio_state.keys())
            last_date = max(self.portfolio_state.keys())
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

    def get_unique_symbols(self) -> List[str]:
        """
        Get a list of unique symbols present in the portfolio history.

        Returns:
            List[str]: A sorted list of unique symbols.
        """
        unique_symbols = set()
        for date_data in self.portfolio_state.values():
            unique_symbols.update(date_data['holdings'].keys())
        return sorted(unique_symbols)

    def update_portfolio_with_market_prices(self):
        """
        Update the portfolio state history with market prices and calculate additional metrics.
        Updates the portfolio_state attribute with market data and derived metrics.
        """
        dates_to_remove = []
        self.split_adjusted_prices = self._calculate_split_adjusted_prices()

        previous_state = None
        for date, state in self.portfolio_state.items():
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
            if date_obj not in self.historical_data.index:
                dates_to_remove.append(date)
                continue

            self._update_state_with_market_data(
                state, date_obj, previous_state)
            previous_state = state

        for date in dates_to_remove:
            del self.portfolio_state[date]

    def _calculate_split_adjusted_prices(self) -> Dict[str, Dict[datetime.datetime, Decimal]]:
        """
        Calculate split-adjusted prices for all symbols in the portfolio.

        Returns:
            Dict[str, Dict[datetime.datetime, Decimal]]: A dictionary of split-adjusted prices for each symbol and date        """
        split_adjusted_prices = {}
        split_adjustments = self._collect_split_information()

        for symbol in self.get_unique_symbols():
            split_adjusted_prices[symbol] = {}
            for date, price in self.historical_data[symbol].items():
                adjusted_price = Decimal(str(price))
                if symbol in split_adjustments and date < split_adjustments[symbol]['date']:
                    adjusted_price *= split_adjustments[symbol]['ratio']
                split_adjusted_prices[symbol][date] = adjusted_price

        return split_adjusted_prices

    def _collect_split_information(self) -> Dict[str, Dict[str, Any]]:
        """
        Collect information about stock splits from the transaction history.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing split information for each affected symbol.
        """
        split_adjustments = {}
        for date, transactions in self.transaction_history.items():
            for symbol, split_data in transactions['split'].items():
                additional_quantity = Decimal(str(split_data['quantity']))
                if additional_quantity != 0:
                    # Find the holding quantity just before the split
                    split_date = datetime.datetime.strptime(date, '%Y-%m-%d')
                    pre_split_date = (
                        split_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                    if pre_split_date in self.portfolio_state:
                        original_quantity = Decimal(
                            str(self.portfolio_state[pre_split_date]['holdings'][symbol]['quantity']))
                        new_quantity = original_quantity + additional_quantity
                        split_ratio = new_quantity / original_quantity
                        print(f"Split detected for {symbol} on {
                              date}. Ratio: {split_ratio:.4f}")
                        split_adjustments[symbol] = {
                            'date': split_date,
                            'ratio': split_ratio
                        }
        return split_adjustments

    def _update_state_with_market_data(self, state: Dict[str, Any], date_obj: datetime.datetime, previous_state: Dict[str, Any]):
        """
        Update a single portfolio state with market data and calculate derived metrics.
        """
        portfolio_metrics = self._calculate_portfolio_metrics(
            state, date_obj, previous_state)

        self._update_state_metrics(state, portfolio_metrics, previous_state)

    def _calculate_portfolio_metrics(self, state: Dict[str, Any], date_obj: datetime.datetime, previous_state: Dict[str, Any]) -> Dict[str, Decimal]:
        """
        Calculate various portfolio metrics for a given state and date.
        """
        total_market_value = Decimal('0')
        total_cost_basis = Decimal('0')
        daily_gain = Decimal('0')

        for symbol, holding in state['holdings'].items():
            holding_metrics = self._calculate_holding_metrics(
                symbol, holding, date_obj, previous_state)

            if holding_metrics:
                total_market_value += holding_metrics['market_value']
                total_cost_basis += holding_metrics['total_cost']
                daily_gain += holding_metrics['daily_gain']

                self._update_holding_metrics(holding, holding_metrics)
            else:
                self._update_holding_without_market_data(holding)

        cash = state['summary']['cash']
        total_portfolio_value = total_market_value + cash
        unrealized_gain_loss = total_market_value - total_cost_basis
        daily_return = self._calculate_daily_return(
            total_portfolio_value, previous_state)

        # Calculate ROI
        roi = self._calculate_roi(total_market_value, total_cost_basis)

        return {
            'total_market_value': total_market_value,
            'total_cost_basis': total_cost_basis,
            'daily_gain': daily_gain,
            'cash': cash,
            'total_portfolio_value': total_portfolio_value,
            'unrealized_gain_loss': unrealized_gain_loss,
            'daily_return': daily_return,
            'roi': roi  # Add ROI to the returned metrics
        }

    def _calculate_roi(self, total_market_value: Decimal, total_cost_basis: Decimal) -> Decimal:
        """
        Calculate the Return on Investment (ROI).
        """
        return (total_market_value - total_cost_basis) / total_cost_basis if total_cost_basis != 0 else Decimal('0')

    def _calculate_holding_metrics(self, symbol: str, holding: Dict[str, Any], date_obj: datetime.datetime, previous_state: Dict[str, Any]) -> Dict[str, Decimal]:
        """
        Calculate metrics for a single holding.
        """
        try:
            market_price = self.split_adjusted_prices[symbol][date_obj]
            quantity = holding['quantity']
            total_cost = holding['total_cost']

            current_market_value = market_price * quantity
            unrealized_gain_loss = current_market_value - total_cost
            daily_gain = self._calculate_holding_daily_gain(
                symbol, current_market_value, quantity, previous_state)
            daily_return = self._calculate_holding_daily_return(
                symbol, current_market_value, previous_state)

            return {
                'quantity': quantity,
                'total_cost': total_cost,
                'unit_cost': total_cost / quantity if quantity > 0 else Decimal('0'),
                'market_price': market_price,
                'market_value': current_market_value,
                'unrealized_gain_loss': unrealized_gain_loss,
                'daily_gain': daily_gain,
                'daily_return': daily_return
            }
        except KeyError:
            logging.warning(f"No market data available for {
                            symbol} on {date_obj}")
            return None

    def _calculate_holding_daily_gain(self, symbol: str, current_market_value: Decimal, quantity: Decimal, previous_state: Dict[str, Any]) -> Decimal:
        """
        Calculate the daily gain for a single holding.

        Args:
            symbol (str): The symbol of the holding.
            current_market_value (Decimal): The current market value of the holding.
            quantity (Decimal): The quantity of shares held.
            previous_state (Dict[str, Any]): The previous portfolio state.

        Returns:
            Decimal: The daily gain for the holding.
        """
        if previous_state and symbol in previous_state['holdings']:
            previous_market_value = Decimal(
                str(previous_state['holdings'][symbol]['market_price'])) * quantity
            return current_market_value - previous_market_value
        return Decimal('0')

    def _calculate_holding_daily_return(self, symbol: str, current_market_value: Decimal, previous_state: Dict[str, Any]) -> Decimal:
        """
        Calculate the daily return for a single holding.

        Args:
            symbol (str): The symbol of the holding.
            current_market_value (Decimal): The current market value of the holding.
            previous_state (Dict[str, Any]): The previous portfolio state.

        Returns:
            Decimal: The daily return for the holding.
        """
        if previous_state and symbol in previous_state['holdings']:
            previous_market_value = Decimal(
                str(previous_state['holdings'][symbol]['market_value']))
            if previous_market_value != 0:
                return (current_market_value - previous_market_value) / previous_market_value
        return Decimal('0')

    def _calculate_daily_return(self, total_portfolio_value: Decimal, previous_state: Dict[str, Any]) -> Decimal:
        """
        Calculate the daily return for the entire portfolio.
        """
        if previous_state:
            previous_total_portfolio_value = Decimal(
                str(previous_state['summary']['total_portfolio_value']))
            if previous_total_portfolio_value != 0:
                return (total_portfolio_value - previous_total_portfolio_value) / previous_total_portfolio_value
        return Decimal('0')

    def _calculate_total_return(self, total_portfolio_value: Decimal, total_deposits: Decimal) -> Decimal:
        """
        Calculate the total return of the portfolio.
        """
        return (total_portfolio_value - total_deposits) / total_deposits if total_deposits != 0 else Decimal('0')

    def _calculate_cumulative_return(self, total_portfolio_value: Decimal, total_cost_basis: Decimal) -> Decimal:
        """
        Calculate the cumulative return of the portfolio based on cost basis.
        This includes both unrealized gains and any cash/dividends.
        """
        return (total_portfolio_value - total_cost_basis) / total_cost_basis if total_cost_basis != 0 else Decimal('0')

    def _update_state_metrics(self, state: Dict[str, Any], portfolio_metrics: Dict[str, Decimal], previous_state: Dict[str, Any]):
        """
        Update the portfolio state with calculated metrics.
        """
        total_deposits = state['summary']['total_deposits']
        realized_gains = state['summary']['realized_gains']
        total_portfolio_value = portfolio_metrics['total_portfolio_value']
        total_cost_basis = portfolio_metrics['total_cost_basis']

        # Calculate total return (based on deposits)
        total_return = self._calculate_total_return(
            total_portfolio_value, total_deposits)

        # Calculate cumulative return (includes unrealized gains and cash/dividends)
        cumulative_return = self._calculate_cumulative_return(
            total_portfolio_value, total_cost_basis)

        state.update({
            'summary': {
                'total_market_value': portfolio_metrics['total_market_value'],
                'total_portfolio_value': total_portfolio_value,
                'cash': portfolio_metrics['cash'],
                'total_cost_basis': total_cost_basis,
                'unrealized_gain_loss': portfolio_metrics['unrealized_gain_loss'],
                'daily_gain': portfolio_metrics['daily_gain'],
                'daily_return': portfolio_metrics['daily_return'],
                'total_deposits': total_deposits,
                'realized_gains': realized_gains,
                'total_return': total_return,
                'cumulative_return': cumulative_return,
                'roi': portfolio_metrics['roi'],  # Add ROI to the summary
            },
            'holdings': state['holdings'],
            'closed_positions': state['closed_positions']
        })

    def _update_holding_metrics(self, holding: Dict[str, Any], metrics: Dict[str, Decimal]):
        """
        Update metrics for a single holding.
        """
        holding.update(metrics)

    def _update_holding_without_market_data(self, holding: Dict[str, Any]):
        """
        Update a holding when market data is not available.
        """
        holding.update({
            'market_price': None,
            'market_value': None,
            'unrealized_gain_loss': None,
            'daily_gain': Decimal('0'),
            'daily_return': Decimal('0')
        })

    def view_portfolio_on_date(self, date: str) -> Dict[str, Any]:
        """
        Retrieve the portfolio state for a specific date.

        Args:
            date (str): The date to retrieve the portfolio state for, in 'YYYY-MM-DD' format.

        Returns:
            Dict[str, Any]: The portfolio state for the specified date.
        """
        try:
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
            date_str = date_obj.strftime('%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'")

        if date_str in self.portfolio_state:
            return {date_str: self.portfolio_state[date_str]}
        else:
            return "No portfolio data available for this date."

    def pretty_print_portfolio(self, portfolio_data: Dict[str, Any]):
        """
        Print the portfolio data in a formatted JSON structure.

        Args:
            portfolio_data (Dict[str, Any]): The portfolio data to print.
        """
        def default_serializer(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        print(json.dumps(portfolio_data, indent=4, default=default_serializer))

    def save_to_json(self, file_path: str):
        """
        Save the processed portfolio history to a JSON file.

        Args:
            file_path (str): The path to save the JSON file.
        """
        try:
            output_data = {
                "version": "1.0",
                "timestamp": datetime.datetime.now().isoformat(),
                "sectors": self.sectors,
                "portfolios": self.portfolio_state
            }

            with open(file_path, 'w') as file:
                json.dump(output_data, file, indent=2, cls=DecimalEncoder)
            logging.info(f"Portfolio history and sectors saved to {file_path}")
        except IOError as e:
            logging.error(f"Unable to write to file {file_path}: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        portfolio_history = PortfolioHistory()

        portfolio_history.process_transaction_history(
            save_output=True, output_file='portfolio_history.json')

        # Get the latest date
        last_date = max(portfolio_history.portfolio_state.keys())

        # Get the portfolio state for the latest date
        last_portfolio_state = portfolio_history.view_portfolio_on_date(
            last_date)

        # Use pretty_print_portfolio to print the latest portfolio state
        print(f"\nPortfolio state on {last_date}:")
        portfolio_history.pretty_print_portfolio(last_portfolio_state)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

        print(f"\nPortfolio state on {last_date}:")
        portfolio_history.pretty_print_portfolio(last_portfolio_state)
