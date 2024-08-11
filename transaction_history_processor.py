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
        """Initialize the PortfolioHistory object with empty data structures."""
        self.transaction_history: Dict[str, Any] = {}
        self.portfolio_state: Dict[str, Any] = {}
        self.filled_portfolio_state: Dict[str, Any] = {}
        self.historical_data: pd.DataFrame = None
        self.sectors: Dict[str, str] = {}
        self.previous_state: Dict[str, Any] = None
        self.split_adjusted_prices: Dict[str,
                                         Dict[datetime.datetime, Decimal]] = {}
        self.cumulative_realized_gains: Decimal = Decimal('0')
        self.cumulative_deposits: Decimal = Decimal('0')
        self.closed_positions = {}

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
            return self.filled_portfolio_state
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

        for date, transactions in self.transaction_history.items():
            cash = self._update_cash(cash, transactions)
            holdings = self._process_splits(holdings, transactions['split'])
            cash, holdings = self._process_buys(
                cash, holdings, transactions['buy'])
            cash, holdings, realized_gains = self._process_sells(
                cash, holdings, cumulative_realized_gains, transactions['sell'])
            holdings = self._process_reinvestments(
                holdings, transactions['reinvestment'])

            cumulative_realized_gains += realized_gains
            cumulative_deposits += Decimal(str(transactions['deposit']))

            portfolio_state[date] = self._create_portfolio_state(
                cash, holdings, cumulative_realized_gains, cumulative_deposits)

        self.portfolio_state = portfolio_state
        self.cumulative_realized_gains = cumulative_realized_gains
        self.cumulative_deposits = cumulative_deposits

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
                cost = abs(Decimal(str(buy['amount'])))
                holdings[symbol].quantity += quantity
                holdings[symbol].total_cost += cost
                cash -= cost
        return cash, holdings

    def _process_sells(self, cash: Decimal, holdings: Dict[str, Holding], cumulative_realized_gains: Decimal,
                       sells: Dict[str, List[Dict[str, Any]]]) -> tuple:
        """
        Process sell transactions and update cash, holdings, and realized gains.

        Args:
            cash (Decimal): Current cash balance.
            holdings (Dict[str, Holding]): Current holdings.
            cumulative_realized_gains (Decimal): Current cumulative realized gains.
            sells (Dict[str, List[Dict[str, Any]]]): Sell transaction data.

        Returns:
            tuple: Updated cash balance, holdings, and realized gains.
        """
        realized_gains = Decimal('0')
        for symbol, sell_list in sells.items():
            for sell in sell_list:
                sell_quantity = abs(Decimal(str(sell['quantity'])))
                sell_amount = Decimal(str(sell['amount']))
                if holdings[symbol].quantity > 0:
                    avg_cost_per_share = holdings[symbol].total_cost / \
                        holdings[symbol].quantity
                    cost_basis_sold = avg_cost_per_share * sell_quantity
                    realized_gain = sell_amount - cost_basis_sold
                    realized_gains += realized_gain

                    # Record closed position
                    self._record_closed_position(
                        symbol, sell_quantity, cost_basis_sold, sell_amount, realized_gain)

                    holdings[symbol].quantity -= sell_quantity
                    holdings[symbol].total_cost -= cost_basis_sold
                    cash += sell_amount
        return cash, holdings, realized_gains

    def _record_closed_position(self, symbol: str, quantity: Decimal, cost_basis: Decimal,
                                sell_amount: Decimal, realized_gain: Decimal):
        if symbol not in self.closed_positions:
            self.closed_positions[symbol] = {
                'total_quantity': Decimal('0'),
                'total_cost_basis': Decimal('0'),
                'total_sell_amount': Decimal('0'),
                'total_realized_gain': Decimal('0')
            }

        self.closed_positions[symbol]['total_quantity'] += quantity
        self.closed_positions[symbol]['total_cost_basis'] += cost_basis
        self.closed_positions[symbol]['total_sell_amount'] += sell_amount
        self.closed_positions[symbol]['total_realized_gain'] += realized_gain

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
                cost = abs(Decimal(str(reinvest['amount'])))
                holdings[symbol].quantity += quantity
                holdings[symbol].total_cost += cost
        return holdings

    def _create_portfolio_state(self, cash: Decimal, holdings: Dict[str, Holding],
                                cumulative_realized_gains: Decimal, cumulative_deposits: Decimal) -> Dict[str, Any]:
        """
        Create a portfolio state snapshot for a given point in time.

        Args:
            cash (Decimal): Current cash balance.
            holdings (Dict[str, Holding]): Current holdings.
            cumulative_realized_gains (Decimal): Current cumulative realized gains.
            cumulative_deposits (Decimal): Total deposits made.

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

        realized_gain_percentage = (
            cumulative_realized_gains / cumulative_deposits * 100) if cumulative_deposits > 0 else Decimal('0')

        closed_positions_data = {
            'realized_gains': cumulative_realized_gains,
            'realized_gain_percentage': realized_gain_percentage,
            'positions': {
                symbol: {
                    'symbol': symbol,
                    'total_quantity': data['total_quantity'],
                    'total_cost_basis': data['total_cost_basis'],
                    'total_sell_amount': data['total_sell_amount'],
                    'total_realized_gain': data['total_realized_gain'],
                    'realized_gain_percentage': (data['total_realized_gain'] / data['total_cost_basis'] * 100) if data['total_cost_basis'] > 0 else 0
                }
                for symbol, data in self.closed_positions.items()
            }
        }

        return {
            'cash': cash,
            'holdings': current_holdings,
            'total_deposits': cumulative_deposits,
            'closed_positions': closed_positions_data
        }

    def fill_missing_dates(self):
        """
        Fill in missing dates in the portfolio state history with the last known state.
        Updates the filled_portfolio_state attribute.
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

        self.filled_portfolio_state = filled_portfolio_state

    def _copy_last_state(self, last_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a copy of the last known portfolio state.

        Args:
            last_state (Dict[str, Any]): The last known portfolio state.

        Returns:
            Dict[str, Any]: A copy of the last known state.
        """
        return {
            'cash': last_state['cash'],
            'holdings': {
                symbol: {
                    'quantity': data['quantity'],
                    'total_cost': data['total_cost'],
                    'unit_cost': data['unit_cost']
                }
                for symbol, data in last_state['holdings'].items()
            },
            'total_deposits': last_state['total_deposits'],
            'closed_positions': last_state['closed_positions']
        }

    def fetch_historical_data(self):
        """
        Fetch historical price data for all symbols in the portfolio.
        Updates the historical_data and sectors attributes.
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

    def get_unique_symbols(self) -> List[str]:
        """
        Get a list of unique symbols present in the portfolio history.

        Returns:
            List[str]: A sorted list of unique symbols.
        """
        unique_symbols = set()
        for date_data in self.filled_portfolio_state.values():
            unique_symbols.update(date_data['holdings'].keys())
        return sorted(unique_symbols)

    def update_portfolio_with_market_prices(self):
        """
        Update the portfolio state history with market prices and calculate additional metrics.
        Updates the filled_portfolio_state attribute with market data and derived metrics.
        """
        dates_to_remove = []
        self.split_adjusted_prices = self._calculate_split_adjusted_prices()

        for date, state in self.filled_portfolio_state.items():
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
            if date_obj not in self.historical_data.index:
                dates_to_remove.append(date)
                continue

            self._update_state_with_market_data(state, date_obj)
            self.previous_state = state

        for date in dates_to_remove:
            del self.filled_portfolio_state[date]

    def _calculate_split_adjusted_prices(self) -> Dict[str, Dict[datetime.datetime, Decimal]]:
        """
        Calculate split-adjusted prices for all symbols in the portfolio.

        Returns:
            Dict[str, Dict[datetime.datetime, Decimal]]: A dictionary of split-adjusted prices for each symbol and date.
        """
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
        return split_adjustments

    def _update_state_with_market_data(self, state: Dict[str, Any], date_obj: datetime.datetime):
        """
        Update a single portfolio state with market data and calculate derived metrics.

        Args:
            state (Dict[str, Any]): The portfolio state to update.
            date_obj (datetime.datetime): The date of the portfolio state.
        """
        total_market_value, total_cost_basis, daily_gain = self._calculate_portfolio_metrics(
            state, date_obj)

        cash = state['cash']
        invested_value = total_market_value
        total_portfolio_value = invested_value + cash

        unrealized_gain_loss = invested_value - total_cost_basis
        unrealized_gain_loss_percentage = (
            unrealized_gain_loss / total_cost_basis * 100) if total_cost_basis != 0 else Decimal('0')

        daily_return = self._calculate_daily_return(total_portfolio_value)

        self._update_state_metrics(state, total_portfolio_value, invested_value, total_cost_basis, daily_gain,
                                   unrealized_gain_loss, unrealized_gain_loss_percentage, daily_return)

    def _calculate_portfolio_metrics(self, state: Dict[str, Any], date_obj: datetime.datetime) -> tuple:
        """
        Calculate various portfolio metrics for a given state and date.

        Args:
            state (Dict[str, Any]): The portfolio state.
            date_obj (datetime.datetime): The date of the portfolio state.

        Returns:
            tuple: Total market value, total cost basis, and daily gain.
        """
        total_market_value = Decimal('0')
        total_cost_basis = Decimal('0')
        daily_gain = Decimal('0')

        for symbol, holding in state['holdings'].items():
            try:
                market_price = self.split_adjusted_prices[symbol][date_obj]
                quantity = holding['quantity']
                total_cost = holding['total_cost']

                current_market_value = market_price * quantity
                total_market_value += current_market_value
                total_cost_basis += total_cost

                holding_daily_gain = self._calculate_holding_daily_gain(
                    symbol, current_market_value, quantity)
                daily_gain += holding_daily_gain

                holding_daily_return = self._calculate_holding_daily_return(
                    symbol, current_market_value)

                self._update_holding_metrics(holding, quantity, total_cost, market_price,
                                             current_market_value, holding_daily_gain, holding_daily_return)
            except KeyError:
                logging.warning(f"No market data available for {
                                symbol} on {date_obj}")
                self._update_holding_without_market_data(holding)

        return total_market_value, total_cost_basis, daily_gain

    def _calculate_holding_daily_gain(self, symbol: str, current_market_value: Decimal, quantity: Decimal) -> Decimal:
        """
        Calculate the daily gain for a single holding.

        Args:
            symbol (str): The symbol of the holding.
            current_market_value (Decimal): The current market value of the holding.
            quantity (Decimal): The quantity of shares held.

        Returns:
            Decimal: The daily gain for the holding.
        """
        if self.previous_state and symbol in self.previous_state['holdings']:
            previous_market_value = Decimal(
                str(self.previous_state['holdings'][symbol]['market_price'])) * quantity
            return current_market_value - previous_market_value
        return Decimal('0')

    def _calculate_holding_daily_return(self, symbol: str, current_market_value: Decimal) -> Decimal:
        """
        Calculate the daily return for a single holding.

        Args:
            symbol (str): The symbol of the holding.
            current_market_value (Decimal): The current market value of the holding.

        Returns:
            Decimal: The daily return for the holding.
        """
        if self.previous_state and symbol in self.previous_state['holdings']:
            previous_market_value = Decimal(
                str(self.previous_state['holdings'][symbol]['market_value']))
            if previous_market_value != 0:
                return (current_market_value - previous_market_value) / previous_market_value
        return Decimal('0')

    def _calculate_daily_return(self, total_portfolio_value: Decimal) -> Decimal:
        """
        Calculate the daily return for the entire portfolio.

        Args:
            total_portfolio_value (Decimal): The current total portfolio value.

        Returns:
            Decimal: The daily return for the portfolio.
        """
        if self.previous_state:
            previous_total_portfolio_value = Decimal(
                str(self.previous_state['summary']['total_market_value']))
            if previous_total_portfolio_value != 0:
                return (total_portfolio_value - previous_total_portfolio_value) / previous_total_portfolio_value
        return Decimal('0')

    def _update_state_metrics(self, state: Dict[str, Any], total_portfolio_value: Decimal, invested_value: Decimal,
                              total_cost_basis: Decimal, daily_gain: Decimal, unrealized_gain_loss: Decimal,
                              unrealized_gain_loss_percentage: Decimal, daily_return: Decimal):
        """
        Update the portfolio state with calculated metrics.

        Args:
            state (Dict[str, Any]): The portfolio state to update.
            total_portfolio_value (Decimal): The total portfolio value.
            invested_value (Decimal): The total invested value.
            total_cost_basis (Decimal): The total cost basis of the portfolio.
            daily_gain (Decimal): The daily gain of the portfolio.
            unrealized_gain_loss (Decimal): The unrealized gain/loss of the portfolio.
            unrealized_gain_loss_percentage (Decimal): The percentage of unrealized gain/loss.
            daily_return (Decimal): The daily return of the portfolio.
        """
        # Use the existing cash and total_deposits values from the state
        cash = state['summary']['cash'] if 'summary' in state else state.get(
            'cash', Decimal('0'))
        total_deposits = state['summary']['total_deposits'] if 'summary' in state else state.get(
            'total_deposits', Decimal('0'))

        # Update the state with the new structure
        state.update({
            'summary': {
                'total_market_value': total_portfolio_value,
                'cash': cash,
                'invested_value': invested_value,
                'total_cost_basis': total_cost_basis,
                'unrealized_gain_loss': unrealized_gain_loss,
                'unrealized_gain_loss_percentage': unrealized_gain_loss_percentage,
                'daily_gain': daily_gain,
                'daily_return': daily_return,
                'total_deposits': total_deposits
            },
            'holdings': state['holdings'],  # Keep existing holdings data
            # Keep existing closed positions data
            'closed_positions': state['closed_positions']
        })

        # No need to remove old fields as they are not added in the first place

    def _update_holding_metrics(self, holding: Dict[str, Any], quantity: Decimal, total_cost: Decimal, market_price: Decimal,
                                current_market_value: Decimal, daily_gain: Decimal, daily_return: Decimal):
        """
        Update metrics for a single holding.

        Args:
            holding (Dict[str, Any]): The holding to update.
            quantity (Decimal): The quantity of shares held.
            total_cost (Decimal): The total cost of the holding.
            market_price (Decimal): The current market price.
            current_market_value (Decimal): The current market value of the holding.
            daily_gain (Decimal): The daily gain of the holding.
            daily_return (Decimal): The daily return of the holding.
        """
        holding['quantity'] = quantity
        holding['total_cost'] = total_cost
        holding['unit_cost'] = total_cost / quantity if quantity > 0 else 0
        holding['market_price'] = market_price
        holding['market_value'] = current_market_value
        holding['unrealized_gain_loss'] = current_market_value - total_cost
        holding['unrealized_gain_loss_percentage'] = (
            current_market_value - total_cost) / total_cost * 100 if total_cost > 0 else 0
        holding['daily_gain'] = daily_gain
        holding['daily_return'] = daily_return

    def _update_holding_without_market_data(self, holding: Dict[str, Any]):
        """
        Update a holding when market data is not available.

        Args:
            holding (Dict[str, Any]): The holding to update.
        """
        holding['market_price'] = None
        holding['market_value'] = None
        holding['unrealized_gain_loss'] = None
        holding['unrealized_gain_loss_percentage'] = None
        holding['daily_gain'] = 0.0
        holding['daily_return'] = 0.0

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

        if date_str in self.filled_portfolio_state:
            return {date_str: self.filled_portfolio_state[date_str]}
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
                "portfolios": self.filled_portfolio_state
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
        last_date = max(portfolio_history.filled_portfolio_state.keys())

        # Get the portfolio state for the latest date
        last_portfolio_state = portfolio_history.view_portfolio_on_date(
            last_date)

        # Use pretty_print_portfolio to print the latest portfolio state
        print(f"\nPortfolio state on {last_date}:")
        portfolio_history.pretty_print_portfolio(last_portfolio_state)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
