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
    quantity: Decimal
    total_cost: Decimal


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super(DecimalEncoder, self).default(obj)


class PortfolioHistory:
    def __init__(self):
        self.transaction_history: Dict[str, Any] = {}
        self.portfolio_state: Dict[str, Any] = {}
        self.filled_portfolio_state: Dict[str, Any] = {}
        self.historical_data: pd.DataFrame = None
        self.sectors: Dict[str, str] = {}
        self.previous_state: Dict[str, Any] = None
        self.split_adjusted_prices: Dict[str,
                                         Dict[datetime.datetime, Decimal]] = {}
        self.realized_gains: Decimal = Decimal('0')

    def process_transaction_history(self, input_file: str = 'transaction_history.json',
                                    save_output: bool = False,
                                    output_file: str = 'portfolio_history.json') -> Dict[str, Any]:
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
        portfolio_state = {}
        cash = Decimal('0')
        holdings: Dict[str, Holding] = defaultdict(
            lambda: Holding(Decimal('0'), Decimal('0')))
        realized_gains = Decimal('0')

        for date, transactions in self.transaction_history.items():
            cash = self._update_cash(cash, transactions)
            holdings = self._process_splits(holdings, transactions['split'])
            cash, holdings = self._process_buys(
                cash, holdings, transactions['buy'])
            cash, holdings, realized_gains = self._process_sells(
                cash, holdings, realized_gains, transactions['sell'])
            holdings = self._process_reinvestments(
                holdings, transactions['reinvestment'])

            portfolio_state[date] = self._create_portfolio_state(
                cash, holdings, realized_gains)

        self.portfolio_state = portfolio_state
        self.realized_gains = realized_gains

    def _update_cash(self, cash: Decimal, transactions: Dict[str, Any]) -> Decimal:
        return cash + sum(Decimal(str(transactions[key])) for key in ['deposit', 'interest', 'dividend'])

    def _process_splits(self, holdings: Dict[str, Holding], splits: Dict[str, Any]) -> Dict[str, Holding]:
        for symbol, split_data in splits.items():
            if symbol in holdings:
                holdings[symbol].quantity += Decimal(
                    str(split_data['quantity']))
        return holdings

    def _process_buys(self, cash: Decimal, holdings: Dict[str, Holding], buys: Dict[str, List[Dict[str, Any]]]) -> tuple:
        for symbol, buy_list in buys.items():
            for buy in buy_list:
                quantity = Decimal(str(buy['quantity']))
                cost = abs(Decimal(str(buy['amount'])))
                holdings[symbol].quantity += quantity
                holdings[symbol].total_cost += cost
                cash -= cost
        return cash, holdings

    def _process_sells(self, cash: Decimal, holdings: Dict[str, Holding], realized_gains: Decimal,
                       sells: Dict[str, List[Dict[str, Any]]]) -> tuple:
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

                    holdings[symbol].quantity -= sell_quantity
                    holdings[symbol].total_cost -= cost_basis_sold
                    cash += sell_amount
        return cash, holdings, realized_gains

    def _process_reinvestments(self, holdings: Dict[str, Holding], reinvestments: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Holding]:
        for symbol, reinvest_list in reinvestments.items():
            for reinvest in reinvest_list:
                quantity = Decimal(str(reinvest['quantity']))
                cost = abs(Decimal(str(reinvest['amount'])))
                holdings[symbol].quantity += quantity
                holdings[symbol].total_cost += cost
        return holdings

    def _create_portfolio_state(self, cash: Decimal, holdings: Dict[str, Holding], realized_gains: Decimal) -> Dict[str, Any]:
        current_holdings = {
            symbol: {
                'quantity': float(data.quantity),
                'total_cost': float(data.total_cost),
                'unit_cost': float(data.total_cost / data.quantity) if data.quantity > 0 else 0
            }
            for symbol, data in holdings.items() if data.quantity > 0
        }
        return {
            'cash': float(cash),
            'holdings': current_holdings,
            'realized_gains': float(realized_gains)
        }

    def fill_missing_dates(self):
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
            'realized_gains': last_state['realized_gains']
        }

    def fetch_historical_data(self):
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
        unique_symbols = set()
        for date_data in self.filled_portfolio_state.values():
            unique_symbols.update(date_data['holdings'].keys())
        return sorted(unique_symbols)

    def update_portfolio_with_market_prices(self):
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
        total_market_value, total_cost_basis, daily_gain = self._calculate_portfolio_metrics(
            state, date_obj)

        cash = Decimal(str(state['cash']))
        invested_value = total_market_value
        total_portfolio_value = invested_value + cash

        unrealized_gain_loss = invested_value - total_cost_basis
        unrealized_gain_loss_percentage = (
            unrealized_gain_loss / total_cost_basis * 100) if total_cost_basis != 0 else Decimal('0')

        daily_return = self._calculate_daily_return(total_portfolio_value)

        self._update_state_metrics(state, total_portfolio_value, invested_value, total_cost_basis, daily_gain,
                                   unrealized_gain_loss, unrealized_gain_loss_percentage, daily_return)

    def _calculate_portfolio_metrics(self, state: Dict[str, Any], date_obj: datetime.datetime) -> tuple:
        total_market_value = Decimal('0')
        total_cost_basis = Decimal('0')
        daily_gain = Decimal('0')

        for symbol, holding in state['holdings'].items():
            try:
                market_price = self.split_adjusted_prices[symbol][date_obj]
                quantity = Decimal(str(holding['quantity']))
                total_cost = Decimal(str(holding['total_cost']))

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
        if self.previous_state and symbol in self.previous_state['holdings']:
            previous_market_value = Decimal(
                str(self.previous_state['holdings'][symbol]['market_price'])) * quantity
            return current_market_value - previous_market_value
        return Decimal('0')

    def _calculate_holding_daily_return(self, symbol: str, current_market_value: Decimal) -> Decimal:
        if self.previous_state and symbol in self.previous_state['holdings']:
            previous_market_value = Decimal(
                str(self.previous_state['holdings'][symbol]['market_value']))
            if previous_market_value != 0:
                return (current_market_value - previous_market_value) / previous_market_value
        return Decimal('0')

    def _calculate_daily_return(self, total_portfolio_value: Decimal) -> Decimal:
        if self.previous_state:
            previous_total_portfolio_value = Decimal(
                str(self.previous_state['total_market_value']))
            if previous_total_portfolio_value != 0:
                return (total_portfolio_value - previous_total_portfolio_value) / previous_total_portfolio_value
        return Decimal('0')

    def _update_state_metrics(self, state: Dict[str, Any], total_portfolio_value: Decimal, invested_value: Decimal,
                              total_cost_basis: Decimal, daily_gain: Decimal, unrealized_gain_loss: Decimal,
                              unrealized_gain_loss_percentage: Decimal, daily_return: Decimal):
        cash = Decimal(str(state['cash']))

        state['total_market_value'] = float(total_portfolio_value)
        state['cash'] = float(cash)
        state['invested_value'] = float(invested_value)
        state['total_cost_basis'] = float(total_cost_basis)
        state['unrealized_gain_loss'] = float(unrealized_gain_loss)
        state['unrealized_gain_loss_percentage'] = float(
            unrealized_gain_loss_percentage)
        state['realized_gains'] = float(self.realized_gains)
        state['total_gain_loss'] = float(
            unrealized_gain_loss + self.realized_gains)
        state['daily_gain'] = float(daily_gain)
        state['daily_return'] = float(daily_return)
        state['holdings'] = state['holdings']  # Keep existing holdings data

    def _update_holding_metrics(self, holding: Dict[str, Any], quantity: Decimal, total_cost: Decimal, market_price: Decimal,
                                current_market_value: Decimal, daily_gain: Decimal, daily_return: Decimal):
        holding['quantity'] = float(quantity)
        holding['total_cost'] = float(total_cost)
        holding['unit_cost'] = float(
            total_cost / quantity) if quantity > 0 else 0
        holding['market_price'] = float(market_price)
        holding['market_value'] = float(current_market_value)
        holding['unrealized_gain_loss'] = float(
            current_market_value - total_cost)
        holding['unrealized_gain_loss_percentage'] = float(
            (current_market_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
        holding['daily_gain'] = float(daily_gain)
        holding['daily_return'] = float(daily_return)

    def _update_holding_without_market_data(self, holding: Dict[str, Any]):
        holding['market_price'] = None
        holding['market_value'] = None
        holding['unrealized_gain_loss'] = None
        holding['unrealized_gain_loss_percentage'] = None
        holding['daily_gain'] = 0.0
        holding['daily_return'] = 0.0

    def view_portfolio_on_date(self, date: str) -> Dict[str, Any]:
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
        def default_serializer(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        print(json.dumps(portfolio_data, indent=4, default=default_serializer))

    def save_to_json(self, file_path: str):
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
