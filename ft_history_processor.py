import pandas as pd
from collections import defaultdict
from datetime import datetime, date
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Transaction:
    """
    Represents a single financial transaction from the CSV file.
    """

    def __init__(self, row):
        try:
            self.trade_date = datetime.strptime(
                row['TradeDate'], '%Y-%m-%d').date()
            self.record_type = row['RecordType']
            self.action = row['Action']
            self.symbol = row['Symbol'].strip()
            self.quantity = abs(
                float(row['Quantity'])) if row['Quantity'] else 0
            self.price = float(row['Price']) if row['Price'] else 0
            self.amount = abs(float(row['Amount'])) if row['Amount'] else 0
            self.description = row['Description']
            self.cusip = row['CUSIP']
        except KeyError as e:
            raise ValueError(f"Missing required column in CSV: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid data format in CSV: {e}")


class TransactionHistory:
    """
    Manages a collection of financial transactions and provides methods for processing and viewing the transaction history.
    """

    def __init__(self):
        self.transactions_by_date = defaultdict(list)
        self.history = defaultdict(lambda: {
            'buy': defaultdict(list),
            'sell': defaultdict(list),
            'reinvestment': defaultdict(list),
            'split': defaultdict(lambda: {'quantity': 0}),
            'deposit': 0,
            'interest': 0,
            'dividend': 0
        })

    def add_transaction(self, transaction):
        """
        Adds a transaction to the transaction history.

        :param transaction: Transaction object to be added
        """
        self.transactions_by_date[transaction.trade_date].append(transaction)

    def process_transaction_history(self, input_file='FT_History.csv', save_output=False, output_file='transaction_history.json'):
        """
        Processes the transaction history from start to finish, including loading transactions from CSV,
        processing them, and optionally saving the output to a JSON file.

        :param input_file: Path to the input CSV file containing transaction history
        :param save_output: Boolean flag to determine if output should be saved
        :param output_file: Path to save the output JSON file if save_output is True
        :return: The processed transaction history
        """
        try:
            self.load_transactions_from_csv(input_file)
            if not self.transactions_by_date:
                raise ValueError(
                    "No transactions were loaded from the CSV file.")

            self.process_transactions()

            if not self.history:
                raise ValueError("No transactions were processed.")

            if save_output:
                self.save_to_json(output_file)

            logging.info(f"Successfully processed {
                         len(self.transactions_by_date)} transaction dates.")
            return self.history
        except Exception as e:
            logging.error(f"Error processing transaction history: {str(e)}")
            raise

    def load_transactions_from_csv(self, file_path):
        """
        Loads transactions from a CSV file.

        :param file_path: Path to the CSV file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("The CSV file is empty")

            required_columns = ['TradeDate', 'RecordType', 'Action',
                                'Symbol', 'Quantity', 'Price', 'Amount', 'Description', 'CUSIP']
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {
                                 ', '.join(missing_columns)}")

            transactions = [Transaction(row) for _, row in df.iterrows()]
            for transaction in transactions:
                if transaction.symbol == 'BRKB':
                    transaction.symbol = 'BRK-B'

                self.add_transaction(transaction)

            logging.info(f"Successfully loaded {
                         len(transactions)} transactions from {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty")
        except pd.errors.ParserError:
            raise ValueError(
                "Unable to parse the CSV file. Please check the file format.")
        except Exception as e:
            logging.error(f"Error loading transactions from CSV: {str(e)}")
            raise

    def process_transactions(self):
        """
        Processes all loaded transactions and updates the history.
        """
        try:
            for trade_date, daily_transactions in sorted(self.transactions_by_date.items()):
                reinvestment_cusips = set()
                pending_dividends = defaultdict(float)

                # First pass: Process all transactions except dividends
                for transaction in daily_transactions:
                    self._process_single_transaction(
                        transaction, reinvestment_cusips, pending_dividends)

                # Second pass: Process dividends
                self._process_dividends(
                    trade_date, daily_transactions, reinvestment_cusips, pending_dividends)

            logging.info(f"Successfully processed transactions for {
                         len(self.history)} dates.")
        except Exception as e:
            logging.error(f"Error processing transactions: {str(e)}")
            raise

    def _process_single_transaction(self, transaction, reinvestment_cusips, pending_dividends):
        """
        Processes a single transaction and updates the relevant history entries.
        """
        trade_date = transaction.trade_date
        if transaction.record_type == 'Trade':
            if transaction.action == 'BUY':
                if 'STK SPLIT' in transaction.description:
                    self._process_stock_split(trade_date, transaction)
                else:
                    self._process_buy(trade_date, transaction)
            elif transaction.action == 'SELL':
                self._process_sell(trade_date, transaction)
        elif transaction.record_type == 'Financial':
            if transaction.action == 'Other':
                if any(keyword in transaction.description for keyword in ['DEPOSIT', 'Wire Funds Received', 'REBATE']):
                    self._process_deposit(trade_date, transaction)
                elif 'REIN' in transaction.description:
                    self._process_reinvestment(
                        trade_date, transaction, reinvestment_cusips)
            elif transaction.action == 'Interest':
                self.history[trade_date]['interest'] += transaction.amount
            elif transaction.action == 'Dividend':
                pending_dividends[transaction.cusip] += transaction.amount

    def _process_deposit(self, trade_date, transaction):
        """
        Processes a deposit transaction.
        """
        self.history[trade_date]['deposit'] += transaction.amount
        # logging.info(f"Processed deposit: {
        #     transaction.description} - Amount: {transaction.amount}")

    def _process_stock_split(self, trade_date, transaction):
        self.history[trade_date]['split'][transaction.symbol]['quantity'] += transaction.quantity

    def _process_buy(self, trade_date, transaction):
        self.history[trade_date]['buy'][transaction.symbol].append({
            'quantity': transaction.quantity,
            'price': transaction.price,
            'amount': transaction.amount
        })

    def _process_sell(self, trade_date, transaction):
        self.history[trade_date]['sell'][transaction.symbol].append({
            'quantity': transaction.quantity,
            'price': transaction.price,
            'amount': transaction.amount
        })

    def _process_reinvestment(self, trade_date, transaction, reinvestment_cusips):
        rein_price = float(transaction.description.split('@')[1].split()[0])
        self.history[trade_date]['reinvestment'][transaction.symbol].append({
            'quantity': transaction.quantity,
            'price': rein_price,
            'amount': transaction.amount
        })
        reinvestment_cusips.add(transaction.cusip)

    def _process_dividends(self, trade_date, daily_transactions, reinvestment_cusips, pending_dividends):
        for transaction in daily_transactions:
            if transaction.record_type == 'Financial' and transaction.action == 'Dividend':
                if transaction.cusip not in reinvestment_cusips:
                    self.history[trade_date]['dividend'] += pending_dividends[transaction.cusip]

    def view_history_on_date(self, date):
        """
        Returns the transaction history state for a specific date.

        :param date: Date string in 'YYYY-MM-DD' format
        :return: Transaction history state on the given date or a message if no transactions occurred
        """
        try:
            date = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            raise ValueError("Invalid date format. Please use 'YYYY-MM-DD'")

        if date in self.history:
            return {str(date): self.history[date]}
        else:
            return "No transactions on this date."

    def pretty_print_history(self, history):
        """
        Prints a formatted JSON representation of the transaction history.

        :param history: Transaction history data to be printed
        """
        def default_serializer(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        formatted_history = {
            str(date): data for date, data in history.items()}
        print(json.dumps(formatted_history, indent=4, default=default_serializer))

    def save_to_json(self, file_path='transaction_history.json'):
        """
        Saves the entire transaction history to a JSON file.

        :param file_path: Path to save the JSON file (default: 'transaction_history.json')
        """
        try:
            formatted_history = {
                str(date): data for date, data in self.history.items()}
            with open(file_path, 'w') as json_file:
                json.dump(formatted_history, json_file, indent=4, default=str)
            logging.info(f"Transaction history has been saved to {file_path}")
        except IOError as e:
            logging.error(f"Unable to write to file {file_path}: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        transaction_history = TransactionHistory()
        processed_history = transaction_history.process_transaction_history(
            input_file='FT_History.csv',
            save_output=True,
            output_file='transaction_history.json'
        )

        # Example usage of viewing history on a specific date
        date_to_view = '2024-02-21'
        viewed_history = transaction_history.view_history_on_date(date_to_view)
        if isinstance(viewed_history, str):
            print(viewed_history)
        else:
            transaction_history.pretty_print_history(viewed_history)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
