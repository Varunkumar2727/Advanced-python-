"""
exception_handling_demo.py
===========================
A practical demonstration of Python exception handling: try/except/
else/finally, catching multiple exception types, raising exceptions,
custom exception classes, and exception chaining.

Can be run directly for an interactive menu-driven demo, or imported
to use the individual functions/classes elsewhere.

Run:
    python exception_handling_demo.py
"""


# --------------------------------------------------------------------------- #
# 1) Basic try / except / else / finally
# --------------------------------------------------------------------------- #
def safe_divide(a, b):
    """Divide a by b, handling the ZeroDivisionError explicitly.

    Demonstrates the four-part structure:
      try     -> code that might fail
      except  -> runs only if that specific error occurs
      else    -> runs only if NO error occurred
      finally -> always runs, error or not (cleanup code goes here)
    """
    try:
        result = a / b
    except ZeroDivisionError:
        print(f"Error: cannot divide {a} by zero.")
        return None
    else:
        print(f"{a} / {b} = {result}  (no error occurred)")
        return result
    finally:
        print("safe_divide: division attempt finished.\n")


# --------------------------------------------------------------------------- #
# 2) Catching multiple, specific exception types
# --------------------------------------------------------------------------- #
def parse_and_index(values, index_str):
    """Convert index_str to an int and index into `values` with it.

    Shows catching several distinct exception types separately, since
    each one usually needs a different response/message.
    """
    try:
        index = int(index_str)          # can raise ValueError
        return values[index]            # can raise IndexError
    except ValueError:
        print(f"'{index_str}' is not a valid integer index.")
    except IndexError:
        print(f"Index {index_str} is out of range for a list of length {len(values)}.")
    except TypeError as e:
        print(f"Type error while indexing: {e}")
    return None


def safe_dict_lookup(d, key):
    """Look up `key` in dict `d`, handling a missing key gracefully."""
    try:
        return d[key]
    except KeyError:
        print(f"Key '{key}' was not found. Available keys: {list(d.keys())}")
        return None


def safe_read_file(path):
    """Attempt to read a file, handling the common failure cases."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {path}")
    except PermissionError:
        print(f"Permission denied when reading: {path}")
    except OSError as e:
        # Catch-all for other OS-level I/O problems (disk error, etc.)
        print(f"OS error while reading {path}: {e}")
    return None


# --------------------------------------------------------------------------- #
# 3) Custom exception classes
# --------------------------------------------------------------------------- #
class InsufficientFundsError(Exception):
    """Raised when a withdrawal exceeds the available balance."""

    def __init__(self, balance, requested):
        self.balance = balance
        self.requested = requested
        message = f"Cannot withdraw {requested}: only {balance} available."
        super().__init__(message)


class NegativeAmountError(Exception):
    """Raised when an amount that must be positive is negative or zero."""
    pass


class BankAccount:
    """A tiny bank account used to demonstrate raising custom exceptions."""

    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise NegativeAmountError(f"Deposit amount must be positive, got {amount}.")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount <= 0:
            raise NegativeAmountError(f"Withdrawal amount must be positive, got {amount}.")
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount
        return self.balance


def demo_custom_exceptions():
    account = BankAccount(balance=100)
    print(f"Starting balance: {account.balance}")

    for amount in [50, -10, 500]:
        try:
            account.withdraw(amount)
            print(f"Withdrew {amount}. New balance: {account.balance}")
        except NegativeAmountError as e:
            print(f"Negative amount error: {e}")
        except InsufficientFundsError as e:
            print(f"Insufficient funds error: {e}")
    print()


# --------------------------------------------------------------------------- #
# 4) Exception chaining ("raise ... from ...") and re-raising
# --------------------------------------------------------------------------- #
class ConfigLoadError(Exception):
    """Raised when application configuration cannot be loaded."""
    pass


def load_config_value(config, key):
    """Look up a config value; wrap the low-level KeyError in a more
    meaningful, higher-level exception while preserving the original
    traceback via `raise ... from e` (exception chaining)."""
    try:
        return config[key]
    except KeyError as e:
        raise ConfigLoadError(f"Missing required config key: '{key}'") from e


def demo_exception_chaining():
    config = {"host": "localhost", "port": 8080}
    try:
        load_config_value(config, "api_key")
    except ConfigLoadError as e:
        print(f"Caught: {e}")
        print(f"Original cause: {e.__cause__!r}\n")


# --------------------------------------------------------------------------- #
# 5) Nested try/except and cleanup with finally (simulated resource)
# --------------------------------------------------------------------------- #
class FakeConnection:
    """Stand-in for something like a DB or network connection."""

    def __init__(self, name):
        self.name = name
        self.open_ = True
        print(f"Connection '{self.name}' opened.")

    def query(self, should_fail):
        if should_fail:
            raise RuntimeError(f"Query failed on connection '{self.name}'.")
        return "query result"

    def close(self):
        self.open_ = False
        print(f"Connection '{self.name}' closed.")


def demo_nested_try_finally(should_fail):
    conn = FakeConnection("demo-db")
    try:
        try:
            result = conn.query(should_fail)
            print(f"Got result: {result}")
        except RuntimeError as e:
            print(f"Query-level error handled: {e}")
    finally:
        # Guaranteed to run whether the query succeeded, failed, or even
        # if an unrelated exception had propagated past the inner except.
        conn.close()
    print()


# --------------------------------------------------------------------------- #
# Interactive menu (only runs when this file is executed directly)
# --------------------------------------------------------------------------- #
def run_menu():
    menu = """
Exception Handling Demo
========================
1. Safe divide (try/except/else/finally)
2. Parse & index a list (multiple except blocks)
3. Dictionary lookup (KeyError)
4. Read a file (FileNotFoundError / PermissionError / OSError)
5. Custom exceptions (bank account)
6. Exception chaining (raise ... from ...)
7. Nested try/finally (simulated connection cleanup)
q. Quit
"""
    sample_list = [10, 20, 30]
    sample_dict = {"name": "Alice", "role": "engineer"}

    while True:
        print(menu)
        choice = input("Choose an option: ").strip().lower()

        if choice == "1":
            a = input("Enter numerator: ").strip()
            b = input("Enter denominator: ").strip()
            try:
                safe_divide(float(a), float(b))
            except ValueError:
                print("Please enter valid numbers.\n")

        elif choice == "2":
            idx = input(f"List is {sample_list}. Enter an index: ").strip()
            parse_and_index(sample_list, idx)
            print()

        elif choice == "3":
            key = input(f"Dict has keys {list(sample_dict.keys())}. Enter a key: ").strip()
            print("Value:", safe_dict_lookup(sample_dict, key), "\n")

        elif choice == "4":
            path = input("Enter a file path to read: ").strip()
            content = safe_read_file(path)
            if content is not None:
                print("File content (first 200 chars):", content[:200], "\n")

        elif choice == "5":
            demo_custom_exceptions()

        elif choice == "6":
            demo_exception_chaining()

        elif choice == "7":
            fail = input("Simulate a query failure? (y/n): ").strip().lower() == "y"
            demo_nested_try_finally(fail)

        elif choice == "q":
            print("Goodbye.")
            break

        else:
            print("Unknown option, try again.\n")


if __name__ == "__main__":
    run_menu()
