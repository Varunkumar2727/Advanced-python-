# --------------------------------------------------------------------------- #
# Plain function API
# --------------------------------------------------------------------------- #
def add(a, b):
    """Return a + b."""
    return a + b


def subtract(a, b):
    """Return a - b."""
    return a - b


def multiply(a, b):
    """Return a * b."""
    return a * b


def divide(a, b):
    """Return a / b. Raises ZeroDivisionError if b is 0."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return a / b


def modulus(a, b):
    """Return a % b. Raises ZeroDivisionError if b is 0."""
    if b == 0:
        raise ZeroDivisionError("Cannot compute modulus with divisor 0.")
    return a % b


def power(a, b):
    """Return a raised to the power of b."""
    return a ** b


def square_root(a):
    """Return the square root of a. Raises ValueError if a is negative."""
    if a < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return a ** 0.5


# --------------------------------------------------------------------------- #
# Class API -- keeps a running value and supports method chaining
# --------------------------------------------------------------------------- #
class Calculator:
    """A calculator that keeps a running value.

    Example:
        calc = Calculator(10)
        calc.add(5).subtract(3).multiply(2)
        print(calc.result())   # 24
    """

    def __init__(self, start=0):
        self.value = start
        self.history = []  # list of (operation, operand, result) tuples

    def _record(self, op_name, operand, new_value):
        self.history.append((op_name, operand, new_value))
        self.value = new_value
        return self  # enables chaining: calc.add(1).subtract(2)

    def add(self, x):
        return self._record("add", x, add(self.value, x))

    def subtract(self, x):
        return self._record("subtract", x, subtract(self.value, x))

    def multiply(self, x):
        return self._record("multiply", x, multiply(self.value, x))

    def divide(self, x):
        return self._record("divide", x, divide(self.value, x))

    def modulus(self, x):
        return self._record("modulus", x, modulus(self.value, x))

    def power(self, x):
        return self._record("power", x, power(self.value, x))

    def square_root(self):
        return self._record("square_root", None, square_root(self.value))

    def result(self):
        """Return the current running value."""
        return self.value

    def reset(self, start=0):
        """Reset the running value (and clear history) back to `start`."""
        self.value = start
        self.history = []
        return self

    def show_history(self):
        """Print every operation performed since the last reset."""
        if not self.history:
            print("No operations yet.")
            return
        for op_name, operand, new_value in self.history:
            if operand is None:
                print(f"{op_name}() -> {new_value}")
            else:
                print(f"{op_name}({operand}) -> {new_value}")

    def __repr__(self):
        return f"Calculator(value={self.value})"


# --------------------------------------------------------------------------- #
# Demo when run directly (not triggered when imported elsewhere)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("--- Function API ---")
    print("add(4, 5) =", add(4, 5))
    print("subtract(10, 3) =", subtract(10, 3))
    print("multiply(6, 7) =", multiply(6, 7))
    print("divide(20, 4) =", divide(20, 4))
    print("modulus(10, 3) =", modulus(10, 3))
    print("power(2, 8) =", power(2, 8))
    print("square_root(81) =", square_root(81))

    print("\n--- Class API (chained) ---")
    calc = Calculator(10)
    calc.add(5).multiply(2).subtract(4)
    print("Result:", calc.result())
    calc.show_history()

    print("\n--- Simple interactive mode ---")
    print("Enter two numbers and an operator (+, -, *, /, %, ^), or 'q' to quit.")
    while True:
        raw = input("> ").strip()
        if raw.lower() == "q":
            break
        try:
            a_str, op, b_str = raw.split()
            a, b = float(a_str), float(b_str)
            if op == "+":
                print(add(a, b))
            elif op == "-":
                print(subtract(a, b))
            elif op == "*":
                print(multiply(a, b))
            elif op == "/":
                print(divide(a, b))
            elif op == "%":
                print(modulus(a, b))
            elif op == "^":
                print(power(a, b))
            else:
                print("Unknown operator. Use one of: + - * / % ^")
        except ValueError:
            print("Format: <number> <operator> <number>   e.g. 5 + 3")
        except ZeroDivisionError as e:
            print("Error:", e)
