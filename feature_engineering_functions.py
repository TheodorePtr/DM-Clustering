import pandas as pd


from sklearn.base import TransformerMixin, BaseEstimator


class ColumnsCombinator(TransformerMixin, BaseEstimator):
    """
    Combines columns of a pandas DataFrame using arithmetic operators and creates a new column.

    Parameters:
        arguments (list of str): The column names to use as operands in the arithmetic operations.
        operators (list of str): The arithmetic operators to use between the columns. Must be one of ['+', '-', '*', '/'].
        new_column_name (str): The name of the new column to create.

    Raises:
        AssertionError: If the number of arguments is not equal to the number of operators plus one.
                        If any of the operators is not one of ['+', '-', '*', '/'].

    Methods:
        fit(X, y=None)
            Stores a copy of the input DataFrame and returns the instance.

        transform(X)
            Applies the arithmetic operations to the stored copy of the input DataFrame and returns the result.

    Attributes:
        arguments (list of str): The column names to use as operands in the arithmetic operations.
        operators (list of str): The arithmetic operators to use between the columns. Must be one of ['+', '-', '*', '/'].
        new_column_name (str): The name of the new column to create.
        X_new (pandas DataFrame): The copy of the input DataFrame used to store the result.
    """
    def __init__(self, arguments, operators, new_column_name):
        assert len(arguments) == len(operators) + 1, "number of arguments must be equal to number of operators + 1"
        assert all(op in ['+', '-', '*', '/'] for op in operators), "operators must be +, -, * or /"
        self.arguments = arguments
        self.operators = operators
        self.new_column_name = new_column_name

    def fit(self, X, y=None):
        self.X_new = X.copy()
        return self

    def transform(self, X):
        for i in range(len(self.operators)):
            if self.operators[i] == "+":
                self.X_new[self.new_column_name] = X[self.arguments[i]] + X[self.arguments[i+1]]
            elif self.operators[i] == "-":
                self.X_new[self.new_column_name] = X[self.arguments[i]] - X[self.arguments[i+1]]
            elif self.operators[i] == "*":
                self.X_new[self.new_column_name] = X[self.arguments[i]] * X[self.arguments[i+1]]
            elif self.operators[i] == "/":
                self.X_new[self.new_column_name] = X[self.arguments[i]] / X[self.arguments[i+1]]
                
        return self.X_new