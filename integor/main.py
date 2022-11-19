# Imports
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from typing import *

from scipy.optimize import minimize, milp, LinearConstraint, Bounds, NonlinearConstraint


variable_names : List[str] = None
idx_to_names : Dict[int, str] = None
names_to_idx : Dict[str, int] = None
n_variables : int = None

def set_variable_names(variable_names_in : List[str]) -> None:
    """Set the names that define variables of the problem.

    Args:
        variable_names_in (List[str]): a list of string, each string representing the name of variable
    """
    assert isinstance(variable_names_in, list), "variable_names_in must be a list (of strings)"
    assert all([isinstance(name, str) for name in variable_names_in]), "variable_names_in must be a list of strings"
    global variable_names, idx_to_names, names_to_idx, n_variables
    variable_names = variable_names_in
    idx_to_names = {idx: name for idx, name in enumerate(variable_names)}
    names_to_idx = {name: idx for idx, name in enumerate(variable_names)}
    n_variables = len(variable_names)




class Cost:
    """Wrapper around np.ndarray, with a LinearExpression signature and a __repr__ method"""
    def __init__(self, expression : "LinearExpression") -> None:
        self.expression = expression
        # Reconstruct the cost matrix from the expression
        cost_matrix = np.zeros(n_variables)
        for name, coefficient in zip(self.expression.names, self.expression.coefficients):
            cost_matrix[names_to_idx[name]] = coefficient
        self.cost_matrix = cost_matrix

    def __repr__(self) -> str:
        return f"Cost({self.expression})"
        


class Constraint(LinearConstraint):
    """Wrapper around scipy.optimize.LinearConstraint, with a signature in LinearExpression and a __repr__ method"""
    def __init__(self, expression : "LinearExpression", lb : float, ub : float) -> None:
        self.expression = expression
        # Reconstruct constraint vector for scipy.optimize.LinearConstraint
        self.constraint_vector = np.zeros(n_variables)
        for name, coefficient in zip(self.expression.names, self.expression.coefficients):
            self.constraint_vector[names_to_idx[name]] = coefficient
        super().__init__(self.constraint_vector, lb, ub)

    def __repr__(self) -> str:
        return f"Constraint({self.lb[0]} <= {self.expression} <= {self.ub[0]})"



class LinearExpression:

    def __init__(self, names, coefficients) -> None:
        self.names = names
        self.coefficients = coefficients

    # Define operations between LinearExpression, Variable and float/int
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            coefficients = [other * c for c in self.coefficients]
            if other == 0:
                return LinearExpression([], [])
            else:
                return LinearExpression(self.names, coefficients)
        else:
            raise TypeError(f"Cannot multiply {type(self)} and {type(other)}")

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, LinearExpression):
            names = []
            coefficients = []
            # Add variable of first expression. If variable is in second expression, add coefficient
            for name, coefficient in zip(self.names, self.coefficients):
                if name in other.names:
                    names.append(name)
                    coefficients.append(coefficient + other.coefficients[other.names.index(name)])
                else:
                    names.append(name)
                    coefficients.append(coefficient)
            # Add variable of second expression if not in first expression (already added)
            for name, coefficient in zip(other.names, other.coefficients):
                if name not in names:
                    names.append(name)
                    coefficients.append(coefficient)
            # Remove 0 coefficients
            names = [n for n, c in zip(names, coefficients) if c != 0]
            coefficients = [c for c in coefficients if c != 0]
            return LinearExpression(names, coefficients)

        elif isinstance(other, Variable):
            return self + other.as_linear_expression()

        else:
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return self - other

    # Define comparison operators for creating constraints
    def __le__(self, other : float):
        if isinstance(other, int) or isinstance(other, float):
            return Constraint(self, -np.inf, other)
        else:
            raise TypeError(f"Cannot compare {type(self)} and {type(other)}")
    
    def __ge__(self, other : float):
        if isinstance(other, int) or isinstance(other, float):
            return Constraint(self, other, np.inf)
        else:
            raise TypeError(f"Cannot compare {type(self)} and {type(other)}")

    def __eq__(self, other : float):
        if isinstance(other, int) or isinstance(other, float):
            return Constraint(self, other, other)
        else:
            raise TypeError(f"Cannot compare {type(self)} and {type(other)}")

    # Repr
    def __repr__(self) -> str:
        empty = ''
        exp = f"Expression({' + '.join([f'{empty if c == 1 else c} {n}' for c, n in zip(self.coefficients, self.names)])} )"
        return exp



class Variable:

    def __init__(self, name : str) -> None:
        if not isinstance(name, str):
            raise TypeError("Variable name must be a string")
        assert variable_names is not None, "Variable names must be set before creating variables"
        assert name in variable_names, f"Variable name {name} not in variable_names"
        self.name = name
    
    def __add__(self, other : "Variable") -> LinearExpression:
        if isinstance(other, Variable):
            return LinearExpression([self.name, other.name], [1, 1])
        elif isinstance(other, LinearExpression):
            return other + self
        else:
            raise TypeError("Cannot add a non-variable to a variable")

    def __radd__(self, other : "Variable") -> LinearExpression:
        if isinstance(other, Variable):
            return other + self
        else:
            raise TypeError("Cannot add a non-variable to a variable")

    def __sub__(self, other : "Variable") -> LinearExpression:
        if isinstance(other, Variable):
            return LinearExpression([self.name, other.name], [1, -1])
        else:
            raise TypeError("Cannot subtract a non-variable from a variable")

    def __rsub__(self, other : "Variable") -> LinearExpression:
        if isinstance(other, Variable):
            return other - self
        else:
            raise TypeError("Cannot subtract a variable from a non-variable")

    def __mul__(self, coeff : float) -> LinearExpression:
        if isinstance(coeff, int) or isinstance(coeff, float):
            return LinearExpression([self.name], [coeff])
        else:
            raise TypeError("Cannot multiply a variable by a non-number")
    
    def __rmul__(self, coeff : float) -> LinearExpression:
        if isinstance(coeff, int) or isinstance(coeff, float):
            return self * coeff
        else:
            raise TypeError("Cannot multiply a variable by a non-number")

    # As Linear Expression
    def as_linear_expression(self) -> LinearExpression:
        return LinearExpression([self.name], [1])

    # Repr
    def __repr__(self) -> str:
        return f"Variable({self.name})"



def solve_ilp(
    cost_matrix : np.ndarray,
    constraints : List[Constraint],
    integrality : Union[List[int], int] = 1,    # 1 = integer, 0 = continuous
    bounds : Optional[Tuple[float, float]] = None,
    ):
    return milp(c = cost_matrix, constraints=constraints, integrality=integrality, bounds=bounds)


def get_solution(solution_vector) -> Dict[str, float]:
    """Get the solution as a dictionary"""
    solution = {}
    for i, name in enumerate(variable_names):
        solution[name] = solution_vector[i]
    return solution



if __name__ == "__main__":
    set_variable_names(["x0", "x1", "x2", "x3"])
    x0 = Variable("x0")
    x1 = Variable("x1")
    x2 = Variable("x2")

    my_expression = x0 - x1 + 2 * x2 + x1
    my_constraint = my_expression <= 2
    print(my_expression)
    print(my_constraint)