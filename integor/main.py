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
    def __init__(self, expression : "Variable") -> None:
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
    def __init__(self, expression : "Variable", lb : float, ub : float) -> None:
        # Reconstruct constraint vector for scipy.optimize.LinearConstraint, ignoring the constant term
        self.constraint_vector = np.zeros(n_variables)
        for name, coefficient in expression.content.items():
            if name != Variable.unit_name:
                self.constraint_vector[names_to_idx[name]] = coefficient
        # Remove the constant term from the two bounds
        if Variable.unit_name in expression.content:
            lb -= expression.content[Variable.unit_name]
            ub -= expression.content[Variable.unit_name]
        # Call the parent constructor
        super().__init__(A=self.constraint_vector, lb=lb, ub=ub)

    def __repr__(self) -> str:
        # Reconstruct the expression from the constraint vector
        expression = Variable.get_constant_null()
        for idx, coefficient in enumerate(self.constraint_vector):
            if coefficient != 0:
                expression += coefficient * Variable(idx_to_names[idx]) 
        return f"Constraint({self.lb[0]} <= {expression} <= {self.ub[0]})"


class Variable:

    unit_name = '1'

    def __init__(self, 
            name : str = None, 
            value : Union[int, float] = None, 
            variables : List["Variable"] = None, coefficients : List[float] = None,
                ) -> None:
        assert int(name is None) + int(variables is None) + int(value is None) == 2, "Provide one among name, value or variables/coefficients as non None"
        if variables is None: assert coefficients is None and len(coefficients) == len(variables), "Provide both variables and coefficients if creating a variable from other variables, of the same length"

        if name is not None:
            self.name = name
            self.content = {name: 1} 
        
        elif value is not None:
            self.name = None
            self.content = {self.unit_name: value}

        elif variables is not None:
            self.name = None
            self.content = {}
            # Add variables to content
            for variable, coefficient in zip(variables, coefficients):
                for name, value in variable.content.items():
                    if name in self.content:
                        self.content[name] += coefficient * value
                    else:
                        self.content[name] = coefficient * value
            # Remove 0 coefficients
            self.content = {name: value for name, value in self.content.items() if value != 0}

        else:
            raise ValueError("Provide one among name, value or variables/coefficients as non None")

    @classmethod
    def get_constant_unit(cls):
        return Variable(value=1)
    
    @classmethod
    def get_constant_null(cls):
        return Variable(value=0)

    # Multiplication : can happen between a variable and a number only.
    def __mul__(self, other) -> "Variable":
        if isinstance(other, Variable):
            raise TypeError("Cannot multiply two variables.")
        elif isinstance(other, (int, float)):
            return Variable(variables=[self], coefficients=[other])
        else:
            raise TypeError(f"Cannot multiply {type(self)} and {type(other)}")
    def __rmul__(self, other):
        return self * other

    # Addition : can happen between a variable and a number or between two variables
    def __add__(self, other):
        if isinstance(other, Variable):
            return Variable(variables=[self, other], coefficients=[1, 1])
        elif isinstance(other, (int, float)):
            return Variable(variables=[self, self.get_constant_unit()], coefficients=[1, other])
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
        elif isinstance(other, Variable):
            return self - other <= 0
        elif isinstance(other, Variable):
            return self - other <= 0
        else:
            raise TypeError(f"Cannot compare {type(self)} and {type(other)}")
    
    def __ge__(self, other : float):
        if isinstance(other, int) or isinstance(other, float):
            return Constraint(self, other, np.inf)
        elif isinstance(other, Variable):
            return self - other >= 0
        elif isinstance(other, Variable):
            return self - other >= 0
        else:
            raise TypeError(f"Cannot compare {type(self)} and {type(other)}")

    def __eq__(self, other : float):
        if isinstance(other, int) or isinstance(other, float):
            return Constraint(self, other, other)
        elif isinstance(other, Variable):
            return self - other == 0
        elif isinstance(other, Variable):
            return self - other == 0
        else:
            raise TypeError(f"Cannot compare {type(self)} and {type(other)}")

    # Repr
    def __repr__(self) -> str:
        if self.name is not None:
            return self.name
        elif len(self.content) != 0:
            res = ''
            for name, value in self.content.items():
                if name == self.unit_name:
                    res += f' + {value}'
                elif value == 1:
                    res += f' + {name}'
                else:
                    res += f' + {value} * {name}'
            return res[3:]
        else:
            return '0'


def get_cost_matrix(expression : Variable):
    cost_matrix = np.zeros((n_variables,))
    for name, coefficient in expression.content.items():
        if name != Variable.unit_name:
            cost_matrix[names_to_idx[name]] = coefficient
    if Variable.unit_name in expression.content and expression.content[Variable.unit_name] != 0:
        print(f"Warning : the constant term {expression.content[Variable.unit_name]} of the expression {expression} for get_cost_matrix is ignored, as the optimization doesn't change to a constant term. The function optimized will actually be {expression - expression.content[Variable.unit_name]}")
    return cost_matrix


def solve_ilp(
    cost_matrix : np.ndarray,
    constraints : List[Constraint],
    integrality : Union[List[int], int] = 1,    # 1 = integer, 0 = continuous
    bounds : Optional[Tuple[float, float]] = (0, 1),
    ):
    return milp(c = cost_matrix, constraints=constraints, integrality=integrality, bounds=bounds)


def get_solution(solution_vector : np.ndarray) -> Dict[str, float]:
    """Get the solution as a dictionary"""
    solution = {}
    for i, name in enumerate(variable_names):
        solution[name] = solution_vector[i]
    return solution


if __name__ == "__main__":
    set_variable_names(["x0", "x1", "x2", "x3"])
    x0 = Variable(name = "x0")
    x1 = Variable(name = "x1")
    x2 = Variable(name = "x2")

    
    print(x0 + 2 * x2 + x1)
    print(1 + x0)
    print(5*x1)
    print(0 * x0)
    print(x0 - x0)
    print(x0 - 1)
    print(x0 - x0 + 4)
    my_constraint = x0 + 2 * x2 + x1 + 5 <= 2
    print(my_constraint)
    print(get_cost_matrix(x0 + 2 * x2 + x1 ))
    print(get_cost_matrix(x0 + 2 * x2 + x1 + 5))