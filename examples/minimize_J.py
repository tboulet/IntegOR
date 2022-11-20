# Imports
import numpy as np


def J(x):
    """Function to minimize"""
    return x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3]

def f(x):
    """Inequality constraint. f(x) must be positive"""
    return np.array([
        x[0] + x[1] - 1,
        x[2] + x[3] - 1,
        x[0] + x[2] - 1,
    ])

def g(x):
    """Equality constraint. g(x) must be zero"""
    return np.array([
        x[1] + x[3] - 1,
    ])

def verify(x):
    """Verify that the solution satisfies the constraints"""
    print("f(x) = ", f(x), " (should be positive or zero)")
    print("g(x) = ", g(x), " (should be zero)")
    print("J(x) = ", J(x), " (should be minimum)")




# ==================== SOLVE PIPELINE ====================
from integor import set_variable_names, Variable, get_cost_matrix, solve_ilp, get_solution

# Set the names of the variables
set_variable_names(["x0", "x1", "x2", "x3"])

# Define variables as python Variable objects
x0 = Variable("x0")
x1 = Variable("x1")
x2 = Variable("x2")
x3 = Variable("x3")

# Define the constraints from the variables
constraint1 = x0 + x1 >= 1
constraint2 = x2 + x3 >= 1
constraint3 = x0 + x2 >= 1
constraint4 = x1 + x3 == 1
constraints = [constraint1, constraint2, constraint3, constraint4]

# Define the cost from the variables
cost_matrix = get_cost_matrix(x0 + 2 * x1 + 3 * x2 + 4 * x3)

# Integrality - type of the solution : 0 for continuous, 1 for integer-bounded
integrality = np.ones(len(cost_matrix)) * 1  # You can let this as is

# Solve
res = solve_ilp(cost_matrix=cost_matrix, constraints=constraints, integrality=integrality)
solution_vector = res.x

print(res)
print("\nSolution vector: ", solution_vector)

print("\nVerify that the solution satisfies the constraints:")
verify(res.x)

print("\nGet the solution as a dictionary:")
print(get_solution(res.x))