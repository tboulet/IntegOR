from integor import set_variable_names, Cost, Variable, LinearExpression, Constraint

set_variable_names(["x0", "x1", "x2", "x3"])

def test_addition():
    x0 = Variable("x0")
    x1 = Variable("x1")
    x2 = Variable("x2")
    x3 = Variable("x3")
    exp = x0 + x1 + x2 + x3
    exp2 = x0 + x1 + x2

    # Variable with numbers
    try:
        x0 + 1
        assert False
    except: pass
    try:
        1 + x0
        assert False
    except: pass
    # Variable with variables
    x0 + x1
    # Variable with linear expression
    x0 + exp
    exp + x0
    # Linear expression with numbers
    try:
        exp + 1
        assert False
    except: pass
    try:
        1 + exp
        assert False
    except: pass
    # Linear expression with linear expression
    exp + exp2


def test_multiplication():
    x0 = Variable("x0")
    x1 = Variable("x1")
    x2 = Variable("x2")
    x3 = Variable("x3")
    exp = x0 + x1 + x2 + x3
    exp2 = x0 + x1 + x2

    # Variable with numbers
    2 * x0
    x0 * 2
    # Variable with variables
    try:
        x0 * x1
        assert False
    except: pass
    # Variable with linear expression
    try:
        x0 * exp
        assert False
    except: pass
    # Linear expression with numbers
    2 * exp
    exp * 2
    # Linear expression with linear expression
    try:
        exp * exp2
        assert False
    except: pass