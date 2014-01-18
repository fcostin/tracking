"""
tiny pulp demo to remember how the api works - from the pulp website
"""

from pulp import LpVariable, LpProblem, LpMinimize, COIN, LpStatus, value

def main():
    x = LpVariable("x", 0, 3)
    y = LpVariable("y", 0, 1)
    prob = LpProblem("myProblem", LpMinimize)
    prob += x + y <= 2
    prob += -4*x + y
    status = prob.solve(COIN(msg = 0))
    print LpStatus[status]
    print value(x)
    print value(y)

if __name__ == '__main__':
    main()

