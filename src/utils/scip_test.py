from pyscipopt import Model

# model = Model("Example")  # model name is optional

# x = model.addVar("x")
# y = model.addVar("y", vtype="INTEGER")
# model.setObjective(x + y)
# model.addCons(2*x - y*y >= 0)
# model.optimize()
# sol = model.getBestSol()
# print("x: {}".format(sol[x]))
# print("y: {}".format(sol[y]))

model = Model()
file = open("./src/utils/MILP_slew.zpl","r")
model.readProblem("./src/utils/MILP_slew.zpl")
model.optimize()
sol = model.getBestSol()
model.writeBestSol("MILP_slew.sol")