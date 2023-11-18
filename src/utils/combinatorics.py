import pulp

transportation_problem = pulp.LpProblem("Transportation Problem", pulp.LpMinimize)
factories = ["Factory_A", "Factory_B"]
warehouses = ["Warehouse_1", "Warehouse_2"]
distribution_center = "Distribution_Center"
costs = {
    ("Factory_A", "Warehouse_1"): 7,
    ("Factory_B", "Warehouse_2"): 9,
    ("Factory_A", distribution_center): 3,
    ("Factory_B", distribution_center): 4,
    (distribution_center, "Warehouse_1"): 2,
    (distribution_center, "Warehouse_2"): 4,
}
supply = {
    "Factory_A": 80,
    "Factory_B": 70,
    distribution_center: 50,
}
demand = {
    "Warehouse_1": 60,
    "Warehouse_2": 90,
}
x = pulp.LpVariable.dicts("shipments", ((i, j) for i in factories for j in warehouses), lowBound=0, cat="Continuous")
y = pulp.LpVariable.dicts("trucking", ((i, j) for i in factories for j in warehouses), lowBound=0, upBound=50, cat="Integer")
for i in factories:
    transportation_problem += (pulp.lpSum(x[i, j] for j in warehouses) == supply[i])
for j in warehouses:
    transportation_problem += (pulp.lpSum(x[i, j] for i in factories) == demand[j])
# for i in factories:
#     for j in warehouses:
#         transportation_problem += (pulp.lpSum(y[i, j]) <= 50)
# transportation_problem += (x["Factory_A", "Warehouse_2"] == 0)
# transportation_problem += (x["Factory_B", "Warehouse_1"] == 0)
truck_cost = pulp.lpSum(costs[i, j] * y[i, j] for i in factories for j in warehouses if (i, j) in costs)
obj = (
    pulp.lpSum(costs[i, j] * x[i, j] for i in factories for j in warehouses if (i, j) in costs) + truck_cost
)
transportation_problem.setObjective(obj)
transportation_problem.solve()
print("Feasible?:", pulp.LpStatus[transportation_problem.status])
for i in factories:
    for j in warehouses:
        if x[i, j].varValue > 0:
            print(f"Ship {int(x[i, j].varValue)} units from {i} to {j}")
        if y[i, j].varValue is not None and y[i, j].varValue > 0:
            print(f"Use truckers to ship {int(y[i, j].varValue)} units from {i} to {j}")
print("Total Cost = $", pulp.value(obj))