import csv
import os
import time
import oapackage
import oapackage.Doptim
import numpy as np


# run_size = 50
# number_of_factors = 8
# factor_levels = [4,4,3,3,3,3,3,3]
# strength = 0


# arrayclass = oapackage.arraydata_t(factor_levels, run_size, strength, number_of_factors)

# alpha=[1,0,0]

# scores, design_efficiencies, designs, ngenerated = oapackage.Doptim.Doptimize(
#     arrayclass, nrestarts=40, optimfunc=alpha, selectpareto=True
# )
# print(scores)
# print('Generated %d designs, the best score is %.4f' % (len(designs), scores.max() ))
# print('Generated %d designs, the best D-efficiency is %.4f' % (len(designs), design_efficiencies[:,0].max() ))
# selected_array = designs[0]
# print("The array is (in transposed form):\n")
# selected_array.transposed().showarraycompact()
# oa = np.array(selected_array)
# np.savetxt('oa_200_8_opt.txt', oa, fmt='%d')

run_size = 100
strength = 2
number_of_factors = 6
factor_levels = [5,5,5,5,5,5]
arrayclass = oapackage.arraydata_t(factor_levels, run_size, strength, number_of_factors)
print(arrayclass)
al = arrayclass.create_root()
al.showarraycompact()
array_list = [arrayclass.create_root()]
array_list_3columns = oapackage.extend_arraylist(array_list, arrayclass)
array_list_4columns = oapackage.extend_arraylist(array_list_3columns, arrayclass)
print("extended to %d arrays with 3 columns" % len(array_list_3columns))
print("extended to %d arrays with 4 columns" % len(array_list_4columns))