from scipy.optimize import linprog
import datetime
import numpy as np

def is_feasible_positive(A, b):
  num_of_variables = len(A[0])
  bounds_to_apply = []
  for _ in range (0, num_of_variables):
    bounds_to_apply.append((1, None))

  c = np.zeros(num_of_variables)
  res=linprog(c=c.tolist(), A_ub=A,b_ub=b, bounds=bounds_to_apply)
  if res['success'] == True:
    return 1
  return 0