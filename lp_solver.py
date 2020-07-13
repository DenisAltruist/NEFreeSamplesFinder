from scipy.optimize import linprog
import time
import numpy as np

def solve(A, b):
  c = np.ones(len(A[0])).tolist()
  res = linprog(c=c, A_ub=A, b_ub=b, bounds=(None, None), method='interior-point')
  return res['success'], np.array(res['x']).tolist()
  
def is_feasible(A, b):
  res, sol = solve(A, b)
  return res, sol