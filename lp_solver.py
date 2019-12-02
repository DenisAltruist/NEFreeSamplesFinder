from scipy.optimize import linprog
import time
import numpy as np

def solve(A, b):
  c = np.zeros(len(A[0])).tolist()
  res = linprog(c=c, A_ub=A, b_ub=b, method='interior-point')
  return res['success'], np.array(res['x']).tolist()
  
def is_feasible(A, b):
  res, _ = solve(A, b)
  return res