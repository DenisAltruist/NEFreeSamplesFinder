from scipy.optimize import linprog
import datetime
import numpy as np

def solve(A, b):
  c = np.zeros(len(A[0])).tolist()
  res = linprog(c=c, A_ub=A, b_ub=b)
  return res['success'], res['x']
  
def is_feasible(A, b):
  res, _ = solve(A, b)
  return res