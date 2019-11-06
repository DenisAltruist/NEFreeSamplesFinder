from scipy.optimize import linprog
import numpy as np

def is_feasible_positive(A, b):
  print(A)
  print(b)
  num_of_variables = len(A[0])
  bounds_to_apply = []
  for _ in range (0, num_of_variables):
    bounds_to_apply.append((1, None))

  c = np.zeros(num_of_variables)
  res=linprog(c=c.tolist(), A_ub=A.tolist(),b_ub=b.tolist(), bounds=bounds_to_apply)
  print(res)
  if res['success'] == True:
    return 1
  return 0

def main(A):
  ans = is_feasible_positive(A[:,:-1], A[:,-1])
  out = open("res.txt", "w")
  out.write(str(ans))
  out.close()

if __name__ == "__main__":
  A = np.loadtxt("ineq.txt")
  main(A)