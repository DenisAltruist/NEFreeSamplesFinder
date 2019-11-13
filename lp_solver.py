from cvxopt import spmatrix, matrix, solvers
import datetime
import numpy as np

def is_feasible_positive(I, J, V, B, num_of_variables):
  solvers.options['show_progress'] = False
  A = spmatrix(V, I, J, size=(len(B), num_of_variables))
  B = matrix(B)
  c = np.zeros(num_of_variables).tolist()
  sol=solvers.lp(matrix(c), A, B)
  # print(sol['status'])
  return sol['status'] == 'optimal'
