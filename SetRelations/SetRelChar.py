#############################################################
# Sublinear Characterization functions for Set Relations
# This software is released by Yuto Ogata under GNU General Public Licence v3.0
#############################################################

import numpy as np
from pulp import *

def help():
    print('--- Sublinear Scalarization for set relations ----------------')
    print('SetRelChar contains functions dual1, dual2, dual3 for checking S1 <= S2.')
    print('Read the following instruction.')
    print('dual1 and dual2 require np.array type arguments A1, b1, A2, b2, C, k')
    print('where A1, A2, C are matrices, b1, b2, k are vectors.')
    print()
    print('dual3 requires np.array type arguments A1, b1, V2, C, k.')
    print('Note that S2 needs to be finitely generated as')
    print('S2 = (x: A2x <= b2) = co V2 + cone W2')
    print('and it must be checked in advance that W2 subset cone (W1 cup C[i])')
    print('Any function can display detailed results by verbose=True.')

# dual function #1 #####################
class dual1:
  def __init__(self,A1,b1,A2,b2,C,k,verbose=False):
    self.A1 = A1
    self.b1 = b1
    self.A2 = A2
    self.b2 = b2
    self.C = C
    self.k = k
    self.verbose = verbose
    self.value = None

  def solve(self):
    results = []
    m = LpProblem(sense=LpMaximize)
    x = np.array([LpVariable(f'x_{i+1}') for i in range(len(self.A1[0]))])
    y = np.array([LpVariable(f'y_{i+1}') for i in range(len(self.A2[0]))])
    for i in range(len(self.C)):
      m += self.C[i].dot(x - y) / self.C[i].dot(self.k)
      for j in range(len(self.A1)):
        m += self.A1[j].dot(x) <= self.b1[j]
      for j in range(len(self.A2)):
        m += self.A2[j].dot(y) <= self.b2[j]
      m.solve()
      if m.status == 1:
        results += [value(m.objective)]
        if self.verbose == True:
          print(f'[sub{i+1}]', LpStatus[m.status],end=' ')
          print(value(m.objective))
          print('   (', end=' ')
          for i in range(len(x)):
            print(f'x{i+1}=',value(x[i]), end=' ')
          for i in range(len(y)):
            print(f'y{i+1}=',value(y[i]), end=' ')
          print(')')
      elif m.status == -2:
        results += [np.inf]
        if self.verbose == True:
          print(LpStatus[m.status])
      elif m.status == -1:
        results += [-np.inf]
        if self.verbose == True:
          print(LpStatus[m.status])
    self.value = max(results)
    if self.verbose == True:
      for k in range(len(results)):
        if results[k] == self.value:
          print(f'sub{k+1} is adopted')
          return True

# dual function #2 #####################
class dual2:
  def __init__(self,A1,b1,A2,b2,C,k,verbose=False):
    self.A1 = A1
    self.b1 = b1
    self.A2 = A2
    self.b2 = b2
    self.C = C
    self.k = k
    self.verbose = verbose
    self.value = None

  def solve(self):
    results_sub = []
    m_sub = LpProblem(sense=LpMaximize)
    y = np.array([LpVariable(f'y_{i+1}') for i in range(len(self.A2[0]))])

    for i in range(len(self.C)):
      m_sub += (-1 * self.C[i].dot(y)) / self.C[i].dot(self.k)
      for j in range(len(self.A2)):
        m_sub += self.A2[j].dot(y) <= self.b2[j]
      m_sub.solve()
      if m_sub.status == 1:
        results_sub += [value(m_sub.objective)]
        if self.verbose == True:
          print(f'[sub{i+1}]', LpStatus[m_sub.status],end=' ')
          print(value(m_sub.objective))
          print('   (', end=' ')
          for j in range(len(y)):
            print(f'y{j+1}=',value(y[j]), end=' ')
          print(')')
      elif m_sub.status == -2:
        results_sub += [np.inf]
        if self.verbose == True:
          print(f'[sub{i+1}]', LpStatus[m_sub.status])
      elif m_sub.status == -1:
        results_sub += [-np.inf]
        if self.verbose == True:
          print(f'[sub{i+1}]', LpStatus[m_sub.status])

    if max(results_sub) == np.inf:
      self.value = np.inf
      return True

    else:
      m = LpProblem()
      x = np.array([LpVariable(f'x_{i+1}') for i in range(len(self.A1[0]))])
      t = LpVariable('t')
      e = [LpVariable(f'e_{i}',lowBound=0) for i in range(len(self.C))]  # slack variable
      m += t
      for i in range(len(self.C)):
        m += self.C[i].dot(x) / self.C[i].dot(self.k) + results_sub[i] + e[i] == t
      for j in range(len(self.A1)):
        m += self.A1[j].dot(x) <= self.b1[j]
      m.solve()
      if m.status == 1:
        self.value = value(m.objective)
        if self.verbose == True:
          print(LpStatus[m.status],end=' ')
          print(value(m.objective))
          print('   (', end=' ')
          for j in range(len(x)):
            print(f'x{j+1}=',value(x[j]), end=' ')
          for i in range(len(self.C)):
            if e[i] == 0:
              y_res = i
          print(f',sub{i+1} is adopted:', results_sub[i], end=' ')
          print(')')
      else:
        print('Some problems occered:', LpStatus[m.status])

# i = 3

# S2 needs to be finitely generated as
# S2 = (x: A2x <= b2) = co V_2 + cone W_2
# it must be checked in advance that W2 subset cone (W1 cup C[i])

class dual3:
  def __init__(self,A1,b1,V2,C,k,verbose=False):
    self.A1 = A1
    self.b1 = b1
    self.V2 = V2
    self.C = C
    self.k = k
    self.verbose = verbose
    self.value = None

  def solve(self):
      results = []
      for k in range(len(self.V2)):
        m = LpProblem()
        x = np.array([LpVariable(f'x_{j+1}') for j in range(len(self.A1[0]))])
        t = LpVariable('t')
        m += t
        for i in range(len(self.C)):
          m += self.C[i].dot(x - self.V2[k]) / self.C[i].dot(self.k) <= t
        for j in range(len(self.A1)):
          m += self.A1[j].dot(x) <= self.b1[j]
        m.solve()
        if m.status == 1:
          results += [value(m.objective)]
          print(f'[sub{k+1}]', LpStatus[m.status],end=' ')
          print(value(m.objective))
          if self.verbose == True:
            print('   (', end=' ')
            for j in range(len(x)):
              print(f'x{j+1}=',value(x[j]), end=' ')
            for j in range(len(self.V2[k])):
              print(f'v{j+1}=',value(self.V2[k][j]), end=' ')
            print(')')
          else:
            results += [-1 * np.inf]
            print(LpStatus[m.status],end=' ')
            print(value(m.objective))
      self.value = max(results)
      if self.verbose == True:
        for i in range(len(results)):
          if results[i] == self.value:
            print(f'sub{i+1} is adopted')
            return True
      return True