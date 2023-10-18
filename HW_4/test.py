import numpy as np
A = np.array([1, 2, 3, 4, 5])
B = [A.copy() for _ in range(5)]
C = A + 1
E_val = np.ones(10)
B[0] = C
print(B) 
print(E_val)
print(E_val[0])
E_val[0] = 10 
print(E_val[0])