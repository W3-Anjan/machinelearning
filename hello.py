import numpy as np
from numpy import array
from numpy import inf
from numpy.linalg import norm
from numpy.linalg import eig



#Eigen Decomposition
Array = np.array([[-6,3],
                  [4,5]])
print(Array)
values, vector = eig(Array)
print('Eigen Value',values,'\n')
print('Eigen Vector', vector,'\n')

#Norm
a = np.array([1,2,3])
print(a)
l1 = norm(a,1)
l2 = norm(a,2)
maxNorm = norm(a, inf)
print('L1 Norm: ',l1, '\n')
print('L2 Norm,', l2, '\n')
print('Max Norm,', maxNorm, '\n')

#Vector (1-dimensional array) (1st order tensor)
x = np.array([1,2,3,4,5,6])
#print(x)
#print(type(x))

#Matrix (2-dimensional array) (2nd order tensor)
# Row/ Column (3*3)
m = np.array([[1,5,2],
              [4,7,4],
              [2,0,9]])
#print(m)
#matrix transpose (clmn to row, row to clmn)
#print('Matrix Transpose:\n', m.transpose(), '\n')

# matrix determinant
#print ('Matrix Determinant:', np.linalg.det(m), '\n')


#Tensor (N-dimensional array)
#Row/Column/Stack or Layer (3*3*3)


