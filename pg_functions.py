import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import inv

def t2v(A):
    """
    Homogeneous transformation to vector
    A = H = [R d
             0 1]
             
    Rotation matrix:
    R = [+cos(theta), -sin(theta)
         +sin(theta), +cos(theta)]
         
    translation vector:
    d = [x y]'
    """
    v = np.zeros((3,1), dtype = np.float64)
    
    v[0] = A[0,2] # x
    v[1] = A[1,2] # y
    v[2] = np.arctan2(A[1,0], A[0,0]) # theta
    
    return v
    
def v2t(v):
    """
    Vector to Homogeneous transformation
    A = H = [R d
             0 1]
             
    Rotation matrix:
    R = [+cos(theta), -sin(theta)
         +sin(theta), +cos(theta)]
         
    translation vector:
    d = [x y]'
    """
    x = v[0]
    y = v[1]
    theta = v[2]
    
    A = np.array([[+np.cos(theta), -np.sin(theta), x],
                  [+np.sin(theta), +np.cos(theta), y],
                  [             0,              0, 1]])
    
    return A

def solve(H, b, sparse_solve):
    """
    Solve sparse linear system H * dX = -b
    """            
    # Keep first node fixed
    H[:3,:3] += np.eye(3)
    
    if sparse_solve:
        # Transformation to sparse matrix form
        H_sparse = csr_matrix(H) 
        # Solve sparse system
        dX = spsolve(H_sparse, b)
    else:    
        # Solve linear system
        dX = np.linalg.solve(H, b)
        
    # Keep first node fixed    
    dX[:3] = [0, 0, 0] 
    
    # Check NAN
    dX[np.isnan(dX)] = 0
    
    return dX