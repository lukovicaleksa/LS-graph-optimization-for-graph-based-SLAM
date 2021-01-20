import numpy as np
import matplotlib.pyplot as plt
import time 

from pg_functions import *


class Vertex:
    """
    Pose Graph Vertex(Node) Class
    """
    def __init__(self, x, y, theta):
        """
        Vertex constructor
        """
        self.x = x         # x coordinate position [m]
        self.y = y         # y coordinate position [m]
        self.theta = theta # yaw - rotation [radians]
        
        
class Edge:
    """
    Pose Graph Edge Class
    """
    def __init__(self, id_from, id_to, mean, inf_mat):
        """
        Edge constructor
        """
        self.id_from = id_from    # viewing frame of this edge
        self.id_to = id_to        # pose being observed from the viewing frame
        self.mean = mean          # Predicted Virtual measurement - z_ij
        self.inf_mat = inf_mat    # Edge Information matrix - omega_ij (~ 1 / covariance)

        
class PoseGraph:
    """
    Pose Graph class
    """
    def __init__(self, vertices = [], edges = [], verbose = True):
        """
        Pose Graph constructor
        """
        self.vertices = vertices # Pose graph vertices(nodes)
        self.edges = edges       # Pose graph edges(constraints)
        self.H = []              # Information matrix of the system (constraints contribution)
        self.b = []              # Coefficient vector
        self.verbose = verbose   # Show optimization steps

    def read_graph(self, v_data, e_data):
        """
        Initialize Pose Graph with Data/Edges.dat and Data/Vertices.dat
        """
        # Vertices file
        vertices = np.loadtxt(v_data, usecols = range(1,5))
        
        for v in vertices:
            self.vertices.append(Vertex(x = v[1], 
                                        y = v[2], 
                                        theta = v[3]))
            
        # Edges file
        edges = np.loadtxt(e_data, usecols = range(1,12))
        
        for e in edges:
            e_mean = e[2:5]
            e_inf_mat = np.zeros((3,3), dtype = np.float64)
            e_inf_mat[0,0] = e[5]
            e_inf_mat[1,0] = e_inf_mat[0,1] = e[6]
            e_inf_mat[1,1] = e[7]
            e_inf_mat[2,2] = e[8]
            e_inf_mat[0,2] = e_inf_mat[2,0] = e[9]
            e_inf_mat[1,2] = e_inf_mat[2,1] = e[10]
            
            self.edges.append(Edge(id_from = int(e[0]), 
                                   id_to = int(e[1]),
                                   mean = e_mean,
                                   inf_mat = e_inf_mat))
            
    def optimize(self, max_iterations, tolerance):
        """
        Pose Graph optimization - Least Squares error minimization
        """
        norm_dX_out = []
        tol_cnt = 0 # tolerance counter
        opt_iterations = None # number of iterations in optimization process 
        t_opt_start = time.time()
        
        for i in range(max_iterations):
            t_step_start = time.time() 
            
            if self.verbose:
                print('\r\nPose Graph Optimization, iteration %d.' % (i + 1))
                
            # Create empty matrix H and vector b 
            # dim(H) = 3n x 3n 
            self.H = np.zeros((len(self.vertices) * 3, len(self.vertices) * 3), dtype = np.float64)
            # dim(b) = 3n x 1
            self.b = np.zeros((len(self.vertices) * 3, 1), dtype = np.float64)

            if self.verbose:
                print ('Linearizing ...')
            
            # Linearization of error function and formulating a sparse Linear system
            self.linearize()
            
            if self.verbose:
                print('Solving ...')
                
            dX = solve(H = self.H, b = -self.b, sparse_solve = True)
            norm_dX = np.linalg.norm(dX)
            norm_dX_out.append(norm_dX)
            
            if self.verbose:
                print('Updating vertices ...')
                
            # Update graph vertices(nodes) with dX = [dx dy dtheta]
            self.update_vertices(dX)
            
            if self.verbose:
                print('Step duration: %f [s]' % (time.time() - t_step_start))
                print('|dX| = %f' % norm_dX)
                
            # Convergence check
            if i >= 1 and np.abs(norm_dX_out[i] - norm_dX_out[i-1]) < tolerance:
                tol_cnt += 1
            else:
                tol_cnt = 0
                
            if tol_cnt >= 3:
                opt_iterations = i + 1
                break
            
        if self.verbose:
            if opt_iterations == None:
                print('\r\nOptimization process finished - maximum number of iterations reached!')
            else:        
                print('\r\nOptimization process converged after %d iterations!' % opt_iterations)
            print('Optimization process duration: %.2f [s]' % (time.time() - t_opt_start))
        
        return norm_dX_out
            
    def linearize(self):
        """
        Linearization of error function and formulating a sparse Linear system
        """
        for e in self.edges:
            # get indexes of connected nodes
            i = e.id_from
            j = e.id_to
            
            # i-th and j-th node
            X_i = np.array([self.vertices[i].x, self.vertices[i].y, self.vertices[i].theta])
            t_i = X_i[:2].reshape(2,1)
            theta_i = X_i[2]
            
            X_j = np.array([self.vertices[j].x, self.vertices[j].y, self.vertices[j].theta])
            t_j = X_j[:2].reshape(2,1)
            theta_j = X_j[2]
            
            # Prediction for X_j seen from frame of X_i
            Z_ij = e.mean
            t_ij = Z_ij[:2].reshape(2,1)
            theta_ij = Z_ij[2]
            
            omega_ij = e.inf_mat
            
            # Rotation matrix of i-th node
            R_i = v2t(X_i)[:2,:2]
            
            # Rotation matrix of i-j constraint
            R_ij = v2t(Z_ij)[:2,:2]
            
            # Error vector calculation
            # e = [R_ij * (R_i * (t_j - t_i) - t_ij)
            #      theta_j - theta_i - theta_ij]
            e_ij = t2v(np.dot(np.dot(inv(v2t(Z_ij)), inv(v2t(X_i))), v2t(X_j)))
            
            # Derivative of R_i with respect to theta_i
            dR_i = np.array([[-np.sin(theta_i), -np.cos(theta_i)],
                             [+np.cos(theta_i), -np.sin(theta_i)]])
            
            # Jacobians of the error function
            # A_ij = de_ij/dx_i = [-R_ij * R_i , R_ij * dR_i * (t_j - t_i)
            #                                 0, -1]
            A_ij_11_block = np.dot(-R_ij.T, R_i.T)
            A_ij_12_block = np.dot(np.dot(R_ij.T, dR_i.T), t_j - t_i)
            A_ij_21_22_block = np.array([0, 0, -1])
            
            A_ij = np.vstack((np.hstack((A_ij_11_block, A_ij_12_block)),A_ij_21_22_block))
            
            # B_ij = de_ij/dx_j = [R_ij * R_i, 0
            #                               0, 1]
            B_ij_11_block = np.dot(R_ij.T, R_i.T)
            B_ij_12_block = np.zeros((2,1), dtype = np.float64)
            B_ij_21_22_block = np.array([0, 0, 1])
            
            B_ij = np.vstack((np.hstack((B_ij_11_block, B_ij_12_block)),B_ij_21_22_block))
        
            # Compute the contributions of the constraints to the linear system
            # Contribution to H
            contr_H_ii = np.dot(np.dot(A_ij.T, omega_ij), A_ij)
            contr_H_ij = np.dot(np.dot(A_ij.T, omega_ij), B_ij)
            contr_H_ji = contr_H_ij.T
            contr_H_jj = np.dot(np.dot(B_ij.T, omega_ij), B_ij)
            
            # Contribution to b
            contr_b_i = np.dot(np.dot(A_ij.T, omega_ij), e_ij).reshape(3,1)
            contr_b_j = np.dot(np.dot(B_ij.T, omega_ij), e_ij).reshape(3,1)
            
            # Update H
            self.H[3*i:3*(i+1),3*i:3*(i+1)] += contr_H_ii
            self.H[3*i:3*(i+1),3*j:3*(j+1)] += contr_H_ij
            self.H[3*j:3*(j+1),3*i:3*(i+1)] += contr_H_ji
            self.H[3*j:3*(j+1),3*j:3*(j+1)] += contr_H_jj

            # Update b 
            self.b[3*i:3*(i+1)] += contr_b_i
            self.b[3*j:3*(j+1)] += contr_b_j
            
    def update_vertices(self, dX):
        """
        Update graph vertices(nodes) with dX = [dx dy dtheta]
        """
        dX = dX.reshape(len(self.vertices), 3)
        
        for i in range(len(self.vertices)):
            self.vertices[i].x += dX[i,0]
            self.vertices[i].y += dX[i,1]
            self.vertices[i].theta += dX[i,2]
    
    def plot_graph(self, title, show_constraints = False):
        """
        Pose Graph Plot
        """
        node_x = np.array([])
        node_y = np.array([])

        for v in self.vertices:
            node_x = np.append(node_x, v.x)
            node_y = np.append(node_y, v.y)

        plt.scatter(node_x, node_y, s = 4, label = 'nodes')
        
        if show_constraints:
            first_edge = True
            for e in self.edges:
                if first_edge:
                    label = 'edges'
                    first_edge = False
                else:
                    label = None

                plt.plot([self.vertices[e.id_from].x, self.vertices[e.id_to].x],
                         [self.vertices[e.id_from].y, self.vertices[e.id_to].y],
                         color = 'red',
                         lw = 0.5,
                         label = label)

        plt.title(title)
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.legend()