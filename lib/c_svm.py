

import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
import utils
from utils import *






class SVM:
    def hinge_loss(self, w, X, y, C):

        yz = y * np.dot(X,w) 
        yz = np.maximum(np.zeros_like(yz), (1-yz)) 

        return sum(yz) * C

    def train(self, x, y, x_control, C=1, max_iter=100, lamb=1, gamma=None, apply_fairness_constraints=0, sensitive_attrs=['race'], sensitive_attrs_to_cov_thresh={}):
        if apply_fairness_constraints == 0:
            w = self.train_default(x, y, lamb, C, max_iter)
            return w

        elif apply_fairness_constraints == 1:
            print("Running custom model")

            if gamma is not None and gamma != 0:
                w = minimize(fun=hinge_loss,
                             x0=np.random.rand(x.shape[1], ),
                             args=(x, y, C),
                             method='SLSQP',
                             options={"maxiter": max_iter},
                             constraints=[]
                             )

                old_w = deepcopy(w.x)   
                
                def constraint_gamma_all(w, x, y, C, initial_loss_arr):
                    new_loss = hinge_loss(w, x, y, C)
                    return ((1.0 + gamma) * initial_loss_arr) - new_loss

                def constraint_protected_people(w, x, y):  
                    return np.dot(w, x.T)  
                

                def constraint_unprotected_people(w, old_loss, x, y, C):
                    new_loss = hinge_loss(w, np.array([x]), np.array(y),C)
                    return ((1.0 + gamma) * old_loss) - new_loss
                
                constraints = []
                unconstrained_loss_arr = hinge_loss(w.x, x, y,C)
                predicted_labels = np.sign(np.dot(w.x, x.T))
                for i in range(0, len(predicted_labels)):
                    if predicted_labels[i] == 1.0 and x_control[sensitive_attrs[0]][i] == 1.0:  
                        c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args': (x[i], y[i])})  
                        constraints.append(c)
                    else:
                        c = ({'type': 'ineq', 'fun': constraint_unprotected_people,
                              'args': (unconstrained_loss_arr, x[i], y[i],C)})
                        constraints.append(c)

                def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
                    cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * np.dot(weight_vec, x_in.T)
                    return float(abs(sum(cross_cov))) / float(x_in.shape[0])

                w = minimize(fun=cross_cov_abs_optm_func,
                             x0=old_w,
                             args=(x, x_control[sensitive_attrs[0]]),
                             method='SLSQP',
                             options={"maxiter": 100000},
                             constraints=constraints
                             )

            else:
                constraints = self.get_constraint_list_cov(x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh)
                w = minimize(
                    fun=self.hinge_loss,
                    x0=np.random.rand(x.shape[1], ),
                    args=(x, y, C),
                    method='SLSQP',
                    options={"maxiter": max_iter},
                    constraints=constraints
                )
        return w.x

    def predict(self, X, w):
        return np.sign(np.dot(np.array(X), w))

    def get_constraint_list_cov(self, x_train, y_train, x_control_train, sensitive_attrs, sensitive_attrs_to_cov_thresh):
        constraints = []
        

        for attr in sensitive_attrs:


            attr_arr = x_control_train[attr]
            attr_arr = [int(x) for x in attr_arr]
            attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)
                
            if index_dict is None: # binary attribute
                thresh = sensitive_attrs_to_cov_thresh[attr]

                c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, attr_arr_transformed, thresh, False)})
                constraints.append(c)
            else: 

                for attr_val, ind in index_dict.items():
                    attr_name = attr_val                
                    thresh = sensitive_attrs_to_cov_thresh[attr][attr_name]
                
                    t = attr_arr_transformed[:,ind]
                    c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train,t ,thresh, False)})
                    constraints.append(c)


        return constraints
    
    def train_default(self,X,Y,lamb,C,epochs=500,lr=1):
        n,m=X.shape[1] ,X.shape[0]
        w= np.zeros((n, 1))

        for epoch in range(1,int(epochs)):
            for i,x in enumerate(X):
                x_train=x.reshape(1,-1)
                y_train = Y[i].reshape(1,1)
                y_hat= np.dot(x_train,w) +np.sum(lamb * (w ** 2))
                val=y_train *y_hat
                loss = [0  if val> 1 else 1 -val]
                if loss==0:
                    grad=np.zeros(w.shape)
                    w = w - lr * (grad + 2 * lamb * w * C)
                else:
                    grad=(-y_train*x_train).T
                    w = w - lr * (grad + 2 *lamb * w * C )

        return w






