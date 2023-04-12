#Load the modules

import numpy as np
from random import seed, shuffle
import A2_loss_function as lf 
from scipy.optimize import minimize # minimizing loss function
from multiprocessing import Pool, Process, Queue
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt 
import sys

#Set random seed 

SEED = 704
seed(SEED)
np.random.seed(SEED)


# The train_model function trains a classifier subject to various fairness constraints. If no constraints are provided, it trains an unaltered classifier. The function takes in several inputs, such as training data (x, y), control features (x_control), the loss function, and flags for applying fairness and accuracy constraints, among others.

def train_model(x, y, x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma=None):

    """
This function trains a model while considering various fairness constraints. If no constraints are provided, it simply trains an unmodified classifier. Example usage can be found in "synthetic_data_demo/decision_boundary_demo.py".

Inputs:

X: (n) x (d+1) numpy array, where n is the number of examples, d is the number of features, and one feature is the intercept.

y: 1-D numpy array with n entries.

x_control: a dictionary {"s": [...]}, where "s" is the sensitive feature name, and the value is a 1-D list with n elements representing the sensitive feature values.

loss_function: the loss function to be optimized, currently implemented for logistic loss, but other functions like hinge loss can be added.

apply_fairness_constraints: optimize accuracy under fairness constraint (0/1 values).

apply_accuracy_constraint: optimize fairness under accuracy constraint (0/1 values).

sep_constraint: apply a fine-grained accuracy constraint, as described in Section 3.3 of arxiv.org/abs/1507.05259v3. See "synthetic_data_demo/decision_boundary_demo.py" for examples.

Note: apply_fairness_constraints and apply_accuracy_constraint cannot both be 1 simultaneously.

sensitive_attrs: ["s1", "s2", ...], a list of sensitive features to apply fairness constraints, each should have a corresponding array in x_control.

sensitive_attrs_to_cov_thresh: the covariance threshold that the classifier should meet (only required when apply_fairness_constraints=1, not needed for the other constraints).

gamma: determines the allowable loss in accuracy when using apply_accuracy_constraint and sep_constraint.
Outputs:

w: the learned weight vector for the classifier.

    """


    assert((apply_accuracy_constraint == 1 and apply_fairness_constraints == 1) == False) # fairness and accuracy constraints cannot be applied at the same time

    max_iter = 100000 # maximum iterations number for the minimization algorithm

    if apply_fairness_constraints == 0:
        constraints = []
    else:
        constraints = get_constraint_list_cov(x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh)      

    if apply_accuracy_constraint == 0: #If accuracy constraints are not applied, it minimizes the provided loss function subject to the fairness constraints (if any) using the Sequential Least Squares Quadratic Programming (SLSQP) optimization algorithm.

        f_args=(x, y)
        w = minimize(fun = loss_function,
            x0 = np.random.rand(x.shape[1],),
            args = f_args,
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = constraints
            )

    else:

        # If accuracy constraints are applied, the function first trains a classifier without any constraints by minimizing the loss function. Then, it computes the unconstrained loss for each training example.
        
        w = minimize(fun = loss_function,
            x0 = np.random.rand(x.shape[1],),
            args = (x, y),
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = []
            )

        old_w = deepcopy(w.x)
        
        #When sep_constraint is set to True, the algorithm will apply individual constraints for each person in the dataset. For individuals belonging to a protected group (e.g., a minority race), the constraint ensures that their classification remains in the positive class. For individuals not in the protected group, the constraint allows a certain level of loss in accuracy, controlled by the parameter gamma. This way, the model can achieve a balance between fairness and accuracy while taking into account the specific requirements for each group of individuals.

        def constraint_gamma_all(w, x, y,  initial_loss_arr):
            
            gamma_arr = np.ones_like(y) * gamma # setting gamma for each individual
            new_loss = loss_function(w, x, y)
            old_loss = sum(initial_loss_arr)
            return ((1.0 + gamma) * old_loss) - new_loss

        def constraint_protected_people(w,x,y): # dont confuse the protected here with the sensitive feature protected/non-protected values -- protected here means that these points should not be misclassified to negative class
            return np.dot(w, x.T) # if this is positive, the constraint is satisfied
        def constraint_unprotected_people(w,ind,old_loss,x,y):
            
            new_loss = loss_function(w, np.array([x]), np.array(y))
            return ((1.0 + gamma) * old_loss) - new_loss

        constraints = []
        predicted_labels = np.sign(np.dot(w.x, x.T))
        unconstrained_loss_arr = loss_function(w.x, x, y, return_arr=True)

        if sep_constraint == True: # separate gamma for different groups
            for i in range(0, len(predicted_labels)):
                if predicted_labels[i] == 1.0 and x_control[sensitive_attrs[0]][i] == 1.0: # for now we are assuming just one sensitive attr for reverse constraint, later, extend the code to take into account multiple sensitive attrs
                    c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args':(x[i], y[i])}) # this constraint makes sure that these people stay in the positive class even in the modified classifier             
                    constraints.append(c)
                else:
                    c = ({'type': 'ineq', 'fun': constraint_unprotected_people, 'args':(i, unconstrained_loss_arr[i], x[i], y[i])})                
                    constraints.append(c)
        else: # same gamma for every individual
            c = ({'type': 'ineq', 'fun': constraint_gamma_all, 'args':(x,y,unconstrained_loss_arr)})
            constraints.append(c)

        def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
            cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * np.dot(weight_vec, x_in.T)
            return float(abs(sum(cross_cov))) / float(x_in.shape[0])


        w = minimize(fun = cross_cov_abs_optm_func,
            x0 = old_w,
            args = (x, x_control[sensitive_attrs[0]]),
            method = 'SLSQP',
            options = {"maxiter":100000},
            constraints = constraints
            )

    try:
        assert(w.success == True)
    except:
        print( "Optimization process failed to converge. Verify the solution provided by the optimizer.")
        print(f"Solution obtained: {w}")

    return w.x



def compute_cross_validation_error(x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh_arr, gamma=None):


    """
    Calculates the cross-validation error for a classifier under different fairness constraints. This function serves as a wrapper for "train_model(...)", and all inputs, except for num_folds, are the same. Refer to train_model(...) for more details.

    Returns lists of train/test accuracy (with each list containing values for all folds), the proportions of various sensitive groups in the positive class (for train and test sets), and the covariance between the sensitive feature and the distance from the decision boundary (for both train and test folds).
    """
    train_folds = []
    test_folds = []
    n_samples = len(y_all)
    train_fold_size = 0.7 # the rest of 0.3 is for testing

    # split the data into folds for cross-validation
    for i in range(0,num_folds):
        perm = range(0,n_samples) # shuffle the data before creating each fold
        shuffle(perm)
        x_all_perm = x_all[perm]
        y_all_perm = y_all[perm]
        x_control_all_perm = {}
        for k in x_control_all.keys():
            x_control_all_perm[k] = np.array(x_control_all[k])[perm]


        x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test = split_into_train_test(x_all_perm, y_all_perm, x_control_all_perm, train_fold_size)

        train_folds.append([x_all_train, y_all_train, x_control_all_train])
        test_folds.append([x_all_test, y_all_test, x_control_all_test])

    def train_test_single_fold(train_data, test_data, fold_num, output_folds, sensitive_attrs_to_cov_thresh):

        x_train, y_train, x_control_train = train_data
        x_test, y_test, x_control_test = test_data

        w = train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
        train_score, test_score, correct_answers_train, correct_answers_test = check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
        
        distances_boundary_test = (np.dot(x_test, w)).tolist()
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
        cov_dict_test = print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)

        distances_boundary_train = (np.dot(x_train, w)).tolist()
        all_class_labels_assigned_train = np.sign(distances_boundary_train)
        correlation_dict_train = get_correlations(None, None, all_class_labels_assigned_train, x_control_train, sensitive_attrs)
        cov_dict_train = print_covariance_sensitive_attrs(None, x_train, distances_boundary_train, x_control_train, sensitive_attrs)

        output_folds.put([fold_num, test_score, train_score, correlation_dict_test, correlation_dict_train, cov_dict_test, cov_dict_train])


        return


    output_folds = Queue()
    processes = [Process(target=train_test_single_fold, args=(train_folds[x], test_folds[x], x, output_folds, sensitive_attrs_to_cov_thresh_arr[x])) for x in range(num_folds)]

    # Run processes
    for p in processes:
        p.start()


    # Get the reuslts
    results = [output_folds.get() for p in processes]
    for p in processes:
        p.join()
    
    
    test_acc_arr = []
    train_acc_arr = []
    correlation_dict_test_arr = []
    correlation_dict_train_arr = []
    cov_dict_test_arr = []
    cov_dict_train_arr = []

    results = sorted(results, key = lambda x : x[0]) # sort w.r.t fold num
    for res in results:
        fold_num, test_score, train_score, correlation_dict_test, correlation_dict_train, cov_dict_test, cov_dict_train = res

        test_acc_arr.append(test_score)
        train_acc_arr.append(train_score)
        correlation_dict_test_arr.append(correlation_dict_test)
        correlation_dict_train_arr.append(correlation_dict_train)
        cov_dict_test_arr.append(cov_dict_test)
        cov_dict_train_arr.append(cov_dict_train)

    
    return test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr



def print_classifier_fairness_stats(acc_arr, correlation_dict_arr, cov_dict_arr, s_attr_name):
    
    correlation_dict = get_avg_correlation_dict(correlation_dict_arr)
    non_prot_pos = correlation_dict[s_attr_name][1][1]
    prot_pos = correlation_dict[s_attr_name][0][1]
    p_rule = (prot_pos / non_prot_pos) * 100.0
    
    print("Accuracy: %0.2f" % (np.mean(acc_arr)))
    print("Protected/non-protected in +ve class: %0.0f%% / %0.0f%%" % (prot_pos, non_prot_pos))
    print("P-rule achieved: %0.0f%%" % (p_rule))
    print("Covariance between sensitive feature and decision from distance boundary: %0.3f" % (np.mean([v[s_attr_name] for v in cov_dict_arr])))
    print()
    return p_rule

def compute_p_rule(x_control, class_labels):

    """ Compute the p-rule based on Doctrine of disparate impact """

    non_prot_all = sum(x_control == 1.0) # non-protected group
    prot_all = sum(x_control == 0.0) # protected group
    non_prot_pos = sum(class_labels[x_control == 1.0] == 1.0) # non_protected in positive class
    prot_pos = sum(class_labels[x_control == 0.0] == 1.0) # protected in positive class
    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all)
    p_rule = (frac_prot_pos / frac_non_prot_pos) * 100.0
    print()
    print("Total data points: %d" % (len(x_control)))
    print("# non-protected examples: %d" % (non_prot_all))
    print("# protected examples: %d" % (prot_all))
    print("Non-protected in positive class: %d (%0.0f%%)" % (non_prot_pos, non_prot_pos * 100.0 / non_prot_all))
    print("Protected in positive class: %d (%0.0f%%)" % (prot_pos, prot_pos * 100.0 / prot_all))
    print("P-rule is: %0.0f%%" % ( p_rule ))
    return p_rule




def add_intercept(x):

    """ Add intercept to the data before linear classification """
    m,n = x.shape
    intercept = np.ones(m).reshape(m, 1) # the constant b
    return np.concatenate((intercept, x), axis = 1)

def check_binary(arr):
    "give an array of values, see if the values are only 0 and 1"
    s = sorted(set(arr))
    if s[0] == 0 and s[1] == 1:
        return True
    else:
        return False

def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    for k in in_arr:
        if str(type(k)) != "<type 'numpy.float64'>" and type(k) != int and type(k) != np.int64:
            print(str(type(k)))
            print("************* ERROR: Input arr does not have integer types")
            return None
        
    in_arr = np.array(in_arr, dtype=int)
    assert(len(in_arr.shape)==1) # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    
    index_dict = {} # value to the column number
    for i in range(0,len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []    
    for i in range(0,len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1 # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict

def check_accuracy(model, x_train, y_train, x_test, y_test, y_train_predicted, y_test_predicted):


    """
Returns the train/test accuracy of the model. 
You can either pass the model (w) or provide the predicted labels (y_predicted).
    """
    if model is not None and y_test_predicted is not None:
        print("Either the model (w) or the predicted labels should be None")
        raise Exception("Either the model (w) or the predicted labels should be None")

    if model is not None:
        y_test_predicted = np.sign(np.dot(x_test, model))
        y_train_predicted = np.sign(np.dot(x_train, model))

    def get_accuracy(y, Y_predicted):
        correct_answers = (Y_predicted == y).astype(int) # will have 1 when the prediction and the actual label match
        accuracy = float(sum(correct_answers)) / float(len(correct_answers))
        return accuracy, sum(correct_answers)

    train_score, correct_answers_train = get_accuracy(y_train, y_train_predicted)
    test_score, correct_answers_test = get_accuracy(y_test, y_test_predicted)

    return train_score, test_score, correct_answers_train, correct_answers_test

def test_sensitive_attr_constraint_cov(model, x_arr, y_arr_dist_boundary, x_control, thresh, verbose):

    
    """
Calculates the covariance between the sensitive attribute value and the distance from the decision boundary. If the model is None, it is assumed that y_arr_dist_boundary contains the distance from the decision boundary. If the model is not None, a dot product of the model and x_arr is computed. For the case of SVM, the distance from the boundary is passed because the intercept is internalized for the class, and the distance is calculated using the project function.

This function returns -1 if the constraint specified by the thresh parameter is not satisfied; otherwise, it returns +1. If the return value is greater than or equal to 0, the constraint is satisfied.
    """

    


    assert(x_arr.shape[0] == x_control.shape[0])
    if len(x_control.shape) > 1: # make sure we just have one column in the array
        assert(x_control.shape[1] == 1)
    
    arr = []
    if model is None:
        arr = y_arr_dist_boundary # simply the output labels
    else:
        arr = np.dot(model, x_arr.T) # the product with the weight vector -- the sign of this is the output label
    
    arr = np.array(arr, dtype=np.float64)


    cov = np.dot(x_control - np.mean(x_control), arr ) / float(len(x_control))

        
    ans = thresh - abs(cov) # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    # ans = thresh - cov # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    if verbose is True:
        print("Covariance is", cov)
        print("Diff is:", ans)
        print()
    return ans

def print_covariance_sensitive_attrs(model, x_arr, y_arr_dist_boundary, x_control, sensitive_attrs):


    """
    reutrns the covariance between sensitive features and distance from decision boundary
    """

    arr = []
    if model is None:
        arr = y_arr_dist_boundary # simplt the output labels
    else:
        arr = np.dot(model, x_arr.T) # the product with the weight vector -- the sign of this is the output label
    

    sensitive_attrs_to_cov_original = {}
    for attr in sensitive_attrs:

        attr_arr = x_control[attr]


        bin_attr = check_binary(attr_arr) # check if the attribute is binary (0/1), or has more than 2 vals
        if bin_attr == False: # if its a non-binary sensitive feature, then perform one-hot-encoding
            attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)

        thresh = 0

        if bin_attr:
            cov = thresh - test_sensitive_attr_constraint_cov(None, x_arr, arr, np.array(attr_arr), thresh, False)
            sensitive_attrs_to_cov_original[attr] = cov
        else: # sensitive feature has more than 2 categorical values            
            
            cov_arr = []
            sensitive_attrs_to_cov_original[attr] = {}
            for attr_val, ind in index_dict.items():
                t = attr_arr_transformed[:,ind]
                cov = thresh - test_sensitive_attr_constraint_cov(None, x_arr, arr, t, thresh, False)
                sensitive_attrs_to_cov_original[attr][attr_val] = cov
                cov_arr.append(abs(cov))

            cov = max(cov_arr)
            
    return sensitive_attrs_to_cov_original


def get_correlations(model, x_test, y_predicted, x_control_test, sensitive_attrs):
    

    """
    returns the fraction in positive class for sensitive feature values
    """

    if model is not None:
        y_predicted = np.sign(np.dot(x_test, model))
        
    y_predicted = np.array(y_predicted)
    
    out_dict = {}
    for attr in sensitive_attrs:

        attr_val = []
        for v in x_control_test[attr]: attr_val.append(v)
        assert(len(attr_val) == len(y_predicted))


        total_per_val = defaultdict(int)
        attr_to_class_labels_dict = defaultdict(lambda: defaultdict(int))

        for i in range(0, len(y_predicted)):
            val = attr_val[i]
            label = y_predicted[i]

            # val = attr_val_int_mapping_dict_reversed[val] # change values from intgers to actual names
            total_per_val[val] += 1
            attr_to_class_labels_dict[val][label] += 1

        class_labels = set(y_predicted.tolist())

        local_dict_1 = {}
        for k1,v1 in attr_to_class_labels_dict.items():
            total_this_val = total_per_val[k1]

            local_dict_2 = {}
            for k2 in class_labels: # the order should be the same for printing
                v2 = v1[k2]

                f = float(v2) * 100.0 / float(total_this_val)


                local_dict_2[k2] = f
            local_dict_1[k1] = local_dict_2
        out_dict[attr] = local_dict_1

    return out_dict



def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs, sensitive_attrs_to_cov_thresh):

    """
    get the list of constraints to be fed to the minimizer
    """

    constraints = []


    for attr in sensitive_attrs:


        attr_arr = x_control_train[attr]
        attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)
                
        if index_dict is None: # binary attribute
            thresh = sensitive_attrs_to_cov_thresh[attr]
            c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, attr_arr_transformed,thresh, False)})
            constraints.append(c)
        else: # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately


            for attr_val, ind in index_dict.items():
                attr_name = attr_val                
                thresh = sensitive_attrs_to_cov_thresh[attr][attr_name]
                
                t = attr_arr_transformed[:,ind]
                c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, t ,thresh, False)})
                constraints.append(c)


    return constraints



def split_into_train_test(x_all, y_all, x_control_all, train_fold_size):

    split_point = int(round(float(x_all.shape[0]) * train_fold_size))
    x_all_train = x_all[:split_point]
    x_all_test = x_all[split_point:]
    y_all_train = y_all[:split_point]
    y_all_test = y_all[split_point:]
    x_control_all_train = {}
    x_control_all_test = {}
    for k in x_control_all.keys():
        x_control_all_train[k] = x_control_all[k][:split_point]
        x_control_all_test[k] = x_control_all[k][split_point:]

    return x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test


def get_avg_correlation_dict(correlation_dict_arr):
    # make the structure for the correlation dict
    correlation_dict_avg = {}
    # print correlation_dict_arr
    for k,v in correlation_dict_arr[0].items():
        correlation_dict_avg[k] = {}
        for feature_val, feature_dict in v.items():
            correlation_dict_avg[k][feature_val] = {}
            for class_label, frac_class in feature_dict.items():
                correlation_dict_avg[k][feature_val][class_label] = []

    # populate the correlation dict
    for correlation_dict in correlation_dict_arr:
        for k,v in correlation_dict.items():
            for feature_val, feature_dict in v.items():
                for class_label, frac_class in feature_dict.items():
                    correlation_dict_avg[k][feature_val][class_label].append(frac_class)

    # now take the averages
    for k,v in correlation_dict_avg.items():
        for feature_val, feature_dict in v.items():
            for class_label, frac_class_arr in feature_dict.items():
                correlation_dict_avg[k][feature_val][class_label] = np.mean(frac_class_arr)

    return correlation_dict_avg



def plot_cov_thresh_vs_acc_pos_ratio(x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs):


    # very the covariance threshold using a range of decreasing multiplicative factors and see the tradeoffs between accuracy and fairness
    it = 0.05
    cov_range = np.arange(1.0, 0.0-it, -it).tolist()
    if apply_accuracy_constraint == True:
        if sep_constraint == False:
            it = 0.1
            cov_range = np.arange(0.0, 1.0 + it, it).tolist()
        if sep_constraint == True:
            cov_range =  [0,1,5,10,20,50,100,500,1000]

    
    positive_class_label = 1 # positive class is +1
    train_acc = []
    test_acc = []
    positive_per_category = defaultdict(list) # for each category (male / female), the frac of positive

    # first get the original values of covariance in the unconstrained classifier -- these original values are not needed for reverse constraint    
    test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr = compute_cross_validation_error(x_all, y_all, x_control_all, num_folds, loss_function, 0, apply_accuracy_constraint, sep_constraint, sensitive_attrs, [{} for i in range(0,num_folds)], 0)

    for c in cov_range:
        print("LOG: testing for multiplicative factor: %0.2f" % c)
        sensitive_attrs_to_cov_original_arr_multiplied = []
        for sensitive_attrs_to_cov_original in cov_dict_train_arr:
            sensitive_attrs_to_cov_thresh = deepcopy(sensitive_attrs_to_cov_original)
            for k in sensitive_attrs_to_cov_thresh.keys():
                v = sensitive_attrs_to_cov_thresh[k]
                if type(v) == type({}):
                    for k1 in v.keys():
                        v[k1] = v[k1] * c
                else:
                    sensitive_attrs_to_cov_thresh[k] = v * c
            sensitive_attrs_to_cov_original_arr_multiplied.append(sensitive_attrs_to_cov_thresh)


        test_acc_arr, train_acc_arr, correlation_dict_test_arr, correlation_dict_train_arr, cov_dict_test_arr, cov_dict_train_arr  = compute_cross_validation_error(x_all, y_all, x_control_all, num_folds, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_original_arr_multiplied, c)
        test_acc.append(np.mean(test_acc_arr))


        correlation_dict_train = get_avg_correlation_dict(correlation_dict_train_arr)
        correlation_dict_test = get_avg_correlation_dict(correlation_dict_test_arr)
        
        # just plot the correlations for the first sensitive attr, the plotting can be extended for the other values, but as a proof of concept, we will jsut show for one
        s = sensitive_attrs[0]    
        
        for k,v in correlation_dict_test[s].items():
            if v.get(positive_class_label) is None:
                positive_per_category[k].append(0.0)
            else:
                positive_per_category[k].append(v[positive_class_label])
    
    positive_per_category = dict(positive_per_category)
    
    p_rule_arr = (np.array(positive_per_category[0]) / np.array(positive_per_category[1])) * 100.0
    

    ax = plt.subplot(2,1,1)
    plt.plot(cov_range, positive_per_category[0], "-o" , color="green", label = "Protected")
    plt.plot(cov_range, positive_per_category[1], "-o", color="blue", label = "Non-protected")
    ax.set_xlim([min(cov_range), max(cov_range)])
    plt.xlabel('Multiplicative loss factor')
    plt.ylabel('Perc. in positive class')
    if apply_accuracy_constraint == False:
        plt.gca().invert_xaxis()
        plt.xlabel('Multiplicative covariance factor (c)')
    ax.legend()

    ax = plt.subplot(2,1,2)
    plt.scatter(p_rule_arr, test_acc, color="red")
    ax.set_xlim([min(p_rule_arr), max(max(p_rule_arr), 100)])
    plt.xlabel('P% rule')
    plt.ylabel('Accuracy')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.show()


def get_line_coordinates(w, x1, x2):
    y1 = (-w[0] - (w[1] * x1)) / w[2]
    y2 = (-w[0] - (w[1] * x2)) / w[2]    
    return y1,y2