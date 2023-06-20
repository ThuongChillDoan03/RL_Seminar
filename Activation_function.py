import numpy as np
from numba import njit, jit_module, jit 

@njit()
def Identity(x):
    return x

@njit()
def BinaryStep(x):
    x[x>=0] = 1.0
    x[x<0] = 0.0
    return x

@njit()
def Sigmoid(x):
    return 1.0 / (1.0 + np.e**(-x))

@njit()
def NegativePositiveStep(x):
    x[x>=0] = 1.0
    x[x<0] = -1.0
    return x

@njit()
def Tanh(x):
    return (np.e**(x) - np.e**(-x)) / (np.e**(x) + np.e**(-x))

@njit()
def ReLU(x):
    return x * (x>0)

@njit()
def LeakyReLU(x):
    x[x<0] *= 0.01
    return x

@njit()
def PReLU(x, a=0.5):
    x[x<0] *= 0.5
    return x

@njit()
def Gaussian(x):
    return np.e**(-x**2)

@njit()
def id_function(id, res_mat, Identity, BinaryStep, Sigmoid, NegativePositiveStep, 
                                            Tanh, ReLU, LeakyReLU, PReLU, Gaussian):
    if id == 0: return Identity(res_mat)
    elif id == 1: return BinaryStep(res_mat)
    elif id == 2: return Sigmoid(res_mat)
    elif id == 3: return NegativePositiveStep(res_mat)
    elif id == 4: return Tanh(res_mat)
    elif id == 5: return ReLU(res_mat)
    elif id == 6: return LeakyReLU(res_mat)
    elif id == 7: return PReLU(res_mat)
    else: return Gaussian(res_mat)

@njit()
def neural_network(res_mat, data, list_action):
    for i in range(len(data)):
        if i % 2 == 0:
            res_mat = np.dot(res_mat, data[i])
            max_x = np.max(np.abs(res_mat))
            max_x_1 = max_x/25
            res_mat = res_mat / max_x_1
        else:
            id = int(data[i][0][0])
            # res_mat = list_activation_function[id](res_mat)
            res_mat = id_function(id, res_mat, Identity, BinaryStep, Sigmoid, 
                        NegativePositiveStep, Tanh, ReLU, LeakyReLU, PReLU, Gaussian)
    res_arr = res_mat[list_action]
    arr_max = np.where(res_arr == np.max(res_arr))[0]
    action_max_idx = np.random.choice(arr_max)
    return list_action[action_max_idx]

