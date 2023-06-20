import numpy as np
from numba import njit, jit_module


@njit()
def data_to_layer(state,data0, data1):
    state = np.dot(state,data0)
    state *= state > 0
    active = state > 0
    layer1 = data1.flatten() * active
    return layer1
 
@njit()
def test2(state,file_per_2):
    layer = np.zeros(getActionSize())
    for id in range(len(file_per_2[0])):
        layer += data_to_layer(state,file_per_2[0][id], file_per_2[1][id])
    base = np.zeros(getActionSize())
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    for act in actions:
        base[act] = 1
    layer *= base
    base += layer
    action = np.random.choice(np.where(base == np.max(base))[0])
    return action

###########################################################

@njit()
def basic_act(state,base):
    actions = getValidActions(state)
    actions = np.where(actions == 1)[0]
    for act in base:
        if act in actions:
            return act
    ind = np.random.randint(len(actions))
    action = actions[ind]
    return action

@njit()
def test2(state,file_per_2):
    action = basic_act(state,file_per_2)
    return action

###########################################################

@njit()
def advance_act(state,data):
    for id in range(len(data[1])):
        x = data[1][id].reshape(len(data[1][id]), 1)
        mt = np.dot(state,x)
        if mt[0] <= 0:
            action = basic_act(state,data[0][id-1])
            return int(action)
        else:
            action = basic_act(state,data[0][id])
            return int(action)
    return np.random.choice(np.where(getValidActions(state) == 1)[0])

@njit()
def test2(state, file_per_2):
    action = advance_act(state,file_per_2)
    return action