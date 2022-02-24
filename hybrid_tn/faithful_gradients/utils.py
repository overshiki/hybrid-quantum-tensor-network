import numpy as np
def pr2array(pr):
    parameters = []
    k_list = []
    for k in pr.keys():
        k_list.append(k)
        parameters.append(pr[k])

    parameters = np.array(parameters)
    return parameters, k_list

def array2pr(parameters, k_list):
    _pr = {}
    for k, p in zip(k_list, parameters.tolist()):
        _pr[k] = p
    pr = PR(_pr)
    return pr