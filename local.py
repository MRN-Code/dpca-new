import json
import os
import sys
import numpy as np


def listRecursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in listRecursive(v, key):
                yield found
        if k == key:
            yield v


def local_1(args):

    input_list = args["input"]
    myFile = input_list["samples"]
    
    # read local data
    filename = os.path.join(args["state"]["baseDirectory"], myFile)
    tmp = np.load(filename)
    Xs = tmp['arr_0']
    K = tmp['arr_3']
    cov = tmp['arr_2']
    R = 2 * K
    
    # compute partial square root
    Ns = Xs.shape[1]
    Cs = (1/Ns) * np.dot(Xs, Xs.T)
    U, S, V = np.linalg.svd(Cs)
    U = U[:, :R]
    S = S[:R]
    tmp = np.diag(np.sqrt(S))
    P = np.dot(U, tmp) 
    
    # dump outputs
    computation_output = {
        "output": {
            "psr": P.tolist(),
            "cov": cov.tolist(),
            "computation_phase": 'local_1'
        }
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

#    parsed_args = json.loads(sys.argv[1])
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
