import json
import numpy as np
import sys
from ancillary import list_recursive


def remote_1(args):

    input_list = args["input"]

    # combine the partial square roots from the local sites
    sums = 0
    for site in input_list:
        Ps = np.array(input_list[site]["psr"])
        sums = sums + np.dot(Ps, Ps.T)

    sums = sums / len(input_list)

    # compute SVD
    u, s, v = np.linalg.svd(sums)
    K = 2
    u = u[:, :K]

    # compute captured energy in the top-K subspace
    cov = np.array(input_list[site]["cov"])
    en = np.trace(np.dot(np.dot(u.T, cov), u))

    computation_output = {"output": {"en": en}, "success": True}
    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
