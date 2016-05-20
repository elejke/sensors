import numpy as np
# import itertools


def gap(i, y_p, y_s):
    return y_p[i-1] == 1 and y_s[i-1] == 0 and y_p[i] == 0 and y_s[i] == 1 or\
            y_p[i-1] == 0 and y_s[i-1] == 1 and y_p[i] == 1 and y_s[i] == 0


def approx_pq(y_pass, y_sig):
    """
    Good approximation for pass_quality function
    -------------
    Arguments:
        y_pass:
        y_sig:
    """

    error_xs = []
    missed = 0
    r = 0.00000001
    height = y_pass[0] + y_sig[0]
    current_error = y_pass[0] + y_sig[0]
    error = 0.0000000000001

    diffs = np.concatenate([[0], np.diff(y_pass + y_sig)])
    for (i, (s, s_abs)) in enumerate(zip(diffs, np.abs(diffs))):
        height += s
        if gap(i, y_pass, y_sig):
            current_error = 1
            r+=1
        else:
            current_error += s_abs

        if height == 0:
            if current_error * (current_error != 4):
                error_xs.append([i, (current_error * (current_error != 4) + 1) / 3])
            error += (current_error * (current_error != 4) + 1) / 3
            r += (current_error == 4)
            missed += (current_error == 2) * (y_pass[i-1] == 1)
            current_error = 0

    return float(r)/(r + error), r, error, (error_xs, missed)