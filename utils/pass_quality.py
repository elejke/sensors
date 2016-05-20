import numpy as np
import itertools

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

    # print 'r=', r, 'err=', error

    return float(r)/(r + error), r, error, (error_xs, missed)

def approx_pq_dbg(y_pass, y_sig):
    """
    Good approximation for pass_quality function
    -------------
    Arguments:
        y_pass:
        y_sig:
    """
    flag = 0
    error_xs = []
    r = 0
    height = y_pass[0] + y_sig[0]
    current_error = y_pass[0] + y_sig[0]
    error = 0

    diffs = np.concatenate([[0], np.diff(y_pass + y_sig)])
    for (i, (s, s_abs)) in enumerate(zip(diffs, np.abs(diffs))):
        height += s
        if gap(i, y_pass, y_sig):
            current_error = 1
            r += 1 #experiment
        else:
            current_error += s_abs
        if height > 0:
            flag = 1
        if height == 0:
            if current_error * (current_error != 4):
                error_xs.append([i, (current_error * (current_error != 4) + 1) / 3])
            error += (current_error * (current_error != 4) + 1) / 3
            r += flag
            flag = 0
            current_error = 0

    # print 'r=', r, 'err=', error

    return float(r)/(r + error), r, error, error_xs


def pass_quality(y_true, y_pred):
    right_pass = 0 # number of right detected passes
    height = 0
    error = 0
    current_error = 0
    diffs = np.concatenate([[0], np.diff(y_true + y_pred)])
    for s, s_abs in zip(diffs, np.abs(diffs)):
        height += s
        current_error += s_abs
        if height == 0:
            error += (current_error * (current_error != 4) + 1) / 3
            right_pass += (current_error == 4)
            current_error = 0

    return float(right_pass)/(right_pass + error)


def pass_quality_(y_pass, y_sig):
    """ Quality function for pass detection accuracy
    described here: http://itas2015.iitp.ru/pdf/1570161751.pdf
    =============
    Arguments:
        y_pass: array-like, list of ints;
        referenced pass values {0, 1}
        y_sig: array-like, list of ints;
        Predicted pass values {0, 1}
    returns:
        pq: float:
        A number between 0 and 1
    """
    # initial state:
    errors = []

    state = y_pass[0] * 1 + y_sig[0] * 2
    r, err, l, k = 0, 0, 0, 0

    l_, k_cur = y_pass[0], y_sig[0]

    for num, (pas, sig) in enumerate(itertools.izip(y_pass, y_sig), start=1):
        # state processing:
        if state == 0:
            state = pas * 1 + sig * 2
            k_cur = sig
            l_ = pas
        elif state == 1:
            state = pas * 1 + sig * 2
            if state == 0:
                l = l_
                k = k_cur
                l_ = 0
                k_cur = 0
            elif state == 2:
                l = l_
                k = k_cur
                l_ = 0
                k_cur = 1
            elif state == 3:
                l_ = 1
                k_cur = 1
        elif state == 2:
            state = pas * 1 + sig * 2
            if state == 0:
                l = l_
                k = k_cur
            elif state == 1:
                l, k = max(l_, k_cur), 0
                l_ = 1
                k_cur = 0
            elif state == 3:
                l_ = 1
                k_cur = 1
        elif state == 3:
            if pas == 0 and sig == 0:
                state = 6
            else:
                state += (pas * 1 + sig * 2) % 3
        elif state == 4:
            if pas == 0:
                if sig == 0:
                    state = 6
                else:
                    state = 2
                    if l_ == 1 and k_cur == 1:
                        r += 1
                    l, k = max(l_, k_cur), 0
                    k_cur = 1
                    l_ = 0
            else:
                if sig == 1:
                    state = 3
                    k_cur += 1
        elif state == 5:
            if sig == 0:
                if pas == 0:
                    state = 6
                else:
                    state = 1
                    if l_ == 1 and k_cur == 1:
                        r += 1
                    l, k = max(l_, k_cur), 0
                    k_cur = 0
                    l_ = 1
            else:
                if pas == 1:
                    state = 3
                    l_ += 1
        elif state == 6:
            if l_ == 1 and k_cur == 1:
                l, k = 0, 0
                r += 1
            else:
                l, k = max(l_, k_cur), 0
            state, l_, k_cur = 0, 0, 0

        err += l + k
        l, k = 0, 0

    if state in {3, 4, 5}:
        if l_ != 1 or k_cur != 1:
            err += max(l_, k_cur)
            errors.append(num)
        else:
            r += 1
    else:
        err += max(l_, k_cur)
        errors.append(num)

    # print 'r=', r, 'err=', err
    return float(r)/(r + err), r, err, errors

