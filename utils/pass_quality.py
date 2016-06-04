import numpy as np
import os
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
            r += 1
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


def pass_quality(y_test, y_pred):
    correct = 0
    fine = 0

    if 1:#for (dirpath, dirnames, filenames) in os.walk(dir):
        if 1:#for filename in filenames:
            pass_state = 0
            exp_pass = 0
            ref_pass = 0
            exp_group_pass_cnt = 0
            ref_group_pass_cnt = 0
            for i in range(len(y_test)):
                #j = json.loads(line.split('\n')[0])
                #j = line[-1]
                if 1:#j['sensors'][side] != None:
                    if exp_pass != y_pred[i]:
                        exp_pass = y_pred[i]
                        if exp_pass:
                            exp_group_pass_cnt += 1
                    if ref_pass != y_test[i]:
                        ref_pass = y_test[i]
                        if ref_pass:
                            ref_group_pass_cnt += 1

                    if not pass_state:
                        if exp_pass or ref_pass:
                            pass_state = 1
                    else:
                        if not exp_pass and not ref_pass:
                            if exp_group_pass_cnt == 1 and ref_group_pass_cnt == 1:
                                correct += 1
                                fine += 0
                            else:
                                correct += 0
                                fine += max(exp_group_pass_cnt, ref_group_pass_cnt)
                                # print filename
                            ref_group_pass_cnt = 0
                            exp_group_pass_cnt = 0
                            pass_state = 0
                    # print '{} pass_state={}, ref_pass={},
                            # exp_pass={}'.format(j['frame'], pass_state, ref_pass, exp_pass)

    # print 'correct passes:', correct, '\nfine:', fine
    res = 100.0 * correct / (correct + fine)
    # print 'pass quality:', res
    return res, correct, fine


def pass_quality_files(data, result_colunm='res', y_column='y', dir='data/final_data/sensors_logs_correct_selected/'):
    correct = 0
    fine = 0

    for (dirpath, dirnames, filenames) in os.walk(dir):
        for filename in filenames:
            pass_state = 0
            exp_pass = 0
            ref_pass = 0
            exp_group_pass_cnt = 0
            ref_group_pass_cnt = 0
            for line in data.loc[filename].iterrows():
                #j = json.loads(line.split('\n')[0])
                j = line[-1]
                if 1:#j['sensors'][side] != None:
                    if exp_pass != j[result_colunm]:
                        exp_pass = j[result_colunm]
                        if exp_pass:
                            exp_group_pass_cnt += 1
                    if ref_pass != j[y_column]:
                        ref_pass = j[y_column]
                        if ref_pass:
                            ref_group_pass_cnt += 1

                    if not pass_state:
                        if exp_pass or ref_pass:
                            pass_state = 1
                    else:
                        if not exp_pass and not ref_pass:
                            if exp_group_pass_cnt == 1 and ref_group_pass_cnt == 1:
                                correct += 1
                                fine += 0
                            else:
                                correct += 0
                                fine += max(exp_group_pass_cnt, ref_group_pass_cnt)
                                # print filename
                            ref_group_pass_cnt = 0
                            exp_group_pass_cnt = 0
                            pass_state = 0
                    # print '{} pass_state={}, ref_pass={}, exp_pass={}'.format(j['frame'], pass_state, ref_pass, exp_pass)

    print 'correct passes:', correct, '\nfine:', fine
    res = 100.0 * correct / (correct + fine)
    #print 'pass quality:', res
    return res, correct, fine