import json
import os.path
import argparse

parser = argparse.ArgumentParser(description='Computes pass quality')
parser.add_argument('input_folder', metavar='folder_with_logs', type=str, help='input path to folder with sensor logs')
args = parser.parse_args()

correct = 0
fine = 0

for (dirpath, dirnames, filenames) in os.walk(args.input_folder):
    for filename in filenames:
        sensors_log_name = os.path.join(args.input_folder, filename)
        if filename.find('left') > 0:
            side = 'left'
        elif filename.find('right') > 0:
            side = 'right'
        with open(sensors_log_name, 'r') as f:
            pass_state = False
            exp_pass = False
            ref_pass = False
            exp_group_pass_cnt = 0
            ref_group_pass_cnt = 0

            for line in f:
                j = json.loads(line.split('\n')[0])
                if j['sensors'][side] != None:
                    if exp_pass != j['sensors'][side]['pass']:
                        exp_pass = j['sensors'][side]['pass']
                        if exp_pass:
                            exp_group_pass_cnt += 1
                    if ref_pass != j['sensors'][side]['ref_pass']:
                        ref_pass = j['sensors'][side]['ref_pass']
                        if ref_pass:
                            ref_group_pass_cnt += 1

                    if not pass_state:
                        if exp_pass or ref_pass:
                            pass_state = True
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
                            pass_state = False
                    # print '{} pass_state={}, ref_pass={}, exp_pass={}'.format(j['frame'], pass_state, ref_pass, exp_pass)

print 'correct passes:', correct, '\nfine:', fine
res = 100.0 * correct / (correct + fine)
print 'pass quality:', res
