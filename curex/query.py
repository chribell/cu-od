import pandas as pd

# ground truth (attacks per sec)
attacks = pd.read_csv('./attacks.csv', names=['second', 'num_of_attacks'], header=None)
# outliers (output by GPU)
outliers = pd.read_csv('./outliers.csv', names=['second'], header=None)

# seconds in outliers are starting from 1, subtract to ensure correctness
outliers['second'] = outliers['second'] - 1
attacks.index = attacks['second']

tp = 0
fp = 0

for i, row in outliers.iterrows():
    # if num_of_attacks greater than zero then tp else fp
    if attacks.loc[row.second].num_of_attacks > 0:
        tp += 1
    else:
        fp += 1
        
actual_attacks = attacks[attacks.num_of_attacks > 0]
fn = len(actual_attacks) - tp

precision = (tp / (tp + fp)) * 100
recall = (tp / (tp + fn)) * 100

print('Seconds with attack: ', len(actual_attacks))
print('Outliers: ', len(outliers))
print('TP: ', tp)
print('FP: ', fp)
print('FN: ', fn)
print('Precision: ', "{:.2f}".format(precision))
print('Recall: ', "{:.2f}".format(recall))

print(len(outliers), tp, fp, fn, precision, recall)

