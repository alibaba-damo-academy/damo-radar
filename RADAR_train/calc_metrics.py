import re
import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


all_labels = json.load(open('../ckpt/merlin_labels.json'))
dir_path = '../results'


# anatomy-level
csv_path = os.path.join(dir_path, f'checkpoint_ZeroShotResult_anatomy.csv')
results = pd.read_csv(csv_path)
test_items = list(results.columns[1:])
# print('\nanatomy level: ')

all_aucs = []
all_aucs_anatomy = []
all_aucs_whole = []
for i, test_item in enumerate(test_items):
    organ, disease = test_item.split('_')[:2]
    disease = disease.replace('-', '_')

    if disease == 'aortic_aneurysm':
        disease = 'abdominal_aortic_aneurysm'
    if test_item == 'surgically_absent_gallbladder':
        organ, disease = 'gallbladder', 'surgically_absent_gallbladder'
    
    gt_labels = []
    pd_scores = []
    label_json = all_labels[disease]
    
    for file_name, prob in zip(results['file_name'], results[test_item]):
        patient_id = file_name[:-7]
        
        try:
            label = float(label_json[patient_id])
            if label == -1:  # following merlin's protocal
                continue
        except:
            # print(f'{patient_id} no label')  # a little data
            continue
        
        if np.isnan(prob):
            prob = 0.  # not intact organs
        
        gt_labels.append(label)
        pd_scores.append(prob)
    
    # using seg for surgically_absent_gallbladder
    if disease == 'surgically_absent_gallbladder':
        model_pred = np.array(pd_scores)
        model_pred = (model_pred<1000).astype(np.float32).tolist()
        pd_scores = model_pred

    # compute auc
    diease_auc = roc_auc_score(gt_labels, pd_scores)
    print(test_item, np.round(diease_auc, 4))
    all_aucs.append(diease_auc)
    all_aucs_anatomy.append(diease_auc)



# whole-level
# print('\nwhole level: ')
csv_path = os.path.join(dir_path, f'checkpoint_ZeroShotResult_whole.csv')
results = pd.read_csv(csv_path)
test_items = list(results.columns[1:])

for i, test_item in enumerate(test_items):
    disease = test_item
    gt_labels = []
    pd_scores = []
    label_json = all_labels[disease]
    
    for file_name, prob in zip(results['file_name'], results[test_item]):
        patient_id = file_name[:-7]
        
        try:
            label = float(label_json[patient_id])
            if label == -1:  # following merlin's protocal
                continue
        except:
            # print(f'{patient_id} no label')  # a little data
            continue
        
        if np.isnan(prob):
            prob = 0.  # not intact organs
        
        gt_labels.append(label)
        pd_scores.append(prob)
    
    # compute auc
    diease_auc = roc_auc_score(gt_labels, pd_scores)
    print(test_item, np.round(diease_auc, 4))
    all_aucs.append(diease_auc)
    all_aucs_whole.append(diease_auc)
    
print(f'AvgAUC_anatomy: {np.mean(all_aucs_anatomy):.4f}')
print(f'AvgAUC_whole: {np.mean(all_aucs_whole):.4f}')
print(f'AvgAUC: {np.mean(all_aucs):.4f}')

print('done')
