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
map_radar_merlin = {'主动脉_主动脉瘤': 'abdominal_aortic_aneurysm', '主动脉_粥样硬化': 'atherosclerosis', '大肠_粘膜下水肿': 'submucosal_edema', '大肠_阑尾炎': 'appendicitis', '小肠_梗阻': 'bowel_obstruction', '心脏_主动脉瓣钙化': 'aortic_valve_calcification', '心脏_心影（脏）增大': 'cardiomegaly', '肝_肝内胆管扩张': 'biliary_ductal_dilation', '肝_肝大': 'hepatomegaly', '肝_脂肪肝': 'hepatic_steatosis', '肺_胸腔积液': 'pleural_effusion', '肺_膨胀不全': 'atelectasis', '肾_低密度影': 'renal_hypodensities', '肾_囊肿': 'renal_cyst', '肾_肾积水': 'hydronephrosis', '胆囊_结石': 'gallstones', '胰腺_萎缩': 'pancreatic_atrophy', '脾_脾大': 'splenomegaly', '腰椎_骨折': 'fracture', '食管_裂孔疝': 'hiatal_hernia', '胆囊_术后胆囊缺失': 'surgically_absent_gallbladder'}

csv_path = '../results/RADAR_infer_results_MerlinTestset.csv'
print('--> csv_file: ', csv_path)
results = pd.read_csv(csv_path)
test_items = list(results.columns[1:])

all_aucs = []
for i, test_item in enumerate(test_items):
    organ, disease = test_item.split('_')
    gt_labels = []
    pd_scores = []
    label_json = all_labels[map_radar_merlin[f'{organ}_{disease}']]
    
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

    # using seg for 胆囊_术后胆囊缺失/surgically_absent_gallbladder
    if test_item == '胆囊_术后胆囊缺失':  # surgically_absent_gallbladder
        model_pred = np.array(pd_scores)
        model_pred = (model_pred<1000).astype(np.float32).tolist()
        pd_scores = model_pred
    
    # compute auc
    diease_auc = roc_auc_score(gt_labels, pd_scores)
    print(test_item, map_radar_merlin[test_item], diease_auc)
    all_aucs.append(diease_auc)
    
print(f'AvgAUC: {np.mean(all_aucs):.4f}')
print('done')


