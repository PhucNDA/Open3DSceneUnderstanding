import torch
import numpy as np
import os
import csv
import open_clip
import shutil
from tqdm import tqdm, trange

if os.path.exists('submission_opensun3d') == False:
    os.mkdir('submission_opensun3d')
    os.mkdir('submission_opensun3d/predicted_masks')
scenes = []
with open('queries_test_scenes.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        scenes.append(row)
scenes.pop(0)
tag = [0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14, 15 ,16, 17, 18, 19, 20, 21, 22, 23, 24]
# thre= {0: 0.88, 1:0.99, 2:0.94, 3:0.95, 4: 0.87, 5:0.74, 6:None, 7:0.99, 8:None,9:None, 10: 0.99, 11: 0.99, 12:0.99, 13: 0.99, 14: 0.99, 15:0.93, 16:0.97, 17: 0.99, 18: 0.99, 19: 0.97, 20:0.89, 21: 0.98, 22:0.99, 23: 0.99, 24: None }
# thre= {0: 0.88, 1:0.99, 2:0.94, 3:0.95, 4: 0.87, 5:0.74, 6:0.9, 7:0.99, 8:0.98, 9:0.80, 10: 0.99, 11: 0.99, 12:0.99, 13: 0.99, 14: 0.99, 15:0.93, 16:0.97, 17: 0.99, 18: 0.99, 19: 0.97, 20:0.89, 21: 0.98, 22:0.99, 23: 0.99, 24: 0.8 }
thre= {0: 0.88, 1:0.99, 2:0.89, 3:0.95, 4: 0.78, 5:0.82, 6:0.89, 7:0.99, 8:0.99, 9:0.46, 10: 0.98, 11: 0.98, 12:0.99, 13: 0.98, 14: 0.99, 15:0.93, 16:0.98, 17: 0.99, 18: 0.9958, 19: 0.97, 20:0.86, 21: 0.83, 22:0.99, 23: 0.99, 24: 0.85 }

for id in trange(len(scenes)):
    if id not in tag:
        continue
    scene_id = scenes[id][0]
    class_names = [scenes[id][5]]
    path = 'submission_opensun3d/'+scene_id + '.txt'
    gt_data = '../../Dataset/iccvw/ChallengeTestSet/pcl/'+scene_id+'.pth'
    instance_path = '../../Dataset/iccvw/ChallengeTestSet/versionfinal/final_result/'+scene_id+'.pth'
    try:
        point, color = torch.load(gt_data)
        inst_result = torch.load(instance_path)
    except:
        continue
    instance = inst_result['ins']
    confidence = inst_result['conf']
    n_instance = torch.unique(instance).shape[0] - 1
    with open('submission_opensun3d/' + scene_id + '.txt', 'a') as file:
        cnt = 0
        for ind in range(n_instance):
            mask = np.array(instance == ind).astype(int)
            score = confidence[ind]
            if (np.where(mask==1)[0].shape[0]<10):
                continue
            if (score.item()<thre[id]):
                continue
            if id == 0 and score.item()>0.90:
                continue
            cnt += 1
            mask_path = 'submission_opensun3d/predicted_masks/' + scene_id + '_' + str(cnt).zfill(3)
            np.savetxt(mask_path + '.txt',  mask, fmt='%d')
            # file.write(mask_path + '.txt' + ' ' + str(score.item()) + '\n')
            file.write(mask_path.replace('submission_opensun3d/','') + '.txt' + ' ' + str(0.75) + '\n')



