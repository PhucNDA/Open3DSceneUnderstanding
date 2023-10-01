import csv
import torch 
import numpy as np
import subprocess
from multiprocessing import Pool
import os

scenes = []
with open('queries_test_scenes.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        scenes.append(row)
scenes.pop(0)

def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result

def bash_template(gpu, scene_value, captions_value, rotate):
    bash_command_template = "export PYTHONPATH=/home/tinhn/3D/ICCVW && CUDA_VISIBLE_DEVICES=5 python demo_grounded.py --scene '{scene}' --captions '{captions}' --rotate {rotate} --config ../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint ../../groundingdino_swint_ogc.pth --sam_checkpoint ../../sam_vit_h_4b8939.pth --box_threshold 0.5 --text_threshold 0.25 --device 'cuda' --voxel_size 0.05 --clip_checkpoint ../../ovseg_clip_l_9a1909.pth"
    return bash_command_template.format(scene=scene_value, captions=captions_value, rotate=rotate)

def run(path, arg1, arg2, arg3, arg4, arg5, arg6, arg7):
    arg2 = arg2.replace(' ','_')
    arg3 = arg3.replace(' ','_')
    result = subprocess.run([path, arg1, arg2, arg3, arg4, arg5, str(arg6), str(arg7)])
    return result

script_path = '/home/phucnda/open3d/iccvw_resource/ICCVW/feature_fusion/demo_grounded_bashv3.sh'
# script_path = '/home/phucnda/open3d/iccvw_resource/ICCVW/feature_fusion/openseg.sh'
perscene = True
ind = [8]
if perscene:
    for i in ind:
        print ('----', scenes[i][0],'----')
        bash_command = run(script_path, scenes[i][0], scenes[i][5], scenes[i][6], scenes[i][3], scenes[i][7], 0.4, 0.4)
else:
    # 4gpu
    # interval = [0, 6, 11, 18, 25]
    interval = [0, 4, 8, 12, 16, 20, 25]
    # 3gpu
    # interval = [0,7,16,25]
    # 2gpu
    # interval = [0,12,25]
    # 2gpu
    # interval = [0,1,2,3,4]
    card = 5
    for id in range (8):
        if id!=card:
            continue
        startid = interval[id]
        endid = interval[id + 1]
        for i in range(startid, endid, 1):
            print ('----', scenes[i][0],'----')
            if i in ind:
                continue
            ### Check existence -- temporary solution
            check_existed = False
            if check_existed == True and os.path.exists('../../Dataset/iccvw/ChallengeTestSet/versionfinal/final_result/'+scenes[i][0]+'.pth') == True:
                print('existed ' + scenes[i][0])
                continue
            bash_command = run(script_path, scenes[i][0], scenes[i][5], scenes[i][6], scenes[i][3], scenes[i][7], 0.25, 0.2)
