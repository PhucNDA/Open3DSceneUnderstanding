import torch
import numpy as np
import os
from plyfile import PlyData
from tqdm import tqdm
# datapath = '../Dataset/iccvw/ChallengeDevelopmentSet/42445173/42445173_3dod_mesh.ply'
datapath = '../Dataset/iccvw/ChallengeTestSet/'
pthpath = os.path.join(datapath, 'pcl')
# os.mkdir(pthpath)

scenes = sorted(os.listdir(datapath))
# scenes.remove('pcl')
# scenes.remove('metadata.csv')
for scene in tqdm(scenes):
    scene_dir = os.path.join(datapath,scene)
    ply_file = os.path.join(scene_dir, [s for s in os.listdir(scene_dir) if s.endswith('.ply')][0])
    plydata = PlyData.read(ply_file)
    x_values = np.array(plydata['vertex']['x']).reshape(-1,1)
    y_values = np.array(plydata['vertex']['y']).reshape(-1,1)
    z_values = np.array(plydata['vertex']['z']).reshape(-1,1)
    coord = np.concatenate((x_values,y_values,z_values),axis=1)

    r_values = np.array(plydata['vertex']['red']).reshape(-1,1)
    g_values = np.array(plydata['vertex']['green']).reshape(-1,1)
    b_values = np.array(plydata['vertex']['blue']).reshape(-1,1)
    color = np.concatenate((r_values,g_values,b_values),axis=1) / 127.5 - 1
    torch.save((coord, color), os.path.join(pthpath, scene) + '.pth')