import torch
import pyviz3d.visualizer as viz
import random
import pointops
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from os.path import join
from SegmentAnything3D.util import *
from open3d import *  
from plyfile import PlyData

def generate_palette(n):
    palette = []
    for _ in range(n):
        # Generate random RGB values between 0 and 255
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        # Append the RGB tuple to the palette list
        palette.append((red, green, blue))
    return palette

def assign(n_instance, instance_label, color, pallete):
    for i in range(n_instance):
        color[torch.where(instance_label==i)[0],:]=pallete[i]
    return color

def cal_group(input_dict, new_input_dict, match_inds, ratio=0.5):
    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    group_1[group_1 != -1] += group_0.max() + 1
    
    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # Calculate the group number correspondence of overlapping points
    group_overlap = {}
    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1

    # Update group information for point cloud 1
    for group_i, overlap_count in group_overlap.items():
        # for group_j, count in overlap_count.items():
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]
        total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
        # print(count / total_count)
        if count / total_count >= ratio:
            group_1[group_1 == group_i] = group_j
    return group_1

def pcd_ensemble(org_path, new_path, point, vis_path):
    new_pcd = torch.load(new_path)
    new_pcd = num_to_natural(remove_small_group(new_pcd, 20))
    with open(org_path) as f:
        segments = json.load(f)
        org_pcd = np.array(segments['segIndices'])
    match_inds = [(i, i) for i in range(len(new_pcd))]
    new_group = cal_group(dict(group=new_pcd), dict(group=org_pcd), match_inds)
    print(new_group.shape)
    visualize_partition(point, new_group, vis_path)


# Scene
img_folder = 'lowres_wide'
inputdir = '../../Dataset/iccvw/ChallengeDevelopmentSet/computed_feature'
arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
data_type = "ChallengeDevelopmentSet" # or "ChallengeTestSet"
scene_ids = ['42445173', '42445677', '42445935', '42446478', '42446588'] # Dev Scenes
scene_id = scene_ids[3]
gt_data = '../../Dataset/iccvw/ChallengeDevelopmentSet/pcl/'+scene_id+'.pth'
point_feature_path = inputdir+'/'+scene_id+'_ovseg.pt'

point, color = torch.load(gt_data)
color = (color + 1) * 127.5

ensemble = False
if ensemble:
    # Class agnostic 3D proposals
    tmp = '../../Dataset/iccvw/ChallengeTestSet/versionfinal/final_result/42897564.pth'
    agnostic = '42897564.pth'
    scene_id = tmp[-12:-4]
    gt_data = '../../Dataset/iccvw/ChallengeTestSet/pcl/'+scene_id+'.pth'
    point, color = torch.load(gt_data)
    point = point[::2,:]
    color = color[::2,:]
    # breakpoint()
    color = (color + 1) * 127.5
    # Visualize point cloud      
    v = viz.Visualizer()
    data = torch.load(agnostic)
    conf = data['conf']
    instance = data['masks']
    v = viz.Visualizer()
    v.add_points(f'pcl', point, color.astype(np.float32), point_size=20, visible=True)
    for i in range(10):
        tmp = color.copy()
        tmp[np.where(instance[i]==1)[0],:]=(255,0,0)
        v.add_points(f', sem' + str(i)+': ' +  str(conf[i])[:4], point, tmp, point_size=20, visible=True)
    v.save('viz') 
else:
    tmp = '../../Dataset/iccvw/ChallengeTestSet/versionfinal/final_result/47115452.pth'
    inputdir = '../../Dataset/iccvw/ChallengeTestSet/computed_feature1'
    arkitscenes_root_dir = "../../Dataset/iccvw" #"PATH/TO/ARKITSCENES/DOWNLOAD/DIR"
    data_type = "ChallengeTestSet" # or "ChallengeTestSet"
    scene_id = tmp[-12:-4]
    gt_data = '../../Dataset/iccvw/ChallengeTestSet/pcl/'+scene_id+'.pth'
    point_feature_path = inputdir+'/'+scene_id+'_ovseg.pt'

    point, color = torch.load(gt_data)
    color = (color + 1) * 127.5
    # Visualize point cloud      
    v = viz.Visualizer()
    data = torch.load(tmp)
    instance = torch.tensor(data['ins'])
    sem = torch.tensor(data['sem'])
    score = torch.tensor(data['conf'])
    n_instance = torch.unique(instance).shape[0]
    # breakpoint()
    pal=generate_palette(n_instance)
    v.add_points(f'pcl', point, color.astype(np.float32), point_size=20, visible=True)
    for i in range(n_instance - 1):
        tmp = color.copy()
        tmp[torch.where(instance==i)[0],:]=pal[i]
        label = int(sem[torch.where(instance==i)[0]][0].item())
        if score[i].item() >= 0.8:
            v.add_points(f', sem' + str(i)+': ' +  str(label) + ' ' + str(score[i].item())[:4], point, tmp, point_size=20, visible=True)
    # breakpoint()
    color=assign(n_instance, instance, color, pal)
    v.add_points(f'allpcl', point, color.astype(np.float32), point_size=20, visible=True)
    v.save('viz') 