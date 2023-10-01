import os
import torch
import glob
import math
import numpy as np
import nltk
import re
from tensorflow import io
import tensorflow.compat.v1 as tf

def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
                    intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic

def save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id, args):
    '''Save features.'''

    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points

        rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

        mask_entire = torch.zeros(n_points, dtype=torch.bool)
        mask_entire[rand_ind] = True
        mask = torch.zeros(n_points, dtype=torch.bool)
        mask[point_ids] = True
        mask_entire = mask_entire & mask

        torch.save({"feat": feat_bank[mask_entire].half().cpu(),
                    "mask_full": mask_entire
        },  os.path.join(out_dir, scene_id +'_%d.pt'%(n)))
        print(os.path.join(out_dir, scene_id +'_%d.pt'%(n)) + ' is saved!')


class PointCloudToImageMapper(object):
    def __init__(self, image_dim,
            visibility_threshold=0.25, cut_bound=0, intrinsics=None):
        
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        # if self.intrinsics is not None: # global intrinsics
        #     intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int) # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < self.image_dim[0]-self.cut_bound) \
                    * (pi[1] < self.image_dim[1]-self.cut_bound)
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= self.vis_thres * depth_cur
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2]>0 # make sure the depth is in front
            inside_mask = front_mask*inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T


def obtain_intr_extr_matterport(scene):
    '''Obtain the intrinsic and extrinsic parameters of Matterport3D.'''

    img_dir = os.path.join(scene, 'color')
    pose_dir = os.path.join(scene, 'pose')
    intr_dir = os.path.join(scene, 'intrinsic')
    img_names = sorted(glob.glob(img_dir+'/*.jpg'))

    intrinsics = []
    extrinsics = []
    for img_name in img_names:
        name = img_name.split('/')[-1][:-4]

        extrinsics.append(np.loadtxt(os.path.join(pose_dir, name+'.txt')))
        intrinsics.append(np.loadtxt(os.path.join(intr_dir, name+'.txt')))

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    img_names = np.asarray(img_names)

    return img_names, intrinsics, extrinsics

def get_matterport_camera_data(data_path, locs_in, args):
    '''Get all camera view related infomation of Matterport3D.'''

    # find bounding box of the current region
    bbox_l = locs_in.min(axis=0)
    bbox_h = locs_in.max(axis=0)

    building_name = data_path.split('/')[-1].split('_')[0]
    scene_id = data_path.split('/')[-1].split('.')[0]

    scene = os.path.join(args.data_root_2d, building_name)
    img_names, intrinsics, extrinsics = obtain_intr_extr_matterport(scene)

    cam_loc = extrinsics[:, :3, -1]
    ind_in_scene = (cam_loc[:, 0] > bbox_l[0]) & (cam_loc[:, 0] < bbox_h[0]) & \
                    (cam_loc[:, 1] > bbox_l[1]) & (cam_loc[:, 1] < bbox_h[1]) & \
                    (cam_loc[:, 2] > bbox_l[2]) & (cam_loc[:, 2] < bbox_h[2])

    img_names_in = img_names[ind_in_scene]
    intrinsics_in = intrinsics[ind_in_scene]
    extrinsics_in = extrinsics[ind_in_scene]
    num_img = len(img_names_in)

    # some regions have no views inside, we consider it differently for test and train/val
    if args.split == 'test' and num_img == 0:
        print('no views inside {}, take the nearest 100 images to fuse'.format(scene_id))
        #! take the nearest 100 views for feature fusion of regions without inside views
        centroid = (bbox_l+bbox_h)/2
        dist_centroid = np.linalg.norm(cam_loc-centroid, axis=-1)
        ind_in_scene = np.argsort(dist_centroid)[:100]
        img_names_in = img_names[ind_in_scene]
        intrinsics_in = intrinsics[ind_in_scene]
        extrinsics_in = extrinsics[ind_in_scene]
        num_img = 100

    img_names_in = img_names_in.tolist()

    return intrinsics_in, extrinsics_in, img_names_in, scene_id, num_img

def NMS(bounding_boxes, confidence_score, label, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []
    # Bounding boxes
    boxes = np.array(bounding_boxes)
    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    # Confidence scores of bounding boxes
    score = np.array(confidence_score)
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_label = []
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    # Sort by confidence score of bounding boxes
    order = np.argsort(score)
    # Iterate bounding boxes
    # breakpoint()
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]
        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_label.append(label[index])
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_label

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def mask_nms(masks, scores, iou_threshold=0.5):
    # Sort masks based on scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    masks = masks[sorted_indices]
    scores = scores[sorted_indices]

    selected_indices = []

    while len(sorted_indices) > 0:
        current_mask = masks[0]
        current_score = scores[0]
        selected_indices.append(sorted_indices[0])

        sorted_indices = sorted_indices[1:]
        masks = masks[1:]
        scores = scores[1:]

        ious = [calculate_iou(current_mask, masks[i]) for i in range(len(masks))]
        ious = np.array(ious)

        overlapping_indices = np.where(ious > iou_threshold)[0]
        sorted_indices = np.delete(sorted_indices, overlapping_indices)
        masks = np.delete(masks, overlapping_indices, axis=0)
        scores = np.delete(scores, overlapping_indices)

    return selected_indices

def heuristic_nounex(caption, with_preposition):
    # NLP processing
    #nltk.set_proxy('http://proxytc.vingroup.net:9090/')
    #nltk.download("popular", quiet=True)
    #nltk.download("universal_tagset", quiet=True)
    if with_preposition:
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                {<NBAR>} # If pattern is not found, just a single NBAR is ok
        """
    else:
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>} # If pattern is not found, just a single NBAR is ok
        """
    tokenized = nltk.word_tokenize(caption)
    chunker = nltk.RegexpParser(grammar)

    chunked = chunker.parse(nltk.pos_tag(tokenized))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    if current_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)

    return continuous_chunk

def get_nouns(caption):
    caption_words = []
    caption_words.extend(heuristic_nounex(caption, True))
    caption_words.extend(heuristic_nounex(caption, False))
    result = []
    for word in list(set(caption_words)):
        result.append(word.strip())
    
    tokenized_components = re.findall(r'\b\w+\b', caption)
    ordered_components = [component for component in result if component in tokenized_components]
    def custom_sort(item):
        return ordered_components.index(item)
    sorted_list = sorted(result, key=custom_sort)
    
    return sorted_list

def rotate_3d_feature_vector_anticlockwise_90(feature_vector):
    rotated_vector = feature_vector.permute(1, 0, 2)
    rotated_vector = torch.flip(rotated_vector, dims=(0,))

    return rotated_vector

def rotate_3db_feature_vector_anticlockwise_90(feature_vector):
    feature_vector = feature_vector.permute(0, 2, 3, 1)
    rotated_vector = feature_vector.permute(0, 2, 1, 3)
    rotated_vector = torch.flip(rotated_vector, dims=(1,))
    
    return rotated_vector.permute(0, 3, 1, 2)

def read_bytes(path):
    '''Read bytes for OpenSeg model running.'''

    with io.gfile.GFile(path, 'rb') as f:
        file_bytes = f.read()
    return file_bytes

def extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=None, regional_pool=True):
    '''Extract per-pixel OpenSeg features.'''

    # load RGB image
    np_image_string = read_bytes(img_dir)
    # run OpenSeg
    results = openseg_model.signatures['serving_default'](
            inp_image_bytes=tf.convert_to_tensor(np_image_string),
            inp_text_emb=text_emb)
    img_info = results['image_info']
    crop_sz = [
        int(img_info[0, 0] * img_info[2, 0]),
        int(img_info[0, 1] * img_info[2, 1])
    ]
    if regional_pool:
        image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
    else:
        image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
    if img_size is not None:
        feat_2d = tf.cast(tf.image.resize_nearest_neighbor(
            image_embedding_feat, img_size, align_corners=True)[0], dtype=tf.float16).numpy()
    else:
        feat_2d = tf.cast(image_embedding_feat[[0]], dtype=tf.float16).numpy()

    feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1)

    return feat_2d