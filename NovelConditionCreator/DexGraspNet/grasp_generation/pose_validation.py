import os

os.chdir(os.path.dirname(__file__))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import shutil
import numpy as np
import torch
import math
import json
import pymeshlab

from utils.hand_model import HandModel
from utils.object_model import ObjectModel

# prepare arguments

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--data_root_path', type=str, default='../data/meshdata_dexycb')
parser.add_argument(
    '--object_code_list',
    default=[
        '002_master_chef_can',
        # '003_cracker_box',
        # '004_sugar_box',
        # '005_tomato_soup_can',
        # '006_mustard_bottle',
        # '007_tuna_fish_can',
        # '008_pudding_box',
        # '009_gelatin_box',
        # '010_potted_meat_can',
        # '011_banana',
        # '019_pitcher_base',
        # '021_bleach_cleanser',
        # '024_bowl',
        # '025_mug',
        # '035_power_drill',
        # '036_wood_block',
        # '037_scissors',
        # '040_large_marker',
        # '051_large_clamp',
        # '052_extra_large_clamp',
        # '061_foam_brick',
    ],
    type=list)
parser.add_argument('--on_table', action='store_true')
parser.add_argument('--name', default='demo', type=str)
parser.add_argument('--batch_size', default=300, type=int)

args = parser.parse_args()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def plane2pose(plane_parameters):
    r3 = plane_parameters[:3]
    r2 = torch.zeros_like(r3)
    r2[0], r2[1], r2[2] = (-r3[1], r3[0], 0) if r3[2] * r3[2] <= 0.5 else (-r3[2], 0, r3[0])
    r1 = torch.cross(r2, r3)
    pose = torch.zeros([4, 4], dtype=torch.float, device=plane_parameters.device)
    pose[0, :3] = r1
    pose[1, :3] = r2
    pose[2, :3] = r3
    pose[2, 3] = plane_parameters[3]
    pose[3, 3] = 1
    return pose


def check_intersection(data_dict):
    object_code = data_dict['object_code']
    if args.on_table:
        plane = torch.tensor(data_dict['plane'], dtype=torch.float,
                             device=device)  # plane parameters in object reference frame: (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
        pose = plane2pose(plane)  # 4x4 homogeneous transformation matrix from object frame to world frame
        pose = pose.detach().cpu().numpy()
    else:
        pose = None
    qpos = data_dict['qpos']
    hand_pose = torch.concat([torch.tensor(qpos[key], dtype=torch.float, device=device) for key in ['trans', 'rot', 'thetas']])
    if 'contact_point_indices' in data_dict:
        contact_point_indices = torch.tensor(data_dict['contact_point_indices'], dtype=torch.long, device=device)

    # hand model
    hand_model = HandModel(mano_root='mano', contact_indices_path='mano/contact_indices.json', pose_distrib_path='mano/pose_distrib.pt', device=device)

    # object model
    object_model = ObjectModel(data_root_path=args.data_root_path, batch_size_each=1, num_samples=2000, device=device)
    object_model.initialize(object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    if 'contact_point_indices' in data_dict:
        hand_model.set_parameters(hand_pose.unsqueeze(0), contact_point_indices.unsqueeze(0))
    else:
        hand_model.set_parameters(hand_pose.unsqueeze(0))

    hand_mesh = hand_model.get_trimesh_data(i=0, pose=pose)
    self_intersection = hand_mesh.is_self_intersecting()
    if self_intersection:
        return False
    try:
        hand_mesh = hand_model.get_pymeshlab_data(i=0, pose=pose)
        object_mesh = object_model.get_pymeshlab_data(i=0, pose=pose)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(hand_mesh)
        ms.add_mesh(object_mesh)
        ms.generate_boolean_intersection(first_mesh=0, second_mesh=1)
        # ms.save_current_mesh("pymeshlab_intersection.ply")
        cross_intersection = ms.get_geometric_measures()["mesh_volume"]
        print(cross_intersection)
        if cross_intersection > 1e-6:
            return False
    except:
        print('no inter')
        return False

    return True


if __name__ == '__main__':
    # prepare models

    total_batch_size = len(args.object_code_list) * args.batch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in range(torch.cuda.device_count()))
    print(','.join(str(i) for i in range(torch.cuda.device_count())))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    print('running on', device)

    if args.on_table:
        args.name = 'on_table'
    else:
        args.name = 'random'

    # save results
    result_path = os.path.join('../data/experiments', args.name, 'results')
    for i in range(len(args.object_code_list)):
        data_list = []
        cnt = 0
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            data_path = os.path.join(result_path, args.object_code_list[i], f'data_{cnt:04d}.json')
            save_path = data_path.replace(f'data_{cnt:04d}', f'valid_{cnt:04d}')

            with open(data_path, 'r') as f:
                data_dict = json.load(f)

            if args.on_table:
                plane = torch.tensor(data_dict['plane'], dtype=torch.float,
                                     device=device)  # plane parameters in object reference frame: (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
                pose = plane2pose(plane)  # 4x4 homogeneous transformation matrix from object frame to world frame
                pose = pose.detach().cpu().numpy().tolist()
            else:
                pose = None

            valid = check_intersection(data_dict=data_dict)

            qpos = data_dict['qpos']
            mano_trans = np.array(qpos['trans'], dtype=np.float32).tolist()
            mano_pose = np.concatenate((qpos['rot'], qpos['thetas']), dtype=np.float32).tolist()
            mano_shape = np.zeros((10, ), dtype=np.float32).tolist()

            print(j, valid)

            if valid:
                save_dict = {
                    'mano_trans': mano_trans,
                    'mano_pose': mano_pose,
                    'mano_shape': mano_shape,
                    'pose': pose,
                }
                save_path = data_path.replace(f'data_{cnt:04d}', f'valid_{cnt:04d}')
                with open(save_path, 'w') as f:
                    json.dump(save_dict, f)

            cnt += 1
