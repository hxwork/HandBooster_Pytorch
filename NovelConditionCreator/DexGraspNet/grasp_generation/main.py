"""
Last modified date: 2023.04.12
Author: Jialiang Zhang, Ruicheng Wang
Description: Entry of the program, generate small-scale experiments
"""

import os

os.chdir(os.path.dirname(__file__))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
import open3d as o3d
import json
import pymeshlab
import plotly.graph_objects as go

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_table_top, initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.logger import Logger

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
        # # '040_large_marker',
        # '051_large_clamp',
        # '052_extra_large_clamp',
        # '061_foam_brick',
    ],
    # type=list,
    nargs='+')
parser.add_argument('--on_table', action='store_true')
parser.add_argument('--name', default='demo', type=str)
parser.add_argument('--n_contact', default=4, type=int)
parser.add_argument('--batch_size', default=10000, type=int)
parser.add_argument('--n_iter', default=6000, type=int)
parser.add_argument('--poses', default='../data/poses', type=str)
# hyper parameters (** Magic, don't touch! **)
parser.add_argument('--switch_possibility', default=0.5, type=float)
parser.add_argument('--mu', default=0.98, type=float)
parser.add_argument('--step_size', default=0.005, type=float)
parser.add_argument('--stepsize_period', default=50, type=int)
parser.add_argument('--starting_temperature', default=18, type=float)
parser.add_argument('--annealing_period', default=30, type=int)
parser.add_argument('--temperature_decay', default=0.95, type=float)
parser.add_argument('--w_dis', default=100.0, type=float)
parser.add_argument('--w_pen', default=100.0, type=float)
parser.add_argument('--w_prior', default=0.5, type=float)
parser.add_argument('--w_spen', default=10.0, type=float)
parser.add_argument('--w_tpen', default=0.0, type=float)
# initialization settings
parser.add_argument('--jitter_strength', default=0., type=float)
parser.add_argument('--distance_lower', default=0.1, type=float)
parser.add_argument('--distance_upper', default=0.1, type=float)
parser.add_argument('--theta_lower', default=0, type=float)
parser.add_argument('--theta_upper', default=0, type=float)
parser.add_argument('--angle_upper', default=math.pi / 4, type=float)
# energy thresholds
parser.add_argument('--thres_fc', default=0.3, type=float)
parser.add_argument('--thres_dis', default=0.005, type=float)
parser.add_argument('--thres_pen', default=0.001, type=float)

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
        pose = pose.detach().cpu().numpy().tolist()
    else:
        pose = None
    qpos = data_dict['qpos']
    hand_pose = torch.concat([torch.tensor(qpos[key], dtype=torch.float, device=device) for key in ['trans', 'rot', 'thetas']])
    if 'contact_point_indices' in data_dict:
        contact_point_indices = torch.tensor(data_dict['contact_point_indices'], dtype=torch.long, device=device)
    if 'qpos_st' in data_dict:
        qpos_st = data_dict['qpos_st']
        hand_pose_st = torch.concat([torch.tensor(qpos_st[key], dtype=torch.float, device=device) for key in ['trans', 'rot', 'thetas']])

    # hand model
    hand_model = HandModel(mano_root='mano', contact_indices_path='mano/contact_indices.json', pose_distrib_path='mano/pose_distrib.pt', device=device)

    # object model
    object_model = ObjectModel(data_root_path=args.data_root_path, batch_size_each=1, num_samples=2000, device=device)
    object_model.initialize(object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    # visualize
    if 'qpos_st' in data_dict:
        hand_model.set_parameters(hand_pose_st.unsqueeze(0))
        hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue', pose=pose)
    else:
        hand_st_plotly = []
    if 'contact_point_indices' in data_dict:
        hand_model.set_parameters(hand_pose.unsqueeze(0), contact_point_indices.unsqueeze(0))
        hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=True, pose=pose)
    else:
        hand_model.set_parameters(hand_pose.unsqueeze(0))
        hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', pose=pose)
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1, pose=pose)
    plotly_data = hand_st_plotly + hand_en_plotly + object_plotly

    valid = True
    # NOTE intersection determination
    hand_mesh = hand_model.get_trimesh_data(i=0, pose=pose)
    self_intersection = hand_mesh.is_self_intersecting()
    if self_intersection:
        valid = False
    try:
        hand_mesh = hand_model.get_pymeshlab_data(i=0, pose=pose)
        object_mesh = object_model.get_pymeshlab_data(i=0, pose=pose)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(hand_mesh)
        ms.add_mesh(object_mesh)
        ms.generate_boolean_intersection(first_mesh=0, second_mesh=1)
        cross_intersection = ms.get_geometric_measures()["mesh_volume"]
        print(cross_intersection)
        if cross_intersection > 5e-7:
            valid = False
    except:
        print('no inter')

    return valid, plotly_data


if __name__ == '__main__':
    # prepare models

    total_batch_size = len(args.object_code_list) * args.batch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in range(torch.cuda.device_count()))
    print(','.join(str(i) for i in range(torch.cuda.device_count())))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on', device)
    print(args.object_code_list)

    hand_model = HandModel(mano_root='mano', contact_indices_path='mano/contact_indices.json', pose_distrib_path='mano/pose_distrib.pt', device=device)

    object_model = ObjectModel(data_root_path=args.data_root_path, batch_size_each=args.batch_size, num_samples=2000, device=device, random_scale=False)
    object_model.initialize(args.object_code_list)

    if args.on_table:
        args.name = 'on_table_1011'
        initialize_table_top(hand_model, object_model, args)
    else:
        args.name = 'random'
        initialize_convex_hull(hand_model, object_model, args)

    print('total batch size', total_batch_size)
    hand_pose_st = hand_model.hand_pose.detach()

    optim_config = {
        'switch_possibility': args.switch_possibility,
        'starting_temperature': args.starting_temperature,
        'temperature_decay': args.temperature_decay,
        'annealing_period': args.annealing_period,
        'step_size': args.step_size,
        'stepsize_period': args.stepsize_period,
        'mu': args.mu,
        'device': device
    }
    optimizer = Annealing(hand_model, **optim_config)

    os.makedirs(os.path.join('../data/experiments', args.name, 'logs'), exist_ok=True)
    logger_config = {'thres_fc': args.thres_fc, 'thres_dis': args.thres_dis, 'thres_pen': args.thres_pen}
    logger = Logger(log_dir=os.path.join('../data/experiments', args.name, 'logs'), **logger_config)

    # log settings

    with open(os.path.join('../data/experiments', args.name, 'output.txt'), 'w') as f:
        f.write(str(args) + '\n')

    # optimize

    weight_dict = dict(
        w_dis=args.w_dis,
        w_pen=args.w_pen,
        w_prior=args.w_prior,
        w_spen=args.w_spen,
        w_tpen=args.w_tpen,
    )
    energy, E_fc, E_dis, E_pen, E_prior, E_spen, E_tpen = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    energy.sum().backward(retain_graph=True)
    logger.log(energy, E_fc, E_dis, E_pen, E_prior, E_spen, E_tpen, 0, show=False)

    for step in tqdm(range(1, args.n_iter + 1), desc='optimizing'):
        s = optimizer.try_step()

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_prior, new_E_spen, new_E_tpen = cal_energy(hand_model, object_model, on_table=args.on_table, verbose=True, **weight_dict)

        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_prior[accept] = new_E_prior[accept]
            E_spen[accept] = new_E_spen[accept]
            E_tpen[accept] = new_E_tpen[accept]

            logger.log(energy, E_fc, E_dis, E_pen, E_prior, E_spen, E_tpen, step, show=False)

    # save results
    os.makedirs(os.path.join('../data/experiments', args.name, 'results'), exist_ok=True)
    result_path = os.path.join('../data/experiments', args.name, 'results')
    os.makedirs(result_path, exist_ok=True)
    for i in range(len(args.object_code_list)):
        data_list = []
        cnt = 0
        for j in tqdm(range(args.batch_size), desc=f'{args.object_code_list[i]}'):
            idx = i * args.batch_size + j
            scale = object_model.object_scale_tensor[i][j].item()
            hand_pose = hand_model.hand_pose[idx].detach().cpu()
            qpos = dict(
                trans=hand_pose[:3].tolist(),
                rot=hand_pose[3:6].tolist(),
                thetas=hand_pose[6:].tolist(),
            )
            hand_pose = hand_pose_st[idx].detach().cpu()
            qpos_st = dict(
                trans=hand_pose[:3].tolist(),
                rot=hand_pose[3:6].tolist(),
                thetas=hand_pose[6:].tolist(),
            )
            if args.on_table:
                data_dict = dict(
                    object_code=args.object_code_list[i],
                    scale=scale,
                    plane=object_model.plane_parameters[idx].tolist(),
                    qpos=qpos,
                    contact_point_indices=hand_model.contact_point_indices[idx].detach().cpu().tolist(),
                    qpos_st=qpos_st,
                )
                plane = torch.tensor(data_dict['plane'], dtype=torch.float,
                                     device=device)  # plane parameters in object reference frame: (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
                pose = plane2pose(plane)  # 4x4 homogeneous transformation matrix from object frame to world frame
                pose = pose.detach().cpu().numpy().tolist()
            else:
                data_dict = dict(
                    object_code=args.object_code_list[i],
                    scale=scale,
                    qpos=qpos,
                    contact_point_indices=hand_model.contact_point_indices[idx].detach().cpu().tolist(),
                    qpos_st=qpos_st,
                )
                pose = None

            valid, plotly_data = check_intersection(data_dict=data_dict)

            # print(j, valid)

            if valid:

                qpos = data_dict['qpos']
                mano_trans = np.array(qpos['trans'], dtype=np.float32).tolist()
                mano_pose = np.concatenate((qpos['rot'], qpos['thetas']), dtype=np.float32).tolist()
                mano_shape = np.zeros((10, ), dtype=np.float32).tolist()

                save_dict = {
                    'mano_trans': mano_trans,
                    'mano_pose': mano_pose,
                    'mano_shape': mano_shape,
                    'pose': pose,
                }
                save_path = os.path.join(result_path, args.object_code_list[i], f'{cnt:04d}.json')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(save_dict, f)

                if cnt % 5 == 0:
                    fig = go.Figure(plotly_data)
                    fig.update_layout(scene_aspectmode='data')
                    fig.write_html(os.path.join(os.path.dirname(save_path), f'{cnt:04d}.html'))
                    fig.write_image(os.path.join(os.path.dirname(save_path), f'{cnt:04d}.png'))

            cnt += 1
