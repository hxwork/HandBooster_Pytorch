"""
Last modified date: 2023.04.12
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('/data/code/DexGraspNet')
print(sys.path)

import argparse
import torch
import numpy as np
import plotly.graph_objects as go
import open3d as o3d
import trimesh as tm
import pymeshlab

from grasp_generation.utils.hand_model import HandModel
from grasp_generation.utils.object_model import ObjectModel


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str, default='../data/meshdata_dexycb')
    # parser.add_argument('--object_code', type=str, default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    # parser.add_argument('--object_code', type=str, default='002_master_chef_can')
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
    parser.add_argument('--num', type=int, default=9)
    parser.add_argument('--result_path', type=str, default='../data/experiments/demo/results')
    args = parser.parse_args()

    device = 'cpu'

    # load results
    for object_code in args.object_code_list:
        batch_data_dict = np.load(os.path.join(args.result_path, object_code + '.npy'), allow_pickle=True)
        for idx in range(len(batch_data_dict)):
            data_dict = batch_data_dict[idx]
            plane = torch.tensor(data_dict['plane'], dtype=torch.float,
                                 device=device)  # plane parameters in object reference frame: (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
            pose = plane2pose(plane)  # 4x4 homogeneous transformation matrix from object frame to world frame
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

            # # NOTE v1 intersection determination
            # hand_mesh = hand_model.get_trimesh_data(i=0, pose=pose)
            # self_intersection = hand_mesh.is_self_intersecting()
            # hand_mesh = o3d.t.geometry.TriangleMesh.from_legacy(hand_mesh)
            # object_mesh = object_model.get_trimesh_data(i=0, pose=pose)
            # object_mesh = o3d.t.geometry.TriangleMesh.from_legacy(object_mesh)
            # cross_intersection = object_mesh.boolean_intersection(hand_mesh).to_legacy()
            # vertices = np.asarray(cross_intersection.vertices)
            # faces = np.asarray(cross_intersection.triangles)
            # data = [go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color='lightgreen', opacity=1)]
            # fig = go.Figure(data)
            # fig.update_layout(scene_aspectmode='data')
            # save_dir = f'./results/{object_code}'
            # os.makedirs(save_dir, exist_ok=True)
            # fig.write_html(os.path.join(save_dir, f'{idx:02d}_intersection_{self_intersection}_verts_{vertices.shape}_faces_{faces.shape}.html'))
            # print(f'obj_name: {object_code}, idx: {idx}')
            # print(f'vertices: {vertices.shape}')
            # print(f'faces: {faces.shape}')  # NOTE face number should be less than 300

            # # NOTE v2 intersection determination
            # hand_mesh = hand_model.get_trimesh_data(i=0, pose=pose)
            # self_intersection = hand_mesh.is_self_intersecting()

            # hand_mesh = hand_model.get_trimesh_data(i=0, pose=pose, open3d=False)
            # object_mesh = object_model.get_trimesh_data(i=0, pose=pose, open3d=False)
            # cross_intersection = tm.boolean.intersection([hand_mesh, object_mesh])
            # print(cross_intersection)

            # NOTE v3 intersection determination
            hand_mesh = hand_model.get_trimesh_data(i=0, pose=pose)
            self_intersection = hand_mesh.is_self_intersecting()
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
            except:
                print('no inter')

            fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
            if 'energy' in data_dict:
                scale = round(data_dict['scale'], 2)
                energy = data_dict['energy']
                E_fc = round(data_dict['E_fc'], 3)
                E_dis = round(data_dict['E_dis'], 5)
                E_pen = round(data_dict['E_pen'], 5)
                E_prior = round(data_dict['E_prior'], 3)
                E_spen = round(data_dict['E_spen'], 4)
                E_tpen = round(data_dict['E_tpen'], 4)
                result = f'Index {args.num}  scale {scale}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}  E_prior {E_prior}  E_spen {E_spen}  E_tpen {E_tpen}'
                fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
            fig.update_layout(scene_aspectmode='data')
            # fig.show()
            save_dir = f'./results/{object_code}'
            os.makedirs(save_dir, exist_ok=True)
            fig.write_html(os.path.join(save_dir, f'{idx:02d}.html'))
