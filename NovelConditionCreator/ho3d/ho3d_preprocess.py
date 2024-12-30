import os
import json
import glob
import argparse
import numpy as np
import os.path as osp

import torch
import yaml
import copy
import cv2
import trimesh
import pickle
import matplotlib.pyplot as plt
import nori2 as nori
from tqdm import tqdm
from pycocotools.coco import COCO
from refile import smart_open
from collections import defaultdict

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = '0.0'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.1'

from seg import *
from mano import MANO
from preprocessing import load_img, get_bbox, process_bbox, cam2pixel, np_concatenate, np_inverse

import pyrender
from vis import vis_keypoints_with_skeleton, render_hand_obj_property, render_hand_obj_condition

right_mano = MANO(side='right')
fetcher = nori.Fetcher()


def get_data(nori_id):
    data = fetcher.get(nori_id)
    data = pickle.loads(data)
    return data


def load_data():
    annot_path = osp.join(args.root_dir, 'annotations')
    db = COCO(osp.join(annot_path, 'HO3D_{}_data.json'.format(args.data_split)))

    keys = list(db.anns.keys())
    k, m = divmod(len(keys), args.total_split)
    all_keys = [keys[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(args.total_split)]
    current_keys = all_keys[args.current_split]
    anns = [db.anns[aid] for aid in current_keys]
    imgs = [db.imgs[ann["image_id"]] for ann in anns]
    del db, keys, all_keys, current_keys
    return anns, imgs


def load_ycb_objects():
    obj_file = {k: os.path.join(f'{args.root_dir}/models', v, 'textured_simple.obj') for k, v in YCB_CLASSES.items()}
    all_obj = {}
    for k, v in obj_file.items():
        mesh = trimesh.load(v)
        if args.version == 1:
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
            mesh.visual.face_colors = np.array(YCB_COLORS[k])
        elif args.version == 2:
            pass
        else:
            raise NotImplementedError(f'Not implement version {args.version}')
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        all_obj[k] = mesh
    return all_obj


def convert_mano_pose(mano_pose):
    # convert mano_pose from flat_hand_mean=True to flat_hand_mean=False
    from dexycb.manopth.mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments

    mano_path = os.path.join(os.path.join("./manopth", "mano", "models"), 'MANO_RIGHT.pkl')
    smpl_data = ready_arguments(mano_path)
    # Get hand mean
    hands_mean = smpl_data['hands_mean']
    hands_mean = hands_mean.copy()
    mano_pose[3:] -= hands_mean
    return mano_pose


def generate_random_rot_mat(rot=30):
    anglex = np.random.uniform() * np.pi * rot / 180.0
    angley = np.random.uniform() * np.pi * rot / 180.0
    anglez = np.random.uniform() * np.pi * rot / 180.0

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    rot_aug_mat = Rx @ Ry @ Rz
    return rot_aug_mat


def main(anns, imgs):
    all_obj = load_ycb_objects()

    real_info_path = 'ho3d/sampled_real_info.json'
    gen_info_path = 'ho3d/similarity_sampled_gen_info.json'

    with open(real_info_path, 'r') as f:
        real_info = json.load(f)
    with open(gen_info_path, 'r') as f:
        gen_info = json.load(f)

    num_data = len(anns)
    for idx, (ann, img) in tqdm(enumerate(zip(anns, imgs)), total=num_data, desc=f'current split: {args.current_split}'):
        img_id = np.array(ann['image_id'])
        img_fn = np.array(img['file_name'])
        img_shape = (img["height"], img["width"])
        img_path = os.path.join(args.root_dir, args.data_split, img['file_name'])

        cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
        cam_param = np.array([cam_param['focal'][0], cam_param['focal'][1], cam_param['princpt'][0], cam_param['princpt'][1]])

        mano = right_mano
        extra_faces = right_extra_faces

        # random choose an object
        object_code_id = np.random.choice(list(range(1, 22)))
        object_code = YCB_CLASSES[object_code_id]
        random_number = np.random.random()
        # Gen fails, real does not have, not use
        if object_code_id == 18:
            continue
        # generated grasping
        if (random_number < 0.8) or (object_code_id not in [2, 3, 5, 9, 10, 12, 14, 15, 17]):
            # random choose a mano_pose, object_pose
            valid_pose_files = gen_info['img_fn'][str(object_code_id)]
            valid_pose_path = valid_pose_files[np.random.randint(0, len(valid_pose_files))]
            with open(valid_pose_path, 'r') as f:
                tmp = json.load(f)

            # mano_trans and mano_pose
            mano_trans = np.array(tmp['mano_trans'], dtype=np.float32)
            mano_pose = np.array(tmp['mano_pose'], dtype=np.float32)
            mano_pose = convert_mano_pose(mano_pose)
            mesh_y = [all_obj[object_code_id]]
            pose_y = np.eye(4)[:3, :]

            # NOTE v2: random choose a root_joint_coord_cam, mano_shape, global_rotation
            root_joint_coord_cam_files = glob.glob('ho3d/root_joint_coord_cam/*.json')
            root_joint_coord_cam_path = root_joint_coord_cam_files[np.random.randint(0, len(root_joint_coord_cam_files))]
            mano_shape_path = root_joint_coord_cam_path.replace('root_joint_coord_cam', 'mano_shape')
            global_rotation_path = root_joint_coord_cam_path.replace('root_joint_coord_cam', 'global_rotation')

            with open(root_joint_coord_cam_path, 'r') as f:
                tmp = json.load(f)
            root_joint_coord_cam = np.array(tmp, dtype=np.float32)
            with open(mano_shape_path, 'r') as f:
                tmp = json.load(f)
            mano_shape = np.array(tmp, dtype=np.float32)
            with open(global_rotation_path, 'r') as f:
                tmp = json.load(f)
            global_rotation = np.array(tmp, dtype=np.float32)

            # set global rotation for hand and object
            current_root_pose, _ = cv2.Rodrigues(mano_pose[0:3])
            target_root_pose, _ = cv2.Rodrigues(global_rotation)
            res_root_pose = np.dot(target_root_pose, current_root_pose.T)
            mano_pose[0:3] = global_rotation

            # generate random rotation
            rot_aug_mat = generate_random_rot_mat()

            # aug hand global rotation
            mano_pose = mano_pose.reshape(-1, 3)
            root_pose = mano_pose[0, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            mano_pose[0] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)

            # gt joints and verts
            mano_pose = torch.from_numpy(mano_pose[None, ...])
            mano_shape = torch.from_numpy(mano_shape[None, ...])

            verts_cam, joints_cam = mano.layer(th_pose_coeffs=mano_pose, th_betas=mano_shape)
            verts_cam = verts_cam.squeeze().cpu().numpy()
            joints_cam = joints_cam.squeeze().cpu().numpy()
            verts_cam = verts_cam / 1000.
            joints_cam = joints_cam / 1000.

            # set object pose
            pose_y[:3, 3] -= mano_trans
            pose_y[:3, 3] -= joints_cam[0]
            pose_y[:3, :] = np.dot(res_root_pose, pose_y[:3, :])
            pose_y[:3, 3] += root_joint_coord_cam

            # aug object global rotation
            pose_y[:3, 3] -= root_joint_coord_cam
            pose_y = np.dot(rot_aug_mat, pose_y)
            pose_y[:3, 3] += root_joint_coord_cam
            pose_y = [pose_y]

            verts_cam = verts_cam - joints_cam[0] + root_joint_coord_cam
            joints_coord_cam = joints_cam - joints_cam[0] + root_joint_coord_cam
            joints_coord_img = cam2pixel(joints_coord_cam, f=[cam_param[0], cam_param[1]], c=[cam_param[2], cam_param[3]])

        # real grasping
        elif (random_number >= 0.8) and (object_code_id in [2, 3, 5, 9, 10, 12, 14, 15, 17]):
            # random choose a mano_pose, object_pose
            valid_nori_ids = real_info['nori_id'][str(object_code_id)]
            valid_nori_id = valid_nori_ids[np.random.randint(0, len(valid_nori_ids))]

            data = get_data(valid_nori_id)
            img_fn = data['img_fn']
            img_path = os.path.join('/data/data/HO3D/train', img_fn)
            cam_param = data['cam_param']

            mano = right_mano
            extra_faces = right_extra_faces

            # mano pose and shape
            mano_pose = data['mano_pose']
            mano_pose = np.squeeze(mano_pose)
            mano_shape = data['mano_shape']
            mano_shape = np.squeeze(mano_shape)
            joints_coord_cam = data['joints_coord_cam']

            # load meta
            meta_file = img_path.replace('rgb', 'meta').replace('png', 'pkl')
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f, encoding='latin1')
            name_y = meta['objName']
            pose_y = np.concatenate((cv2.Rodrigues(meta['objRot'])[0], meta['objTrans'][:, None]), axis=1)
            pose_y[1] *= -1
            pose_y[2] *= -1
            pose_y = [pose_y]
            current_pose_y = pose_y[0]

            # load YCB meshes
            ycb_ids = YCB_INDICES[name_y]
            mesh_y = [all_obj[ycb_ids]]

            # generate rotation
            rot_aug_mat = generate_random_rot_mat()

            # aug object
            current_pose_y[:3, 3] -= joints_coord_cam[0]
            current_pose_y = np.dot(rot_aug_mat, current_pose_y)
            current_pose_y[:3, 3] += joints_coord_cam[0]
            pose_y[0] = current_pose_y

            # aug hand
            mano_pose = mano_pose.reshape(-1, 3)
            root_pose = mano_pose[0, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            mano_pose[0] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)

            # gt joints and verts
            mano_pose = torch.from_numpy(mano_pose[None, ...])
            mano_shape = torch.from_numpy(mano_shape[None, ...])

            verts_cam, joints_cam = mano.layer(th_pose_coeffs=mano_pose, th_betas=mano_shape)
            verts_cam = verts_cam.squeeze().cpu().numpy()
            joints_cam = joints_cam.squeeze().cpu().numpy()
            verts_cam = verts_cam / 1000.
            joints_cam = joints_cam / 1000.

            verts_cam = verts_cam - joints_cam[0] + joints_coord_cam[0]
            joints_coord_cam = joints_cam - joints_cam[0] + joints_coord_cam[0]
            joints_coord_img = cam2pixel(joints_coord_cam, f=[cam_param[0], cam_param[1]], c=[cam_param[2], cam_param[3]])

        else:
            raise RuntimeError('We have a bug here...')

        # bbox
        bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
        bbox = process_bbox(bbox, img['width'], img['height'], aspect_ratio=1, expansion_factor=1.0)
        if bbox is None:
            continue

        # close the wrist
        face = np.concatenate((mano.face, extra_faces), axis=0)
        normal, depth = render_hand_obj_property(img_shape, verts_cam, face, mesh_y, pose_y, cam_param)
        condition = render_hand_obj_condition(img_shape, verts_cam, face, mesh_y, pose_y, 'right', cam_param)

        # joint image
        skeleton = np.array(mano.skeleton)
        joints2d = np.concatenate((joints_coord_img, joints_coord_cam[:, -1][:, None]), axis=1)
        joint_img = vis_keypoints_with_skeleton(condition.copy(), joints2d.T, skeleton)

        # save intermediate results
        # if True:  # for debug
        if idx % 200 == 0:
            save_dir = f'{args.data_split}_{args.suffix}_v{args.version}_p{args.part}'
            os.makedirs(f'./debug/{save_dir}', exist_ok=True)
            # image
            color = load_img(img_path)
            cat_img = np.concatenate([color, normal, condition, joint_img], axis=1)
            cv2.imwrite(f'./debug/{save_dir}/{args.current_split}-{args.total_split}_{idx:08d}.jpg', cat_img[:, :, ::-1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DexYCB preprocess', add_help=True)
    parser.add_argument('--root_dir', '-r', type=str, default='/data/data/HO3D', help='root data dir')
    parser.add_argument('--data_split', '-p', type=str, default='train', help='data split')
    parser.add_argument('--total_split', '-ts', type=int, default=1, help='total split of data list')
    parser.add_argument('--current_split', '-pi', type=int, default=0, help='current split of data list')
    parser.add_argument('--version', '-v', type=int, default=2, help='generation version')
    parser.add_argument('--part', '-pt', type=int, default=1, help='generation part')
    parser.add_argument('--condition', '-d', type=int, default=1, help='process condition')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    # condition and set suffix
    args.condition = bool(args.condition)
    if args.condition:
        args.suffix = 'condition'
    else:
        args.suffix = 'ori'

    anns, imgs = load_data()

    main(anns, imgs)
