import logging
import os.path as osp
import numpy as np
import torch
import cv2
import nori2 as nori
import copy
from pycocotools.coco import COCO
from common.utils.preprocessing import load_img, process_bbox, augmentation, get_bbox
from common.utils.mano import MANO

logger = logging.getLogger(__name__)

mano = MANO()
fetcher = nori.Fetcher()


class DEX_YCB(torch.utils.data.Dataset):

    def __init__(self, cfg, transform, data_split):
        self.cfg = cfg
        self.transform = transform
        self.data_split = data_split
        self.root_joint_idx = 0
        self.root_dir = 'your_DexYCB_path'
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.datalist = self.load_data_local()

    def load_data_local(self):
        db = COCO(osp.join(self.annot_path, 'DEX_YCB_s0_{}_data.json'.format(self.data_split)))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.root_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            if self.data_split == 'train':
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32)  # meter
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                joints_coord_img = np.array(ann['joints_img'], dtype=np.float32)
                hand_type = ann['hand_type']

                bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img['width'], img['height'], aspect_ratio=1, expansion_factor=1.0)

                if bbox is None:
                    continue

                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)

                data = {
                    'img_path': img_path,
                    'img_shape': img_shape,
                    'joints_coord_cam': joints_coord_cam,
                    'joints_coord_img': joints_coord_img,
                    'bbox': bbox,
                    'cam_param': cam_param,
                    'mano_pose': mano_pose,
                    'mano_shape': mano_shape,
                    'hand_type': hand_type
                }
            else:
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32)
                root_joint_cam = copy.deepcopy(joints_coord_cam[0])
                joints_coord_img = np.array(ann['joints_img'], dtype=np.float32)
                hand_type = ann['hand_type']

                bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img['width'], img['height'], aspect_ratio=1, expansion_factor=1.0)
                if bbox is None:
                    bbox = np.array([0, 0, img['width'] - 1, img['height'] - 1], dtype=np.float32)

                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}

                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)

                data = {
                    'img_path': img_path,
                    'img_shape': img_shape,
                    'joints_coord_cam': joints_coord_cam,
                    'joints_coord_img': joints_coord_img,
                    'root_joint_cam': root_joint_cam,
                    'bbox': bbox,
                    'cam_param': cam_param,
                    'image_id': image_id,
                    'mano_pose': mano_pose,
                    'mano_shape': mano_shape,
                    'hand_type': hand_type
                }

            datalist.append(data)
        return datalist

    def no_black_edge(self, bbox, img_height, img_width):
        # offset bbox if it has black edge, bbox: x, y, w, h
        if bbox[0] < 0:
            bbox[0] = 0.
        elif (bbox[0] + bbox[2]) > img_width:
            offset_x = bbox[0] + bbox[2] - (img_width - 1)
            bbox[0] -= offset_x

        if bbox[1] < 0:
            bbox[1] = 0.
        elif (bbox[1] + bbox[3]) > img_height:
            offset_y = bbox[1] + bbox[3] - (img_height - 1)
            bbox[1] -= offset_y
        return bbox

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = copy.deepcopy(self.datalist[index])
        img_path, img_shape, bbox, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['cam_param']
        bbox = self.no_black_edge(bbox=bbox, img_height=img_shape[0], img_width=img_shape[1])
        cam_param = [cam_param['focal'][0], cam_param['focal'][1], cam_param['princpt'][0], cam_param['princpt'][1]]
        hand_type = data['hand_type']
        do_flip = (hand_type == 'left')
        img = load_img(img_path)

        full_img = copy.deepcopy(img)

        if self.data_split == 'train':
            if self.cfg.train.aug:
                img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, self.cfg.data.input_img_shape, do_flip=do_flip)
            else:
                img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, 'test', self.cfg.data.input_img_shape, do_flip=do_flip)
            img = self.transform(img.astype(np.float32)) / 255.

            # 2D joint coordinate
            joints_img = data['joints_coord_img']
            if do_flip:
                joints_img[:, 0] = img_shape[1] - joints_img[:, 0] - 1
            joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            # normalize to [0,1]
            joints_img[:, 0] /= self.cfg.data.input_img_shape[1]
            joints_img[:, 1] /= self.cfg.data.input_img_shape[0]

            # 3D joint camera coordinate
            joints_coord_cam = data['joints_coord_cam']
            root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            joints_coord_cam -= joints_coord_cam[self.root_joint_idx, None, :]  # root-relative
            if do_flip:
                joints_coord_cam[:, 0] *= -1

            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32)
            joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(1, 0)

            # mano parameter
            mano_pose, mano_shape = data['mano_pose'], data['mano_shape']

            # 3D data rotation augmentation
            mano_pose = mano_pose.reshape(-1, 3)
            if do_flip:
                mano_pose[:, 1:] *= -1
            root_pose = mano_pose[self.root_joint_idx, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)

            input = {
                'img': img,
                'full_img': full_img,
                'joints_img': joints_img,
                'joints_coord_cam': joints_coord_cam,
                'mano_pose': mano_pose,
                'val_mano_pose': mano_pose,
                'mano_shape': mano_shape,
                'root_joint_cam': root_joint_cam,
                'bbox': bbox,
                'cam_param': cam_param,
            }

        else:
            img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, 'test', self.cfg.data.input_img_shape, do_flip=do_flip)
            img = self.transform(img.astype(np.float32)) / 255.

            # root_joint_cam = data['root_joint_cam']
            joints_coord_cam = data['joints_coord_cam']

            # mano parameter
            mano_pose, mano_shape = data['mano_pose'], data['mano_shape']

            # Only for rotation metric
            val_mano_pose = copy.deepcopy(mano_pose).reshape(-1, 3)
            if do_flip:
                val_mano_pose[:, 1:] *= -1
            val_mano_pose = val_mano_pose.reshape(-1)

            # 2D joint coordinate
            joints_img = data['joints_coord_img']
            if do_flip:
                joints_img[:, 0] = img_shape[1] - joints_img[:, 0] - 1
            joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            # normalize to [0,1]
            joints_img[:, 0] /= self.cfg.data.input_img_shape[1]
            joints_img[:, 1] /= self.cfg.data.input_img_shape[0]

            input = {
                'img': img,
                'full_img': full_img,
                'joints_img': joints_img,
                'joints_coord_cam': joints_coord_cam,
                'mano_pose': mano_pose,
                'mano_shape': mano_shape,
                'val_mano_pose': val_mano_pose,
                'hand_type': hand_type,
                'do_flip': int(do_flip),
                'bbox': bbox,
                'cam_param': cam_param,
            }
        return input
