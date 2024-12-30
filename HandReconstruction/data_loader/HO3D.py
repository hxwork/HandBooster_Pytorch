import logging
import os
import os.path as osp
import numpy as np
import cv2
import nori2 as nori
import json
import copy
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from common.utils.preprocessing import load_img, process_bbox, augmentation, get_bbox
from common.utils.transforms import cam2pixel, transform_joint_to_other_db
from common.utils.mano import MANO

logger = logging.getLogger(__name__)

mano = MANO()
fetcher = nori.Fetcher()


class HO3D(Dataset):

    def __init__(self, cfg, transform, data_split):
        super(HO3D, self).__init__()
        self.cfg = cfg
        self.transform = transform
        self.data_split = data_split
        self.root_dir = 'your_HO3D_path'
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.root_joint_idx = 0
        self.mano = MANO()
        if self.data_split == 'train' or self.data_split == 'val':
            self.datalist = self.load_train_val_data_local()
        elif self.data_split == 'test':
            self.datalist = self.load_test_data()
            self.eval_result = [[], []]  # [pred_joints_list, pred_verts_list]
        else:
            raise NotImplementedError

        self.joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1',
                            'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_4', 'Middle_4', 'Ring_4', 'Pinly_4')

    def load_test_data(self):
        db = COCO(osp.join(self.annot_path, 'HO3D_evaluation_data.json'))

        datalist = []
        cnt = 0
        for aid in db.anns.keys():
            cnt += 1
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.root_dir, 'evaluation', img['file_name'])
            img_shape = (img['height'], img['width'])

            root_joint_cam = np.array(ann['root_joint_cam'], dtype=np.float32)
            cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
            bbox = np.array(ann['bbox'], dtype=np.float32)
            bbox = process_bbox(bbox, img['width'], img['height'], aspect_ratio=1, expansion_factor=1.5)

            data = {
                'img_path': img_path,
                'img_shape': img_shape,
                'root_joint_cam': root_joint_cam,
                'bbox': bbox,
                'cam_param': cam_param,
            }

            datalist.append(data)
        return datalist

    def load_train_val_data_local(self):
        db = COCO(osp.join(self.annot_path, 'HO3D_train_data.json'))
        datalist = []
        cnt = 0
        for aid in db.anns.keys():
            cnt += 1
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.root_dir, self.data_split_for_load, img['file_name'])
            img_shape = (img['height'], img['width'])
            if self.data_split == 'train' or self.data_split == 'val':
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32)  # meter
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                joints_coord_img = cam2pixel(joints_coord_cam, cam_param['focal'], cam_param['princpt'])
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
                    'mano_shape': mano_shape
                }
            else:
                root_joint_cam = np.array(ann['root_joint_cam'], dtype=np.float32)
                cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
                bbox = np.array(ann['bbox'], dtype=np.float32)
                bbox = process_bbox(bbox, img['width'], img['height'], aspect_ratio=1, expansion_factor=1.5)

                data = {
                    'img_path': img_path,
                    'img_shape': img_shape,
                    'root_joint_cam': root_joint_cam,
                    'bbox': bbox,
                    'cam_param': cam_param,
                }

            datalist.append(data)

        sampled_datlist = []
        cnt = 0
        for data in datalist:
            cnt += 1
            if self.data_split == 'train' and cnt % 20 == 0:
                continue
            if self.data_split == 'val' and cnt % 20 != 0:
                continue
            sampled_datlist.append(data)

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
        if self.data_split == 'train' or self.data_split == 'val':
            data = copy.deepcopy(self.datalist[index])
            img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
            bbox = self.no_black_edge(bbox=bbox, img_height=img_shape[0], img_width=img_shape[1])
            img = load_img(img_path)

            if self.cfg.train.aug:
                img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, self.cfg.data.input_img_shape, do_flip=False)
            else:
                img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, 'test', self.cfg.data.input_img_shape, do_flip=False)
            img = self.transform(img.astype(np.float32)) / 255.

            # 2D joint coordinate
            joints_img = data['joints_coord_img']
            joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            # normalize to [0,1]
            joints_img[:, 0] /= self.cfg.data.input_img_shape[1]
            joints_img[:, 1] /= self.cfg.data.input_img_shape[0]

            # 3D joint camera coordinate
            joints_coord_cam = data['joints_coord_cam']
            root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            joints_coord_cam -= joints_coord_cam[self.root_joint_idx, None, :]  # root-relative

            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32)
            joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(1, 0)

            # mano parameter
            mano_pose, mano_shape = data['mano_pose'], data['mano_shape']
            # 3D data rotation augmentation
            mano_pose = mano_pose.reshape(-1, 3)
            root_pose = mano_pose[self.root_joint_idx, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)

            input = {
                'img': img,
                'joints_img': joints_img,
                'joints_coord_cam': joints_coord_cam,
                'val_mano_pose': mano_pose,
                'mano_pose': mano_pose,
                'mano_shape': mano_shape,
            }

        else:
            data = copy.deepcopy(self.datalist[index])
            img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
            cam_param = np.array([data['cam_param']['focal'][0], data['cam_param']['focal'][1], data['cam_param']['princpt'][0], data['cam_param']['princpt'][1]])

            # img
            img = load_img(img_path)
            full_img = img.copy()
            img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, self.cfg.data.input_img_shape, do_flip=False)
            img = self.transform(img.astype(np.float32)) / 255.

            root_joint_cam = data['root_joint_cam']
            input = {
                'img': img,
                'full_img': full_img,
                'img_fn': img_path,
                'bbox': bbox,
                'cam_param': cam_param,
                'root_joint_cam': root_joint_cam,
            }

        return input

    def evaluate(self, batch_output, cur_sample_idx):
        annots = self.datalist
        batch_size = len(batch_output)
        for n in range(batch_size):
            data = annots[cur_sample_idx + n]
            output = batch_output[n]
            verts_out = output['pred_verts3d_cam']
            joints_out = output['pred_joints3d_cam']

            # root align
            gt_root_joint_cam = data['root_joint_cam']
            verts_out = verts_out - joints_out[self.root_joint_idx] + gt_root_joint_cam
            joints_out = joints_out - joints_out[self.root_joint_idx] + gt_root_joint_cam

            # convert to openGL coordinate system.
            verts_out *= np.array([1, -1, -1])
            joints_out *= np.array([1, -1, -1])

            # convert joint ordering from MANO to HO3D.
            joints_out = transform_joint_to_other_db(joints_out, self.mano.joints_name, self.joints_name)

            self.eval_result[0].append(joints_out.tolist())
            self.eval_result[1].append(verts_out.tolist())

    def print_eval_result(self, epoch):
        output_json_file = osp.join(self.cfg.base.model_dir, f'pred_{"_".join(i for i in self.cfg.base.model_dir.split("/"))}_e{epoch}.json')
        output_zip_file = osp.join(self.cfg.base.model_dir, f'pred_{"_".join(i for i in self.cfg.base.model_dir.split("/"))}_e{epoch}.zip')

        with open(output_json_file, 'w') as f:
            json.dump(self.eval_result, f)
        print('Dumped %d joints and %d verts predictions to %s' % (len(self.eval_result[0]), len(self.eval_result[1]), output_json_file))

        cmd = 'zip -j {} {}'.format(output_zip_file, output_json_file)
        print(cmd)
        os.system(cmd)
