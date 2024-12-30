import os
import numpy as np
import nori2 as nori
from torch.utils.data import Dataset
from torchvision import transforms as T
from functools import partial
from torch import nn
from data.preprocess import augmentation_list, str_to_nparray

fetcher = nori.Fetcher()


def exists(x):
    return x is not None


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class HO3D(Dataset):

    def __init__(self, data_split, version, part, image_size, convert_image_to=None):
        super().__init__()
        self.data_split = data_split
        self.version = version
        self.part = part
        self.image_size = image_size
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([T.Lambda(maybe_convert_fn), T.ToTensor()])
        self.root_dir = 'your_HO3D_dir'
        self.annot_path = os.path.join(self.root_dir, 'annotations')
        self.root_joint_idx = 0

        self.datalist = self.load_data()

    def load_data(self):
        datalist = []
        # TODO use local data
        return datalist

    def img_fn_seq_name_to_idx(self, img_fn):
        all_seq_names = os.listdir(os.path.join(self.root_dir, 'train'))
        seq_name = img_fn.split('/')[0]
        seq_idx = f'{all_seq_names.index(seq_name):02d}'
        img_fn = img_fn.replace(seq_name, seq_idx)
        return img_fn

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
        nori_id = self.datalist[index]
        data = self.get_data(nori_id)
        bbox, hand_type = data['bbox'], 1
        do_flip = False  # 1 for right_hand

        # img
        img = data['color']
        img_fn = data['img_fn'].split('/')[-2]
        img_path = os.path.join(self.root_dir, data['img_fn'])
        img_height, img_width = 480, 640
        condition = data['condition']
        normal = data['normal']

        img_list = [img, condition, normal]
        bbox = self.no_black_edge(bbox, img_height, img_width)
        aug_img_list, img2bb_trans, bb2img_trans, rot, scale = augmentation_list(
            img_list,
            bbox,
            'test',
            (self.image_size, self.image_size),
            do_flip=do_flip,
        )

        img, condition, normal = aug_img_list
        img = self.transform(img)
        condition = self.transform(condition)
        normal = self.transform(normal)

        # 2D joint coordinate
        joints_img = data['joints_coord_img']
        if do_flip:
            joints_img[:, 0] = img_width - joints_img[:, 0] - 1
        joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
        joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
        # normalize to [0,1]
        joints_img[:, 0] /= self.image_size
        joints_img[:, 1] /= self.image_size

        input = {
            # for DM
            'img': img.float() / 255.,
            'condition': condition.float() / 255.,
            'normal': normal.float() / 255.,
            # for hand recon
            'img_id': np.copy(data['img_id']),
            'mano_pose': np.copy(data['mano_pose']),
            'mano_shape': np.copy(data['mano_shape']),
            'hand_type': 1,
            # for debug
            'img_fn': np.copy(str_to_nparray(self.img_fn_seq_name_to_idx(data['img_fn']))),
            'color': np.copy(data['color']),
            'bbox': np.copy(bbox),
            'joints_coord_cam': np.copy(data['joints_coord_cam']),
            'joints_img': np.copy(joints_img),
            'cam_param': np.copy(data['cam_param']),
        }

        return input
