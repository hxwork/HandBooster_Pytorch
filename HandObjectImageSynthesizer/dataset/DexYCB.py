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


class DexYCB(Dataset):

    def __init__(self, data_split, version, part, image_size, convert_image_to=None):
        super().__init__()
        self.data_split = data_split
        self.version = version
        self.part = part
        self.image_size = image_size
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([T.Lambda(maybe_convert_fn), T.ToTensor()])
        self.root_dir = 'your_DexYCB_dir'
        self.annot_path = os.path.join(self.root_dir, 'annotations')
        self.root_joint_idx = 0

        self.datalist = self.load_data()

    def load_data(self):
        datalist = []
        # TODO use local data
        return datalist

    def img_fn_to_view_idx(self, img_fn):
        _SERIALS = [
            '836212060125',
            '839512060362',
            '840412060917',
            '841412060263',
            '932122060857',
            '932122060861',
            '932122061900',
            '932122062010',
        ]
        view_idx = _SERIALS.index(img_fn)
        return view_idx

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
        bbox = data['bbox']
        data['hand_type'] = 1

        # img
        img = data['color']
        img_fn = data['img_fn'].split('/')[-2]
        img_path = os.path.join(self.root_dir, data['img_fn'])
        img_height, img_width = 480, 640
        view_idx = self.img_fn_to_view_idx(img_fn)
        condition = data['condition']
        normal = data['normal']

        img_list = [img, condition, normal]
        bbox = self.no_black_edge(bbox, img_height, img_width)
        aug_img_list, img2bb_trans, bb2img_trans, rot, scale = augmentation_list(
            img_list,
            bbox,
            'test',
            (self.image_size, self.image_size),
            do_flip=False,
        )

        img, condition, normal = aug_img_list
        img = self.transform(img)
        condition = self.transform(condition)
        normal = self.transform(normal)

        # 2D joint coordinate
        joints_img = data['joints_coord_img']
        joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
        joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
        # normalize to [0,1]
        joints_img[:, 0] /= self.image_size
        joints_img[:, 1] /= self.image_size

        mano_pose = np.copy(data['mano_pose'])
        mano_shape = np.copy(data['mano_shape'])

        input = {
            # for DM
            'img': img.float() / 255.,
            'img_path': np.copy(str_to_nparray(img_path)),
            'view_idx': view_idx,
            'condition': condition.float() / 255.,
            'normal': normal.float() / 255.,
            # for hand recon
            'img_id': np.copy(data['img_id']),
            'mano_pose': mano_pose,
            'mano_shape': mano_shape,
            'hand_type': np.copy(data['hand_type']),
            # for debug
            'img_fn': np.copy(str_to_nparray(data['img_fn'])),
            'color': np.copy(data['color']),
            'bbox': np.copy(bbox),
            'joints_coord_cam': np.copy(data['joints_coord_cam']),
            'joints_img': np.copy(joints_img),
            'cam_param': np.copy(data['cam_param']),
        }

        return input
