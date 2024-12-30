import numpy as np
import cv2
import random
import os
from sklearn.neighbors import NearestNeighbors

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def load_img(path, order='RGB', dtype='uint8', depth=False):
    # Load RGB image
    if not depth:
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if not isinstance(img, np.ndarray):
            raise IOError('Fail to read %s' % path)

        if order == 'RGB':
            img = img[:, :, ::-1].copy()
        img = img.astype(dtype)

    # Load EXR depth
    else:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        img = img.astype(np.float32)

    return img


def load_img_npy(path):
    img = np.load(path)
    img = img.astype(np.float32)
    return img


def get_bbox(joint_img, joint_valid, expansion_factor=1.0):

    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.
    width = (xmax - xmin) * expansion_factor
    xmin = x_center - 0.5 * width
    xmax = x_center + 0.5 * width

    y_center = (ymin + ymax) / 2.
    height = (ymax - ymin) * expansion_factor
    ymin = y_center - 0.5 * height
    ymax = y_center + 0.5 * height

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def process_bbox(bbox, img_width, img_height, aspect_ratio=1, expansion_factor=1.25):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * expansion_factor
    bbox[3] = h * expansion_factor
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox


def get_aug_config(scale_factor=0.25, rot_factor=30, rot_prob=0.6, color_factor=0.2):
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= rot_prob else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

    return scale, rot, color_scale


def get_mf_aug_config(scale_factor=0.25, rot_factor=30, rot_prob=0.6, same_rot=True, color_factor=0.2):
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    if same_rot:
        rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= rot_prob else 0
        rot_list = [rot, rot, rot]
    else:
        rot_list = []
        for _ in range(3):
            rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= rot_prob else 0
            rot_list.append(rot)
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

    return scale, rot_list, color_scale


def augmentation(img, bbox, data_split, input_img_shape, scale_factor=0.25, rot_factor=30, rot_prob=0.6, color_factor=0.2, do_flip=False):
    if data_split == 'train':
        scale, rot, color_scale = get_aug_config(scale_factor, rot_factor, rot_prob, color_factor)
    else:
        scale, rot, color_scale = 1.0, 0.0, np.array([1, 1, 1])
    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, input_img_shape)

    img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, trans, inv_trans, rot, scale


def augmentation_multi_frames(img_list, bbox_list, data_split, input_img_shape, scale_factor=0.25, rot_factor=30, rot_prob=0.6, same_rot=True, color_factor=0.2, do_flip=False):
    if data_split == 'train':
        scale, rot_list, color_scale = get_mf_aug_config(scale_factor, rot_factor, rot_prob, same_rot, color_factor)
    else:
        scale, rot_list, color_scale = 1.0, [0.0, 0.0, 0.0], np.array([1, 1, 1])

    aug_img_list = []
    trans_list = []
    inv_trans_list = []
    for img, bbox, rot in zip(img_list, bbox_list, rot_list):
        img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, input_img_shape)
        img = np.clip(img * color_scale[None, None, :], 0, 255)
        aug_img_list.append(img)
        trans_list.append(trans)
        inv_trans_list.append(inv_trans)
    return aug_img_list, trans_list, inv_trans_list, rot_list, scale


def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape, flags=cv2.INTER_LINEAR):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=flags)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def chamfer_distance(x, y, metric='l2', direction='bi'):
    '''Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default l2
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||_metric}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||_metric}}
    '''

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError('Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'')

    return chamfer_dist


def np_inverse(g: np.ndarray):
    '''Returns the inverse of the SE3 transform

    Args:
        g: ([B,] 3/4, 4) transform

    Returns:
        ([B,] 3/4, 4) matrix containing the inverse

    '''
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    inv_rot = np.swapaxes(rot, -1, -2)
    inverse_transform = np.concatenate([inv_rot, inv_rot @ -trans[..., None]], axis=-1)
    if g.shape[-2] == 4:
        inverse_transform = np.concatenate([inverse_transform, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return inverse_transform


def np_concatenate(a: np.ndarray, b: np.ndarray):
    ''' Concatenate two SE3 transforms

    Args:
        a: First transform ([B,] 3/4, 4)
        b: Second transform ([B,] 3/4, 4)

    Returns:
        a*b ([B, ] 3/4, 4)

    '''

    r_a, t_a = a[..., :3, :3], a[..., :3, 3]
    r_b, t_b = b[..., :3, :3], b[..., :3, 3]

    r_ab = r_a @ r_b
    t_ab = r_a @ t_b[..., None] + t_a[..., None]

    concatenated = np.concatenate([r_ab, t_ab], axis=-1)

    if a.shape[-2] == 4:
        concatenated = np.concatenate([concatenated, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return concatenated