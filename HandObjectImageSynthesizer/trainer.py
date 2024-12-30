import math
import os
import logging

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from collections import namedtuple, defaultdict
from multiprocessing import cpu_count

import torch
import cv2
import copy
import imageio
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import utils
from einops import rearrange, repeat
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.1'

from denoising_diffusion_pytorch.tools import tensor_gpu
from dataset.DexYCB import DexYCB
from dataset.HO3D import HO3D
from hand_recon.loss.loss import filter_data_by_metric
from hand_recon.common.utils.mano import MANO
from hand_recon.common.utils.vis import render_badcase_mesh

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
# helpers functions
logger = logging.getLogger(__name__)

right_mano = MANO(side='right')
left_mano = MANO(side='left')


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num)**2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# classifier free guidance functions


def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class Trainer(object):

    def __init__(self,
                 diffusion_model,
                 dataset_name,
                 *,
                 train_batch_size=16,
                 gradient_accumulate_every=1,
                 train_lr=1e-4,
                 train_num_steps=100000,
                 ema_update_every=10,
                 ema_decay=0.995,
                 adam_betas=(0.9, 0.99),
                 save_and_sample_every=1000,
                 num_samples=25,
                 results_folder='./results',
                 amp=False,
                 fp16=False,
                 split_batches=True,
                 convert_image_to=None,
                 calculate_fid=True,
                 inception_block_idx=2048,
                 data_split='s0_train',
                 version='-1'):
        super().__init__()

        # accelerator
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))

        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision='fp16' if fp16 else 'no', kwargs_handlers=[kwargs])

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        self.data_split = data_split
        self.version = version
        if dataset_name == 'dexycb':
            self.ds = DexYCB(
                data_split=self.data_split,
                version=self.version,
                image_size=self.image_size,
                convert_image_to=convert_image_to,
            )
        elif dataset_name == 'ho3d':
            self.ds = HO3D(
                data_split=self.data_split,
                version=self.version,
                image_size=self.image_size,
                convert_image_to=convert_image_to,
            )
        else:
            raise NotImplementedError

        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        for name, param in diffusion_model.named_parameters():
            if name.startswith('hand_recon_model'):
                param.requires_grad = False

        # optimizer
        self.opt = Adam(filter(lambda p: p.requires_grad, diffusion_model.parameters()), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.model_folder = os.path.join(results_folder, 'model')
        os.makedirs(self.model_folder, exist_ok=True)

        self.summary_folder = os.path.join(results_folder, 'summary')
        os.makedirs(self.summary_folder, exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # NOTE also use training set for testing
        if dataset_name == 'dexycb':
            self.test_ds = DexYCB(
                data_split=self.data_split,
                version=self.version,
                image_size=self.image_size,
            )
        elif dataset_name == 'ho3d':
            self.test_ds = HO3D(
                data_split=self.data_split,
                version=self.version,
                image_size=self.image_size,
            )
        else:
            raise NotImplementedError
        test_dl = DataLoader(self.test_ds, batch_size=self.num_samples, shuffle=True, pin_memory=True, num_workers=cpu_count())
        self.test_dl = cycle(test_dl)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, os.path.join(self.model_folder, f'model-{milestone}.pt'))

    def load(self, milestone):
        if milestone == -1 or milestone == '-1':
            logging.info('Training from scratch')
            return None

        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(os.path.join(self.model_folder, f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'], strict=False)

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'], strict=False)


        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'], strict=False)

    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)

        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...').cpu().numpy()

        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):

        if self.channels == 1:
            real_samples, fake_samples = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3), (real_samples, fake_samples))

        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = tensor_gpu(next(self.dl), device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        milestone = self.step // self.save_and_sample_every

                        if milestone % 10 == 0:
                            with torch.no_grad():
                                input = tensor_gpu(next(self.test_dl), device)
                                all_rets = self.ema.ema_model.sample(input)

                            all_imgs = all_rets[0]
                            all_cond_imgs = all_rets[1]
                            all_rgbs = all_rets[2]

                            utils.save_image(all_imgs, os.path.join(self.summary_folder, f'sample-{milestone}-img.png'), nrow=int(math.sqrt(self.num_samples)))
                            utils.save_image(all_cond_imgs[:, :3, :, :], os.path.join(self.summary_folder, f'sample-{milestone}-seg.png'), nrow=int(math.sqrt(self.num_samples)))
                            utils.save_image(all_cond_imgs[:, 3:, :, :], os.path.join(self.summary_folder, f'sample-{milestone}-normal.png'), nrow=int(math.sqrt(self.num_samples)))
                            utils.save_image(all_rgbs, os.path.join(self.summary_folder, f'sample-{milestone}-gt.png'), nrow=int(math.sqrt(self.num_samples)))

                            img1 = cv2.imread(os.path.join(self.summary_folder, f'sample-{milestone}-gt.png'))[:, :, ::-1]
                            img2 = cv2.imread(os.path.join(self.summary_folder, f'sample-{milestone}-img.png'))[:, :, ::-1]
                            img3 = cv2.imread(os.path.join(self.summary_folder, f'sample-{milestone}-normal.png'))[:, :, ::-1]
                            with imageio.get_writer(os.path.join(self.summary_folder, f'sample-{milestone}.gif'), mode='I', duration=1) as writer:
                                writer.append_data(img1)
                                writer.append_data(img2)
                                writer.append_data(img3)

                            self.save(milestone)

                        self.save('latest')

                pbar.update(1)

        accelerator.print('training complete')


class Tester_DexYCB(object):

    def __init__(self,
                 diffusion_model,
                 hand_recon_model,
                 folder,
                 *,
                 batch_size=16,
                 ema_update_every=10,
                 ema_decay=0.995,
                 results_folder='./results',
                 split_batches=True,
                 convert_image_to=None,
                 data_split='placeholder',
                 version=-1,
                 part=1,
                 total_split=None,
                 current_split=None):
        super().__init__()

        self.folder = folder
        print(f'save folder: {self.folder}')

        # accelerator
        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision='no')

        # model
        self.model = diffusion_model
        self.hand_recon_model = hand_recon_model

        # EMA
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        # sampling and training hyperparameters
        self.batch_size = batch_size
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        self.data_split = data_split
        self.version = version
        self.part = part
        self.total_split = total_split
        self.current_split = current_split
        self.ds = DexYCB(data_split=self.data_split,
                         version=self.version,
                         part=self.part,
                         image_size=self.image_size,
                         convert_image_to=convert_image_to,
                         total_split=self.total_split,
                         current_split=self.current_split)
        dl = DataLoader(self.ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # step counter state
        self.step = 0
        self.batch_num = (len(self.ds) // self.batch_size)

        self.model_folder = os.path.join(results_folder, 'model')
        self.summary_folder = os.path.join(results_folder, 'summary')

        # prepare model, dataloader, optimizer with accelerator
        self.model = self.accelerator.prepare(self.model)
        self.hand_recon_model = self.accelerator.prepare(self.hand_recon_model)

    @property
    def device(self):
        return self.accelerator.device

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(os.path.join(self.model_folder, f'model-{milestone}.pt'), map_location=device)
        print(f"load DM model from {os.path.join(self.model_folder, f'model-{milestone}.pt')}")

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def inference_single_gpu(self):
        self.ema.ema_model.eval()
        self.hand_recon_model.eval()
        self.buf = defaultdict(list)
        self.cnt = 0
        while self.step < self.batch_num:
            input_data = next(self.dl)

            dm_output = self.ema.ema_model.sample(input_data, return_cond=False)

            # prepare input for hand reconstruction
            val_mano_pose = copy.deepcopy(input_data['mano_pose']).reshape(-1, 16, 3)
            val_mano_pose = val_mano_pose.reshape(-1, 48)
            # resize tensor if necessary
            ori_img_for_recon_eval = copy.deepcopy(input_data['img'])
            gen_img_for_recon_eval = copy.deepcopy(dm_output)
            if input_data['img'].size(2) != 128:
                ori_img_for_recon_eval = F.resize(ori_img_for_recon_eval, 128)
                gen_img_for_recon_eval = F.resize(gen_img_for_recon_eval, 128)
                out_shape = (256, 256)
            else:
                out_shape = (128, 128)
            hand_recon_input = {
                'ori_img': ori_img_for_recon_eval,
                'img': gen_img_for_recon_eval,
                'mano_pose': input_data['mano_pose'],
                'mano_shape': input_data['mano_shape'],
                'val_mano_pose': val_mano_pose,
            }
            # hand reconstruction inference
            hand_recon_output = self.hand_recon_model(hand_recon_input)

            # compute metric
            metric, verts = filter_data_by_metric(hand_recon_input, hand_recon_output, save_img=False)

            # gather some data
            all_dm_output = self.accelerator.gather_for_metrics(dm_output)
            all_input_data = self.accelerator.gather_for_metrics(input_data)
            all_metric = self.accelerator.gather_for_metrics(metric)
            all_verts = self.accelerator.gather_for_metrics(verts)

            # save data
            if self.accelerator.is_main_process:
                print(f'{self.step}/{self.batch_num}')
                # move data from gpu to cpu and from tensor to nparray
                all_dm_output = all_dm_output.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                all_input_data = tensor_gpu(all_input_data, check_on=False)
                all_metric = tensor_gpu(all_metric, check_on=False)
                all_verts = tensor_gpu(all_verts, check_on=False)

                # append generated img and metric
                tmp_dict = {}
                tmp_dict['img_id'] = all_input_data['img_id']
                tmp_dict['gen_img'] = all_dm_output
                tmp_dict.update(all_metric)

                # append one batch to buf
                for k, v in tmp_dict.items():
                    self.buf[k].append(v)

                self.cnt += len(all_input_data['img_id'])
                # NOTE save results
                save_results = False
                if save_results:
                    verts_ori, verts_gen, verts_gt = all_verts['verts_ori'], all_verts['verts_gen'], all_verts['verts_gt']
                    gen_score, gen_MPJPE, gen_MPVPE, gen_PA_MPJPE, gen_PA_MPVPE = all_metric['gen_score'], all_metric['gen_MPJPE'], all_metric['gen_MPVPE'], all_metric[
                        'gen_PA_MPJPE'], metric['gen_PA_MPVPE']
                    ori_score, ori_MPJPE, ori_MPVPE, ori_PA_MPJPE, ori_PA_MPVPE = all_metric['ori_score'], all_metric['ori_MPJPE'], all_metric['ori_MPVPE'], all_metric[
                        'ori_PA_MPJPE'], all_metric['ori_PA_MPVPE']
                    for i in range(gen_MPJPE.shape[0]):
                        condition = (gen_MPJPE[i] > 25) or (gen_MPVPE[i] > 25) or (gen_PA_MPJPE[i] > 10) or (gen_PA_MPVPE[i] > 10)

                        ori_img = all_input_data['img'][i].transpose(1, 2, 0) * 255
                        gen_img = all_dm_output[i]
                        img_fn = bytes(all_input_data['img_fn'][i].tolist()).decode()
                        do_flip = False

                        bbox = all_input_data['bbox'][i]
                        joints_coord_cam = all_input_data['joints_coord_cam'][i]
                        pred_ori_mesh = verts_ori[i]
                        pred_gen_mesh = verts_gen[i]
                        gt_mesh = verts_gt[i]
                        cam_param = all_input_data['cam_param'][i]

                        if do_flip:
                            pred_ori_mesh[:, 0] *= -1
                            pred_gen_mesh[:, 0] *= -1
                            gt_mesh[:, 0] *= -1
                            face = left_mano.face
                        else:
                            face = right_mano.face

                        pred_ori_mesh += joints_coord_cam[0, :]
                        pred_gen_mesh += joints_coord_cam[0, :]
                        gt_mesh += joints_coord_cam[0, :]

                        cat_img = render_badcase_mesh(ori_img, gen_img, bbox, pred_ori_mesh, pred_gen_mesh, gt_mesh, face, cam_param, do_flip, out_shape)
                        cat_img = cv2.resize(cat_img, (cat_img.shape[1] * 2, cat_img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
                        cat_img = cv2.copyMakeBorder(cat_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        text_area = np.zeros((100, cat_img.shape[1], 3), np.uint8)
                        text_area[:] = (255, 0, 180)
                        cat_img = cv2.vconcat((text_area, cat_img))

                        if condition:
                            save_path = os.path.join(f'./debug/grasp_{self.folder}/badcase', img_fn)
                        else:
                            save_path = os.path.join(f'./debug/grasp_{self.folder}/goodcase', img_fn)
                        print(save_path)

                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        cv2.putText(cat_img, f'Ori_sum: {ori_score[i]:.2f}, J: {ori_MPJPE[i]:.2f}, PA-J: {ori_PA_MPJPE[i]:.2f}, V: {ori_MPVPE[i]:.2f}, PA-V: {ori_PA_MPVPE[i]:.2f}',
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(cat_img, f'Gen_sum: {gen_score[i]:.2f}, J: {gen_MPJPE[i]:.2f}, PA-J: {gen_PA_MPJPE[i]:.2f}, V: {gen_MPVPE[i]:.2f}, PA-V: {gen_PA_MPVPE[i]:.2f}',
                                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.imwrite(save_path, cat_img)

                    # clear buf and cnt
                    self.buf = defaultdict(list)
                    self.cnt = 0
            self.step += 1
        print(f'{self.step}/{self.batch_num} compelete')


class Tester_HO3D(object):

    def __init__(self,
                 diffusion_model,
                 hand_recon_model,
                 folder,
                 *,
                 batch_size=16,
                 ema_update_every=10,
                 ema_decay=0.995,
                 results_folder='./results',
                 split_batches=True,
                 convert_image_to=None,
                 data_split='placeholder',
                 version=-1,
                 part=1,
                 total_split=None,
                 current_split=None):
        super().__init__()

        self.folder = folder
        print(f'save folder: {self.folder}')

        # accelerator
        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision='no')

        # model
        self.model = diffusion_model
        self.hand_recon_model = hand_recon_model

        # EMA
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        # sampling and training hyperparameters
        self.batch_size = batch_size
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        self.data_split = data_split
        self.version = version
        self.part = part
        self.total_split = total_split
        self.current_split = current_split
        self.ds = HO3D(data_split=self.data_split,
                       version=self.version,
                       part=self.part,
                       image_size=self.image_size,
                       convert_image_to=convert_image_to,
                       total_split=self.total_split,
                       current_split=self.current_split)
        dl = DataLoader(self.ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # step counter state
        self.step = 0
        self.batch_num = (len(self.ds) // self.batch_size)

        self.model_folder = os.path.join(results_folder, 'model')
        self.summary_folder = os.path.join(results_folder, 'summary')

        # prepare model, dataloader, optimizer with accelerator
        self.model = self.accelerator.prepare(self.model)
        self.hand_recon_model = self.accelerator.prepare(self.hand_recon_model)

    def img_fn_seq_idx_to_name(self, img_fn):
        all_seq_names = os.listdir(os.path.join('your_HO3D_dir', 'train'))
        seq_idx = img_fn.split('/')[0]
        seq_name = f'{all_seq_names[int(seq_idx)]}'
        img_fn = img_fn.replace(f'{seq_idx}/', f'{seq_name}/')
        return img_fn

    @property
    def device(self):
        return self.accelerator.device

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(os.path.join(self.model_folder, f'model-{milestone}.pt'), map_location=device)
        print(f"load DM model from {os.path.join(self.model_folder, f'model-{milestone}.pt')}")

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def inference_single_gpu(self):
        self.ema.ema_model.eval()
        self.hand_recon_model.eval()
        self.buf = defaultdict(list)
        self.cnt = 0
        while self.step < self.batch_num:
            input_data = next(self.dl)

            dm_output = self.ema.ema_model.sample(input_data, return_cond=False)

            # prepare input for hand reconstruction
            val_mano_pose = copy.deepcopy(input_data['mano_pose']).reshape(-1, 48)
            # resize tensor if necessary
            ori_img_for_recon_eval = copy.deepcopy(input_data['img'])
            gen_img_for_recon_eval = copy.deepcopy(dm_output)
            if input_data['img'].size(2) != 128:
                ori_img_for_recon_eval = F.resize(ori_img_for_recon_eval, 128)
                gen_img_for_recon_eval = F.resize(gen_img_for_recon_eval, 128)
                out_shape = (256, 256)
            else:
                out_shape = (128, 128)
            hand_recon_input = {
                'ori_img': ori_img_for_recon_eval,
                'img': gen_img_for_recon_eval,
                'mano_pose': input_data['mano_pose'],
                'mano_shape': input_data['mano_shape'],
                'val_mano_pose': val_mano_pose,
            }
            # hand reconstruction inference
            hand_recon_output = self.hand_recon_model(hand_recon_input)

            # compute metric
            metric, verts = filter_data_by_metric(hand_recon_input, hand_recon_output, save_img=False)

            # gather some data
            all_dm_output = self.accelerator.gather_for_metrics(dm_output)
            all_input_data = self.accelerator.gather_for_metrics(input_data)
            all_metric = self.accelerator.gather_for_metrics(metric)
            all_verts = self.accelerator.gather_for_metrics(verts)

            # save data
            if self.accelerator.is_main_process:
                print(f'{self.step}/{self.batch_num}')
                # move data from gpu to cpu and from tensor to nparray
                all_dm_output = all_dm_output.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                all_input_data = tensor_gpu(all_input_data, check_on=False)
                all_metric = tensor_gpu(all_metric, check_on=False)
                all_verts = tensor_gpu(all_verts, check_on=False)

                # append generated img and metric
                tmp_dict = {}
                tmp_dict['img_id'] = all_input_data['img_id']
                tmp_dict['gen_img'] = all_dm_output
                tmp_dict.update(all_metric)

                # append one batch to buf
                for k, v in tmp_dict.items():
                    self.buf[k].append(v)

                self.cnt += len(all_input_data['img_id'])
                # NOTE save results
                save_results = False
                if save_results:
                    verts_ori, verts_gen, verts_gt = all_verts['verts_ori'], all_verts['verts_gen'], all_verts['verts_gt']
                    gen_score, gen_MPJPE, gen_MPVPE, gen_PA_MPJPE, gen_PA_MPVPE = all_metric['gen_score'], all_metric['gen_MPJPE'], all_metric['gen_MPVPE'], all_metric[
                        'gen_PA_MPJPE'], metric['gen_PA_MPVPE']
                    ori_score, ori_MPJPE, ori_MPVPE, ori_PA_MPJPE, ori_PA_MPVPE = all_metric['ori_score'], all_metric['ori_MPJPE'], all_metric['ori_MPVPE'], all_metric[
                        'ori_PA_MPJPE'], all_metric['ori_PA_MPVPE']
                    for i in range(gen_MPJPE.shape[0]):
                        condition = (gen_MPJPE[i] > 50) or (gen_MPVPE[i] > 50) or (gen_PA_MPJPE[i] > 20) or (gen_PA_MPVPE[i] > 20)

                        ori_img = all_input_data['img'][i].transpose(1, 2, 0) * 255
                        gen_img = all_dm_output[i]
                        img_fn = self.img_fn_seq_idx_to_name(bytes(all_input_data['img_fn'][i].tolist()).decode())
                        bbox = all_input_data['bbox'][i]
                        joints_coord_cam = all_input_data['joints_coord_cam'][i]
                        pred_ori_mesh = verts_ori[i]
                        pred_gen_mesh = verts_gen[i]
                        gt_mesh = verts_gt[i]
                        cam_param = all_input_data['cam_param'][i]

                        face = right_mano.face

                        pred_ori_mesh += joints_coord_cam[0, :]
                        pred_gen_mesh += joints_coord_cam[0, :]
                        gt_mesh += joints_coord_cam[0, :]

                        cat_img = render_badcase_mesh(ori_img, gen_img, bbox, pred_ori_mesh, pred_gen_mesh, gt_mesh, face, cam_param, False, out_shape)
                        cat_img = cv2.resize(cat_img, (cat_img.shape[1] * 2, cat_img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
                        cat_img = cv2.copyMakeBorder(cat_img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        text_area = np.zeros((100, cat_img.shape[1], 3), np.uint8)
                        text_area[:] = (255, 0, 180)
                        cat_img = cv2.vconcat((text_area, cat_img))

                        if condition:
                            save_path = os.path.join(f'./debug/ho3d_{self.folder}/badcase', img_fn)
                        else:
                            save_path = os.path.join(f'./debug/ho3d_{self.folder}/goodcase', img_fn)
                        print(save_path)

                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        cv2.putText(cat_img, f'Ori_sum: {ori_score[i]:.2f}, J: {ori_MPJPE[i]:.2f}, PA-J: {ori_PA_MPJPE[i]:.2f}, V: {ori_MPVPE[i]:.2f}, PA-V: {ori_PA_MPVPE[i]:.2f}',
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(cat_img, f'Gen_sum: {gen_score[i]:.2f}, J: {gen_MPJPE[i]:.2f}, PA-J: {gen_PA_MPJPE[i]:.2f}, V: {gen_MPVPE[i]:.2f}, PA-V: {gen_PA_MPVPE[i]:.2f}',
                                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.imwrite(save_path, cat_img)

                    # clear buf and cnt
                    self.buf = defaultdict(list)
                    self.cnt = 0
            self.step += 1
        print(f'{self.step}/{self.batch_num} compelete')
