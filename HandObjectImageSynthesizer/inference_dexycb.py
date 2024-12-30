import os
import argparse

from common.config import Config
from trainer import Tester_DexYCB
from train import fetch_model
import sys

sys.path.append('./')
from hand_recon.model.model import fetch_model as fetch_hand_recon_model

parser = argparse.ArgumentParser('Inference', add_help=True)
parser.add_argument('--model_dir', default='', type=str, help='Directory containing cfg.json')
parser.add_argument('--data_split', default='', type=str, help='data split')
parser.add_argument('--total_split', '-ts', type=int, default=0, help='total split of data list')
parser.add_argument('--current_split', '-pi', type=int, default=1000, help='current split of data list')
parser.add_argument('--version', '-v', type=int, default=1, help='generation version')
parser.add_argument('--part', '-pt', type=int, default=1, help='generation part')
parser.add_argument('--sampling_timesteps', '-t', type=int, default=250, help='sample step')
parser.add_argument('--batch_size', '-bs', type=int, default=25, help='batch size')
parser.add_argument('--resume', default=-1, type=int, help='index of model weights')
args = parser.parse_args()

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'cfg.json')
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    cfg = Config(json_path).cfg

    # Update args into cfg.test
    cfg.test.update(vars(args))
    diffusion_model = fetch_model(cfg)

    # hand reconstruction model
    hand_recon_model = fetch_hand_recon_model(model_name='mobrecon', model_path='./hand_recon/experiment/dexycb.mobrecon.ori/filter/test_model_best.pth')

    tester = Tester_DexYCB(
        diffusion_model=diffusion_model,
        hand_recon_model=hand_recon_model,
        folder=
        f'{cfg.test.data_split}_name_{cfg.model.name}_condition_v{cfg.test.version}_p{cfg.test.part}_res_{cfg.data.image_size}_resume_{cfg.test.resume}_step_{cfg.test.sampling_timesteps}',
        batch_size=cfg.test.batch_size,
        split_batches=True,
        data_split=cfg.test.data_split,
        version=cfg.test.version,
        part=cfg.test.part,
        total_split=cfg.test.total_split,
        current_split=cfg.test.current_split,
        results_folder=cfg.base.model_dir,
    )

    tester.load(milestone=cfg.test.resume)
    tester.inference_single_gpu()
