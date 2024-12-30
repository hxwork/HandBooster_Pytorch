import os
import argparse
from common.config import Config
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='', type=str, help='Directory containing cfg.json')
parser.add_argument('--resume', default=-1, type=int, help='index of model weights')


def fetch_model(cfg):
    if cfg.model.name in ['cf_normal_cond_v2_wloss_grcond']:
        from denoising_diffusion_pytorch.classifier_free_guidance_normal_wloss_grcond import Unet, GaussianDiffusion
        unet_model = Unet(dim=64, num_classes=8, dim_mults=(1, 2, 4, 8))
        diffusion_model = GaussianDiffusion(
            unet_model,
            image_size=cfg.data.image_size,
            timesteps=cfg.data.timesteps,  # number of steps
            sampling_timesteps=cfg.test.sampling_timesteps,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
            loss_type=cfg.loss.name,  # L1 or L2
            objective=cfg.loss.objective,
            beta_schedule=cfg.model.beta_schedule,
        )

    else:
        return NotImplementedError(f'No implement {cfg.model.name}')

    return diffusion_model


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'cfg.json')
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    cfg = Config(json_path).cfg

    # Update args into cfg.base
    cfg.base.update(vars(args))
    diffusion_model = fetch_model(cfg)
    trainer = Trainer(
        diffusion_model,
        dataset_name=cfg.data.dataset_name,
        train_batch_size=cfg.train.batch_size,
        train_lr=cfg.optimizer.lr,
        train_num_steps=cfg.train.num_steps,  # total training steps
        gradient_accumulate_every=cfg.train.gradient_accumulate_every,  # gradient accumulation steps
        ema_decay=cfg.train.ema_decay,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        calculate_fid=True,  # whether to calculate fid during training
        split_batches=True,
        save_and_sample_every=cfg.summary.save_and_sample_every,
        num_samples=cfg.test.num_samples,
        results_folder=cfg.base.model_dir,
        data_split=cfg.data.data_split,
        version=cfg.data.version,
    )

    trainer.load(milestone=cfg.base.resume)
    trainer.train()
