import argparse
import tqdm
import copy
import wandb
import torch.nn as nn
from utils.measure import *
from utils.ema import EMA
from models.basic_template import TrainTask
from .marcodiff_wrapper import Network, WeightNet
from .diffusion_modules import Diffusion


class marcodiff(TrainTask):
    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument("--in_channels", default=3, type=int)
        parser.add_argument("--out_channels", default=1, type=int)
        parser.add_argument("--init_lr", default=2e-4, type=float)

        parser.add_argument('--update_ema_iter', default=10, type=int)
        parser.add_argument('--start_ema_iter', default=2000, type=int)
        parser.add_argument('--ema_decay', default=0.995, type=float)

        parser.add_argument('--T', default=10, type=int)

        return parser

    def set_model(self):
        opt = self.opt
        self.ema = EMA(opt.ema_decay)
        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.T = opt.T

        denoise_fn = Network(in_channels=opt.in_channels)

        model = Diffusion(
            denoise_fn=denoise_fn,
            timesteps=opt.T
        )

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), opt.init_lr)
        ema_model = copy.deepcopy(model)

        self.logger.modules = [model, ema_model, optimizer]
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model

        self.lossfn = nn.MSELoss()
        self.lossfn_sub1 = nn.MSELoss()

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, inputs, n_iter):
        opt = self.opt
        self.model.train()
        self.ema_model.train()
        low_dose, full_dose = inputs
        low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

        gen_full_dose, x_mix, gen_full_dose_sub1, x_mix_sub1 = self.model(
            low_dose, full_dose
        )

        loss = 0.4 * self.lossfn(gen_full_dose, full_dose) + 0.6 * self.lossfn_sub1(gen_full_dose_sub1, full_dose)
        loss.backward()

        if opt.wandb:
            if n_iter == opt.resume_iter + 1:
                wandb.init(project="marcodiff")

        self.optimizer.step()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        loss = loss.item()
        self.logger.msg([loss, lr], n_iter)

        if opt.wandb:
            wandb.log({'epoch': n_iter, 'loss': loss})

        if n_iter % self.update_ema_iter == 0:
            self.step_ema(n_iter)

        return loss

    def denormalize_(self, image):
        image = image * (3072.0 + 1024.0) - 1024.0
        return image

    def trunc(self, mat):
        mat[mat <= -160.0] = -160.0
        mat[mat >= 240.0] = 240.0
        return mat

    @torch.no_grad()
    def test(self, n_iter):
        opt = self.opt
        self.ema_model.eval()

        psnr, ssim, rmse = 0., 0., 0.

        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

            if hasattr(self.ema_model, 'module'):
                gen_full_dose, direct_recons, imstep_imgs = self.ema_model.module.sample(
                    batch_size=low_dose.shape[0],
                    img=low_dose,
                    t=self.T
                )
            else:
                gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                    batch_size=low_dose.shape[0],
                    img=low_dose,
                    t=self.T
                )

            full_dose = self.trunc(self.denormalize_(full_dose))
            gen_full_dose = self.trunc(self.denormalize_(gen_full_dose))

            # data_range = full_dose.max() - full_dose.min()
            psnr_score, ssim_score, rmse_score = compute_measure(full_dose, gen_full_dose, 400)
            psnr += psnr_score / len(self.test_loader)
            ssim += ssim_score / len(self.test_loader)
            rmse += rmse_score / len(self.test_loader)

        print('psnr:{:.4f}, ssim:{:.4f}, rmse:{:.4f}'.format(psnr, ssim, rmse))

        self.logger.msg([psnr, ssim, rmse], n_iter)

        if opt.wandb:
            wandb.log({'epoch': n_iter, 'PSNR': psnr, 'SSIM': ssim, 'RMSE': rmse})

    @torch.no_grad()
    def generate_images(self, n_iter):
        opt = self.opt
        self.ema_model.eval()
        n = 0
        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
            if hasattr(self.ema_model, 'module'):
                gen_full_dose, direct_recons, imstep_imgs = self.ema_model.module.sample(
                    batch_size=low_dose.shape[0],
                    img=low_dose,
                    t=self.T,
                    n_iter=n_iter,
                )
            else:
                gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                    batch_size=low_dose.shape[0],
                    img=low_dose,
                    t=self.T,
                    n_iter=n_iter,
                )

            gen_full_dose = self.transfer_display_window(gen_full_dose)
            self.logger.save_image(gen_full_dose,
                                   n_iter,
                                   'result_{}'.format(n))
            n = n + 1
