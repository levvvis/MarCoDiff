import math
import torch
from torch import nn


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_alpha_schedule(timesteps, s=0.008):
    t = torch.linspace(0, timesteps - 1, timesteps)
    t = t / (timesteps - 1)  # [0, 1]
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return torch.clip(alphas_cumprod, 0.0, 0.9999)


class Diffusion(nn.Module):
    def __init__(self,
                 denoise_fn=None,
                 channels=1,
                 timesteps=10
                 ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)

        alphas_cumprod = cosine_alpha_schedule(timesteps)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('one_minus_alphas_cumprod', 1. - alphas_cumprod)

    #  mean-preserving degradation operator
    def q_sample(self, x_start, x_end, t):
        return (
                extract(self.alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )

    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
                (xt - extract(self.alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    @torch.no_grad()
    def sample(self, batch_size=4, img=None, t=None):
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps

        up_img = img[:, 0].unsqueeze(1)
        down_img = img[:, 2].unsqueeze(1)
        img = img[:, 1].unsqueeze(1)

        noise = img
        x1_bar = img
        direct_recons = []
        imstep_imgs = []

        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)

            full_img = torch.cat((up_img, img, down_img), dim=1)

            if t == self.num_timesteps:
                adjust = False
            else:
                adjust = True

            x1_bar = self.denoise_fn(full_img, step, x1_bar, noise, adjust=adjust)
            x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            img = img - xt_bar + xt_sub1_bar

            direct_recons.append(x1_bar)
            imstep_imgs.append(img)
            t = t - 1

        return img.clamp(0., 1.), torch.stack(direct_recons), torch.stack(imstep_imgs)

    def forward(self, x, y):
        b, c, h, w, device, = *y.shape, y.device

        t_single = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        t = t_single.repeat((b,))

        x_end = x[:, 1].unsqueeze(1)
        x_mix = self.q_sample(x_start=y, x_end=x_end, t=t)
        x_mix = torch.cat((x[:, 0].unsqueeze(1), x_mix, x[:, 2].unsqueeze(1)), dim=1)

        # stage I
        x_recon = self.denoise_fn(x_mix, t, y, x_end, adjust=False)

        # stage II
        if t_single.item() >= 1:
            t_sub1 = t - 1
            t_sub1[t_sub1 < 0] = 0

            x_mix_sub1 = self.q_sample(x_start=x_recon, x_end=x_end, t=t_sub1)
            x_mix_sub1 = torch.cat((x[:, 0].unsqueeze(1), x_mix_sub1, x[:, 2].unsqueeze(1)), dim=1)

            x_recon_sub1 = self.denoise_fn(x_mix_sub1, t_sub1, x_recon, x_end, adjust=True)
        else:
            x_recon_sub1, x_mix_sub1 = x_recon, x_mix

        return x_recon, x_mix, x_recon_sub1, x_mix_sub1
