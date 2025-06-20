import torch
import os.path as osp
import os
import time
import inspect
from torchvision.utils import save_image
from utils.ops import reduce_tensor, load_network


def get_varname(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class LoggerX(object):

    def __init__(self, save_root):
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        self.world_size = 1

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            if isinstance(modules[i], torch.nn.Module):
                self._modules.append(modules[i])
                self._module_names.append(get_varname(modules[i]))

    def checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            if hasattr(module,'module'):
                network = module.module
                torch.save(network.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))
            else:
                torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            if hasattr(module, 'module'):
                network = module.module
                network.load_state_dict(
                    load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))
            else:
                module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        print(output_str)

    def save_image(self, grid_img, n_iter, sample_type):
        save_image(grid_img, osp.join(osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))), 'result'),
                                      '{}_{}.png'.format(n_iter, sample_type)),
                   nrow=1)