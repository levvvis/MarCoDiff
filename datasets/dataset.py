import os.path as osp
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from functools import partial


class CTDataset(Dataset):
    def __init__(self, mode, test_id=9):
        self.mode = mode

        if mode == 'train':
            data_root = ''
        else:
            data_root = ''

        patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]
        if mode == 'train':
            patient_ids.pop(test_id)
        elif mode == 'test':
            patient_ids = patient_ids[test_id:test_id + 1]

        patient_lists = []
        for ind, id in enumerate(patient_ids):
            patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_'.format(id) + '*_target.npy'))))
            patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
        base_target = patient_lists

        patient_lists = []
        for ind, id in enumerate(patient_ids):
            patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_'.format(id) + '*_input.npy'))))
            cat_patient_list = []
            for i in range(1, len(patient_list) - 1):
                patient_path = ''
                for j in range(-1, 2):
                    patient_path = patient_path + '~' + patient_list[i + j]
                cat_patient_list.append(patient_path)
            patient_lists = patient_lists + cat_patient_list
        base_input = patient_lists

        self.input = base_input
        self.target = base_target

        print(len(self.input))
        print(len(self.target))

    def __getitem__(self, index):
        input, target = self.input[index], self.target[index]

        input = input.split('~')
        inputs = []
        for i in range(1, len(input)):
            inputs.append(np.load(input[i])[np.newaxis, ...].astype(np.float32))
        input = np.concatenate(inputs, axis=0)  # (3, 512, 512)

        target = np.load(target)[np.newaxis, ...].astype(np.float32)  # (1, 512, 512)

        return input, target

    def __len__(self):
        return len(self.target)


dataset_dict = {
    'train': partial(CTDataset, mode='train', test_id=9),
    'test': partial(CTDataset, mode='test', test_id=9)
}
