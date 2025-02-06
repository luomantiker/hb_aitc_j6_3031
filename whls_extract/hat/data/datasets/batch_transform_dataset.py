import random
from typing import List

from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class BatchTransformDataset(Dataset):
    """Dataset which uses different transforms in different epochs.

    Args:
        dataset: Target dataset.
        transforms_cfgs: The list of different transform configs.
        epoch_steps: Effective epoch of different transforms.
    """

    def __init__(
        self,
        dataset: Dataset,
        transforms_cfgs: List,
        epoch_steps: List,
    ):
        super(BatchTransformDataset, self).__init__()
        self.dataset = dataset
        if not hasattr(self.dataset, "getdata"):
            assert "dataset not support"
        self.transforms_steps = transforms_cfgs
        self.epoch_steps = epoch_steps
        self.current_epoch = 0
        self.current_transforms = self.transforms_steps[0]

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        for i, epoch_step in enumerate(self.epoch_steps):
            if self.current_epoch == epoch_step:
                print("Change transformer to {}".format(i + 1))
                self.current_transforms = self.transforms_steps[i + 1]

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, item):
        data = self.dataset.__getitem__(item)
        datalist = [data]

        image_needed = 1
        for trans in self.current_transforms:
            if hasattr(trans, "images_needed"):
                needs = trans.images_needed
            else:
                needs = 1
            image_needed *= needs
        indexes = [
            random.randint(0, self.__len__() - 1)
            for _ in range(image_needed - 1)
        ]
        for i in indexes:
            data_addition = self.dataset[i]
            datalist.append(data_addition)
        for trans in self.current_transforms:
            new_datalist = []
            if hasattr(trans, "images_needed"):
                needs = trans.images_needed
            else:
                needs = 1
            for i in range(0, len(datalist), needs):
                if needs == 1:
                    data = trans(datalist[i])
                else:
                    data = trans(datalist[i : i + needs])
                new_datalist.append(data)
            datalist = new_datalist

        return data
