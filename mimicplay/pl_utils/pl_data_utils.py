from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from mimicplay.scripts.aloha_process.simarUtils import nds

class DualDataModuleWrapper(LightningDataModule):
    """
    Same as DataModuleWrapper but there are two train datasets and two valid datasets
    """
    def __init__(
        self,
        train_dataset1,
        valid_dataset1,
        train_dataset2,
        valid_dataset2,
        train_dataloader_params,
        valid_dataloader_params,
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        super().__init__()
        self.train_dataset1 = train_dataset1
        self.valid_dataset1 = valid_dataset1
        self.train_dataset2 = train_dataset2
        self.valid_dataset2 = valid_dataset2
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params

    def train_dataloader(self):
        new_dataloader1 = DataLoader(dataset=self.train_dataset1, **self.train_dataloader_params)
        new_dataloader2 = DataLoader(dataset=self.train_dataset2, **self.train_dataloader_params)
        return [new_dataloader1, new_dataloader2]
    
    def val_dataloader_1(self):
        new_dataloader = DataLoader(dataset=self.valid_dataset1, **self.valid_dataloader_params)
        return new_dataloader

    def val_dataloader_2(self):
        new_dataloader = DataLoader(dataset=self.valid_dataset2, **self.valid_dataloader_params)
        return new_dataloader
    # def val_dataloader(self):
    #     new_dataloader1 = DataLoader(dataset=self.valid_dataset1, **self.valid_dataloader_params)
    #     new_dataloader2 = DataLoader(dataset=self.valid_dataset2, **self.valid_dataloader_params)
    #     return [new_dataloader1, new_dataloader2]

class DataModuleWrapper(LightningDataModule):
    """
    Wrapper around a LightningDataModule that allows for the data loader to be refreshed
    constantly.
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_dataloader_params,
        valid_dataloader_params,
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params

    def train_dataloader(self):
        new_dataloader = DataLoader(dataset=self.train_dataset, **self.train_dataloader_params)
        return new_dataloader
    
    def val_dataloader_1(self):
        new_dataloader = DataLoader(dataset=self.valid_dataset, **self.valid_dataloader_params)
        return new_dataloader

