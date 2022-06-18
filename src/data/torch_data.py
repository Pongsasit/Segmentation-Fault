from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self):
        # TODO
        self.data = []
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO
        return self.data[idx]

def get_data(batch_size: int = 32):
    """
    Args:
        batch_size: int

    Return:
        train_dataloader: DataLoader
        val_dataloader: DataLoader
    """
    train_dataset = CustomDataset()
    val_dataset = CustomDataset()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader

def get_test_data(batch_size: int = 32):
    """
    Args:
        batch_size: int

    Return:
        test_dataloader: DataLoader
    """
    test_dataset = CustomDataset()

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return test_dataloader