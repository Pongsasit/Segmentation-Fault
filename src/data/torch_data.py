from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

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