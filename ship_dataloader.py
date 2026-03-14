from torch.utils.data import DataLoader
from ship_dataset import ShipDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def get_data_loaders(data_dir,batch_size = 4,num_workers = 2):
    train_dataset = ShipDataset(root=data_dir,mode='train')
    valid_dataset = ShipDataset(root=data_dir,mode='valid')
    test_dataset = ShipDataset(root=data_dir,mode='test')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return train_dataloader,valid_dataloader,test_loader

