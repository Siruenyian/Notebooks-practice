import string
import torch
import torchvision.transforms as transforms


# train_transforms = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor()
# ])

# test_transforms = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor()
# ])

def setup(image_path:string, train_transforms:transforms.Compose, test_transforms:transforms.Compose):
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    train_data_custom = torch.utils.data.Dataset(targ_dir=train_dir, 
                                        transform=train_transforms)
    test_data_custom = torch.utils.data.Dataset(targ_dir=test_dir, 
                                        transform=test_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_data_custom, batch_size=1, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data_custom, batch_size=1, shuffle=False)
    return train_dataloader, test_dataloader