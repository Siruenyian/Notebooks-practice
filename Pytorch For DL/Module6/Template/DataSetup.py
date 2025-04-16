import string
import torch
import torchvision
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

def setup(train_dir:string,test_dir:string, transform:transforms.Compose, batch_size):
    train_data_custom = torchvision.datasets.ImageFolder(root=train_dir, 
                                        transform=transform)
    test_data_custom = torchvision.datasets.ImageFolder(root=test_dir, 
                                        transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data_custom, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data_custom, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader, train_data_custom.classes