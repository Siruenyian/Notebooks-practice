



import torch
import torchvision.transforms as transforms
import Engine, Main,Model, Utils, DataSetup
if __name__ == "__main__":
    print("running")
    IMAGE_PATH="/app/dataset"
    device="cuda" if torch.cuda.is_available() else "cpu"
    device="cpu"
    train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    ])
    # I dont think you should abstract it wayy more like this, its just confusigng sometimes
    # test_dataloader, train_dataloader=DataSetup.setup(IMAGE_PATH, train_transforms, test_transforms)

    train_dir = IMAGE_PATH / "train"
    test_dir = IMAGE_PATH / "test"
    train_data_custom = torch.utils.data.Dataset(targ_dir=train_dir, 
                                        transform=train_transforms)
    test_data_custom = torch.utils.data.Dataset(targ_dir=test_dir, 
                                        transform=test_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_data_custom, batch_size=1, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data_custom, batch_size=1, shuffle=False)

    model=Model.TinyVGG(3,
    output_shape=len(train_dataloader.classes)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    history=Engine.train_withdataloader(10, model, train_dataloader, test_dataloader)
    MODEL_PATH="/app/Pytorch For DL/models/6_GoingModularModel.pth"
    torch.save(model.state_dict(), MODEL_PATH)