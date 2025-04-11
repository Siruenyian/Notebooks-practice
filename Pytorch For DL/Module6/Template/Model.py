import torch

class TinyVGG(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.block=torch.nn.Sequential(
            torch.nn.Conv2d(input_shape,10,3,1,1),
            torch.nn.Conv2d(10,10,3,1,1),
            torch.nn.ReLU(),
            # 
            torch.nn.Conv2d(10,10,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            # 
            torch.nn.Conv2d(10,10,3,1,1),
            torch.nn.ReLU(),
            # 
            torch.nn.Conv2d(10,10,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            # 
            torch.nn.Flatten(),
            torch.nn.Linear(32*80,out_features=output_shape)
            )

    def forward(self, x):
        return self.block(x)
