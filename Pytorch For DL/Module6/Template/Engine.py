import string
from tqdm.auto import tqdm
import torch
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=tinyvgg.parameters(), 
#                             lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_withdataloader(epochs:int, model:torch.nn.Module, train_dataloader, test_dataloader, loss_fn, accuracy_fn, optimizer, device:string):
    # epochs=15
    train_loss_values=[]
    test_loss_values=[]
    train_acc_values=[]
    test_acc_values=[]
    epoch_count =[]
    for epoch in tqdm(range(epochs)):
        print(f"epoch: {epoch}")
        model.to(device)
        model.train()
        train_loss, train_acc = 0, 0
        test_loss, test_acc = 0, 0
        for batch, (X,y) in enumerate(train_dataloader):
            X,y=X.to(device),y.to(device)
            logits=model(X)
            loss=loss_fn(logits,y)
            train_loss+=loss
            
            train_acc+=accuracy_fn(y, logits.argmax(dim=1))
            # zero grad
            optimizer.zero_grad()
            # loss backward
            loss.backward()
            # backprop
            optimizer.step()
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        model.to(device)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
        model.eval()
        with torch.inference_mode():
            for X,y in test_dataloader:
                X, y = X.to(device), y.to(device)
                test_logits=model(X)
                test_loss+=loss_fn(test_logits,y)
                test_acc+=accuracy_fn(y, test_logits.argmax(dim=1))
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        train_loss_values.append(train_loss.detach().numpy())
        train_acc_values.append(train_acc)
        test_loss_values.append(test_loss.detach().numpy())
        test_acc_values.append(test_acc)
        epoch_count.append(epoch)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")
    return {"train_loss": train_loss_values,
             "train_acc": train_acc_values,
             "test_loss": test_loss_values,
             "test_acc": test_acc_values}
