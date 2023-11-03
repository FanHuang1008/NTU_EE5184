import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset




class FoodDataset(Dataset):
    def __init__(self,path,tfm,files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label
    

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


    
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    num_epoch = 50
    patience = 10
    _dataset_dir = "/content/gdrive/MyDrive/LeeMLData/food11/"
    _exp_name = "sample"

    train_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.75),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomGrayscale(0.5),
        transforms.RandomRotation(15),
        transforms.RandomInvert(p=0.5),
        transforms.ToTensor()
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    print("Start Training")
    stale = 0
    best_acc = 0
    for epoch in range(num_epoch):
        train_loss, train_acc = [], []

        model.train()
        for batch in tqdm(train_loader):
            imgs, labels = batch
            logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()  # clear gradients
            loss.backward()        # compute gradients
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10) 
            optimizer.step()       # update the parameters
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_acc.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_acc) / len(train_acc)
        print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        #-------------- Validation --------------#
        valid_loss, valid_acc = [], []

        model.eval()
        for batch in tqdm(val_loader):
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_acc.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_acc) / len(valid_acc)
        print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment in {patience} consecutive epochs, early stopping")
                break
    

    print("Finish Training")

    # ---------- Testing ----------
    test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model_best = Classifier().to(device)
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.eval()
    prediction = []
    with torch.no_grad():
        for data,_ in test_loader:
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

    def pad4(i):
        return "0"*(4-len(str(i)))+str(i)
    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
    df["Category"] = prediction
    df.to_csv("submission.csv",index = False)



if __name__ == "__main__":
    main()


class Residual(nn.Module):
    def __init__(self):
        super(Residual, self).__init__()
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
        )
        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256*32*32, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        x1 = self.cnn_layer1(x)
        x1 = nn.ReLU(x1)
        residual = x1

        x2 = self.cnn_layer2(x1)
        x2 = x2 + residual
        x2 = nn.ReLU(x2)

        x3 = self.cnn_layer3(x2)
        x3 = nn.ReLU(x3)
        residual = x3

        x4 = self.cnn_layer4(x3)
        x4 = x4 + residual
        x4 = nn.ReLU(x4)

        x5 = self.cnn_layer5(x4)
        x5 = nn.ReLU(x5)
        residual = x5

        x6 = self.cnn_layer6(x5)
        x6 = x6 + residual
        x6 = nn.ReLU(x6)

        xout = torch.flatten(x6)
        xout = self.fc(xout)
        return xout

