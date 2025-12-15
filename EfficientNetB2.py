import random
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import models
from pathlib import Path
from torchvision import transforms
from sklearn.metrics import accuracy_score

#parameters
DATA_DIR = Path("./dog-breed-identification")
OUTPUT_DIR = Path("./outputs"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "efficientnet_b2"
BATCH_SIZE = 24         # 16~24
EPOCHS = 25
LR = 3e-4               #1e-3~3e-4
VAL_RATIO= 0.1
NUM_WORKERS = 0         #4~8；macOS/MPS 0~2

random.seed(42)
torch.manual_seed(42)

#split train/val
    #breed轉成數字標籤
    #先分 不是dataset後 -> 保留完整標籤資訊、方便日後驗證或 ensemble
labels_df = pd.read_csv(DATA_DIR / "labels.csv")  # 兩欄：id, breed
breeds = sorted(labels_df["breed"].unique())
breed2idx = {b:i for i,b in enumerate(breeds)} #文字轉成數字標籤（label）
num_classes = len(breeds)
print(num_classes)#->120

labels_df["label"] = labels_df["breed"].map(breed2idx)  #_df:"id","breed","label"
val_df = labels_df.sample(frac=VAL_RATIO, random_state=32)
train_df = labels_df.drop(val_df.index).reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


#Dataset
class DogDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / f"{row['id']}.jpg").convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(row["label"], dtype=torch.long)


    def __len__(self):
        return len(self.df)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.ids = sorted([p.stem for p in img_dir.glob("*.jpg")])
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = Image.open(self.img_dir / f"{img_id}.jpg").convert("RGB")
        img = self.transform(img)
        return img_id, img

    def __len__(self):
        return len(self.ids)
    
#transform / 自動配置所需之transform    
weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
#transforms_train = weights.transforms()
transforms_train = transforms.Compose([
    #transforms.RandomResizedCrop(260, scale=(0.8, 1.0)),
    transforms.Resize((260, 260)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
    transforms.RandomRotation(10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    #transforms.RandomErasing(p=0.25)
])
transforms_val =transforms.Compose([
    transforms.Resize((260, 260)),
    #transforms.CenterCrop(260), #先放大再切 260，避免變形、讓輸入更穩定
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ]) 

#DataLoader
train_dataset = DogDataset(train_df, DATA_DIR/"train", transforms_train)
val_dataset   = DogDataset(val_df,   DATA_DIR/"train", transforms_val)
test_dataset   = TestDataset(DATA_DIR/"test",  transforms_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS)

#device model loss optimizer
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

model = models.efficientnet_b2(weights=weights)
in_features = model.classifier[1].in_features     # EfficientNet 分類頭在 classifier[1]
model.classifier[1] = nn.Linear(in_features, 120)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

#optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) #自動調整 optimizer 的學習率    
#更改成依eopch數先凍結再解凍 所以放置main

#Train
def train_one_epoch(epoch,optimizer):
    model.train()
    train_loss=0.0
    total=0

    for images,labels in tqdm(train_loader,desc=f"training Epoch {epoch+1}/{EPOCHS}", ncols=80 ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        t_loss = criterion(outputs, labels)
        t_loss.backward()
        optimizer.step()

        train_loss+=t_loss.item()*images.size(0)
        total+=images.size(0)
    return train_loss/total

#Val
@torch.no_grad()
def validate(epoch):
    model.eval()
    val_loss=0.0
    total=0
    y_true,y_pred=[],[]
    for images,labels in tqdm(val_loader,desc=f"Validating Epoch {epoch+1}/{EPOCHS}", ncols=80 ):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        v_loss = criterion(outputs, labels)

        val_loss+=v_loss.item()*images.size(0)
        total+=images.size(0)

        y_true.extend(labels.cpu().tolist())
        tmp_pred = outputs.argmax(dim=1) # tmp_pred 的型態是 tensor
        y_pred.extend(tmp_pred.tolist())
    return val_loss/total,y_true,y_pred

#Train
def train():
    train_losses, val_losses = [], []
    best_acc=0.0
    best_loss = float("inf")

    for p in model.features.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier[1].parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(EPOCHS):
        if epoch==2:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        train_loss=train_one_epoch(epoch,optimizer)
        val_loss,y_true,y_pred=validate(epoch)
        scheduler.step(val_loss)    #每跑完1個epoch更新LR ->LR逐步下降 幫助收斂
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

        val_acc=accuracy_score(y_true,y_pred)
        print("val_acc=",val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc

            ckpt = {
                "model_state": model.state_dict(),
                "breed2idx": breed2idx,
                "weights": "EfficientNet_B2_Weights.IMAGENET1K_V1",
                "best_loss": best_loss,
                "best_acc": best_acc,
                "epoch": epoch + 1,
            }
            torch.save(ckpt, OUTPUT_DIR / "efficient_B2_best.pth")
            print(f"New best: val_loss={best_loss:.4f}, val_acc={best_acc:.4f} -> saved to outputs/efficient_B2_best.pth")

    # 畫 loss 的歷史記錄圖
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

#Test
def test():
    ckpt = torch.load(OUTPUT_DIR / "efficient_B2_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ids, all_probs = [], []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for img_ids, imgs in test_loader:
            imgs = imgs.to(device)

            outputs = model(imgs)
            probs1 = softmax(outputs)

            # HFlip TTA
            imgs_flip = torch.flip(imgs, dims=[3]) # [B,C,H,W] 在 W 維翻轉
            outputs_flip = model(imgs_flip)
            probs2 = softmax(outputs_flip)

            probs = 0.5 * (probs1 + probs2)

            ids.extend(img_ids)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)   # [N_test, num_classes]

    # 對齊 submission 欄位
    breed_names_sorted = sorted(breed2idx.keys())
    col_indices = [breed2idx[b] for b in breed_names_sorted]
    probs = probs[:, col_indices]

    sub = pd.DataFrame(probs, columns=breed_names_sorted)
    sub.insert(0, "id", ids)

    sub.to_csv(OUTPUT_DIR / "submission_efficient_B2.csv", index=False)
    print("Saved submission_efficient_B2.csv")


if __name__ == "__main__":
    test()

    
