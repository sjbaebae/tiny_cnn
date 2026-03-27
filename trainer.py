import torch
import numpy as np
import struct
import os
from pathlib import Path
import PIL.Image as Image
import torch.nn.functional as F

optimizer = torch.optim.AdamW(lr=1e-3, weight_decay=0.01)
loss = torch.nn.CrossEntropyLoss()

def read_idx(filename: Path) :
    with open(filename, "rb") as f:
        zero1, zero2, dtype_code, num_dims = struct.unpack(">BBBB", f.read(4))
        assert zero1 == 0 and zero2 == 0, "Invalid IDX file"
        
        dtype_map = {
            0x08: np.uint8,
            0x09: np.int8,
            0x0B: np.int16,
            0x0C: np.int32,
            0x0D: np.float32,
            0x0E: np.float64,
        }
        
        dtype = dtype_map[dtype_code]
        shape = struct.unpack(">" + "I" * num_dims, f.read(4 * num_dims))
        return np.fromfile(f, dtype=dtype).reshape(shape)

def load_data(data_dir: Path = Path("data")):
    train_images = read_idx(data_dir / "train-images-idx3-ubyte")
    train_labels = read_idx(data_dir / "train-labels-idx1-ubyte")
    test_images = read_idx(data_dir / "t10k-images-idx3-ubyte")
    test_labels = read_idx(data_dir / "t10k-labels-idx1-ubyte")
    
    # resize to 16x16 and normalize to 0~1

    train_images = torch.from_numpy(train_images).float() / 255.0   # (N,H,W,C)
    train_images = train_images.permute(0, 3, 1, 2)                            # (N,C,H,W)
    train_images = F.interpolate(train_images, size=(16, 16), mode='bilinear', align_corners=False)
    
    test_images = torch.from_numpy(test_images).float() / 255.0
    test_images = test_images.permute(0, 3, 1, 2)
    test_images = F.interpolate(test_images, size=(16, 16), mode='bilinear', align_corners=False)
    
    train_labels = torch.from_numpy(train_labels).long()
    test_labels = torch.from_numpy(test_labels).long()

    # use dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader
    
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss_val = loss_fn(y_pred, y)
        loss_val.backward()
        optimizer.step()
        total_loss += loss_val.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x)
            loss_val = loss_fn(y_pred, y)
            total_loss += loss_val.item()
    return total_loss / len(dataloader)
    