import torch
import numpy as np
import struct
import os
from pathlib import Path
import PIL.Image as Image
import torch.nn.functional as F

from torch_version.mnist_mlp import MLP

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

def load_data(data_dir: Path = Path("data"), flatten: bool = False):
    train_images = read_idx(data_dir / "train-images-idx3-ubyte")
    train_labels = read_idx(data_dir / "train-labels-idx1-ubyte")
    test_images = read_idx(data_dir / "t10k-images-idx3-ubyte")
    test_labels = read_idx(data_dir / "t10k-labels-idx1-ubyte")
    
    # normalize to 0~1. Image size of 28 is okay no need to do F.interpolate
    # channel size here is 1. So only 3 dims (N, H, W)

    train_images = torch.from_numpy(train_images).float() / 255.0   # (N,H,W)
    test_images = torch.from_numpy(test_images).float() / 255.0

    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
    
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
    

if __name__ == "__main__":
    train_loader, test_loader = load_data(flatten=True)
    IMAGE_SIZE = train_loader.dataset[0][0].shape[0]
    model = MLP(in_features=IMAGE_SIZE, out_features=10, hidden_features=128, num_hidden_layers=2, activation="relu")
    optimizer = torch.optim.AdamW(lr=1e-3, weight_decay=0.01, params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        test_loss = evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")