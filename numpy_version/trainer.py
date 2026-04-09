import numpy as np
import struct
import os
from pathlib import Path
import PIL.Image as Image
from tensor import Tensor
# from mnist_cnn import CNN_Base
from mnist_mlp import MLP
# add optimizers
from optimizer import AdamW

# following pytorch convention. TensorDataset basically just aligns each x with y along the same index
#  and provides __getitem__ and __len__. Passes it down to numpy
class TensorDataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.num_samples = len(data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


# Basic Dataloader following pytorch convention. Just randomly sample at each iteration. Provide batches per __next__ in iter
class DataLoader:
    def __init__(self, dataset: TensorDataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.indices = np.arange(self.num_samples)
        # batch index
        self.current_idx = 0

    def __iter__(self):
        if self.shuffle:
            # randomly shuffle each iteration
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch_data = self.dataset[batch_indices]
        self.current_idx += self.batch_size
        return batch_data

    def __len__(self):
        return self.num_samples

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

def load_data(data_dir: Path = Path("../data"), flatten: bool = False):
    train_images = read_idx(data_dir / "train-images-idx3-ubyte")
    train_labels = read_idx(data_dir / "train-labels-idx1-ubyte")
    test_images = read_idx(data_dir / "t10k-images-idx3-ubyte")
    test_labels = read_idx(data_dir / "t10k-labels-idx1-ubyte")
    
    # normalize to 0~1. Image size of 28 is okay no need to do F.interpolate
    # channel size here is 1. So only 3 dims (N, H, W)

    train_images = Tensor(train_images).unsqueeze(1).float() / 255.0   # (N,H,W)
    test_images = Tensor(test_images).unsqueeze(1).float() / 255.0

    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

    
    train_labels = Tensor(train_labels).long()
    test_labels = Tensor(test_labels).long()

    # use dataloaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader
    
def train(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        x, y = batch
        optimizer.zerograd()
        y_pred = model(x)
        loss_val = loss_fn(y_pred, y)
        print(f"Batch: {batch_idx}, Loss: {loss_val.item()}")
        loss_val.backward()
        optimizer.step()
        total_loss += loss_val.item()

        # institute early stopping if loss is less than 0.1
        if loss_val.item() < 0.1:
            print("Early stopping triggered")
            break
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    for batch in dataloader:
        x, y = batch
        with torch.no_grad():
            y_pred = model(x)
            loss_val = loss_fn(y_pred, y)
            total_loss += loss_val.item()
    return total_loss / len(dataloader)
    

if __name__ == "__main__":
    # train MLP
    train_loader, test_loader = load_data(flatten=True)
    IMAGE_SIZE = train_loader.dataset[0][0].shape[0]
    model = MLP(in_features=IMAGE_SIZE, out_features=10, hidden_features=128, num_hidden_layers=2, activation="relu")
    optimizer = AdamW(lr=1e-3, weight_decay=0.01, params=model.parameters())
    from nn.losses import CrossEntropyLoss
    loss_fn = CrossEntropyLoss()
    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, loss_fn)
        test_loss = evaluate(model, test_loader, loss_fn)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

    # train CNN
    # train_loader, test_loader = load_data(flatten=False)
    # model = CNN_Base(in_channels=1, out_channels=10)
    # optimizer = torch.optim.AdamW(lr=1e-3, weight_decay=0.01, params=model.parameters())
    # loss_fn = torch.nn.CrossEntropyLoss()
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # model.to(device)
    # for epoch in range(10):
    #     train_loss = train(model, train_loader, optimizer, loss_fn, device)
    #     test_loss = evaluate(model, test_loader, loss_fn, device)
    #     print(f"Epoch {epoch}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")