# MNIST from Scratch

Reproduced MNIST handwritten digit classification, ground up. Intended to showcase how to rebuild foundational deep learning architectures and training pipelines from scratch, rather than relying solely on high-level frameworks. 

The plan is to start by training models using PyTorch, and then systematically peel back the layers—replacing modules, the autograd engine, and the optimizer—until everything is running on pure NumPy.

## Roadmap & Current Progress

- **Phase 1: Raw PyTorch (Current)**
  - [x] Basic MLP implemented manually using `nn.Parameter` and raw matrix multiplications instead of `nn.Linear` (`mnist_mlp.py`).
  - [x] Custom training loop that directly parses the raw IDX files rather than using `torchvision.datasets` (`trainer_torch.py`).
  - [x] Custom CNN architectures built from scratch (`mnist_cnn.py`), featuring multiple Conv2D implementations:
    - Brute force nested loops (`_convolve_brute`)
    - `im2col` matrix multiplication variants:
      - Slow, loop-based implementation (`convolve_im2col_slow`)
      - Fast, fully vectorized implementation (`convolve_im2col_fast`)
  - [x] Implement pooling layers (e.g., MaxPool2d, AvgPool2d) for CNNs.
- **Phase 2: Dropping PyTorch Components**
  - [ ] Write a custom auto-grad engine in Python/NumPy to replace `loss.backward()`.
  - [ ] Implement custom optimizers (e.g., AdamW, SGD) to replace `torch.optim`.
- **Phase 3: Pure NumPy Engine**
  - [ ] Fully replace Tensors with `np.ndarray` and strip out `torch` dependencies completely.

## Project Structure

- `data/` - Expects the raw MNIST dataset (`*-idx*-ubyte` files).
- `torch_version/` - The PyTorch-based implementation.
  - `mnist_mlp.py` - Custom MLP and barebones Linear layer implementation.
  - `mnist_cnn.py` - Custom CNN architecture and Conv2D implementations (brute force, loop-based im2col, vectorized im2col).
  - `trainer.py` - The main training loop, evaluation code, and logic to handle the `struct` parsing of MNIST's native IDX file format.
- `numpy_version/` - The pure NumPy implementation (in progress).
  - `mnist_mlp.py` - Custom MLP implementation.
  - `autograd.py` - Custom autograd engine in Python/NumPy.
- `download.py` - Script to download the raw MNIST dataset.

## Usage

1. Download the raw MNIST `.idx` and `.idx3/1-ubyte` binaries into the `data/` directory. Run with python `download.py`
2. Run the PyTorch training loop:
   ```bash
   python torch_version/trainer.py
   ```
