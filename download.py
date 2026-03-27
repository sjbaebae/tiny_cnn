# download mnist dataset into data directory

import requests
import os
from pathlib import Path

# check if data directory exists, if not create it
if not os.path.exists("data"):
    os.makedirs("data")

# check for the following files: t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte, train-images-idx3-ubyte, train-labels-idx1-ubyte
required_files = ["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"]
for file in required_files:
    if not os.path.exists(os.path.join("data", file)):
        # download from https://storage.googleapis.com/cvdf-datasets/mnist
        url = f"https://storage.googleapis.com/cvdf-datasets/mnist/{file}.gz"

        response = requests.get(url)

        if response.ok:
            with open(os.path.join('data', file), "wb") as f:
                f.write(response.content)
            print(f"Downloaded {file}")
        else:
            print(f"Failed to download {file}")