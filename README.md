!pip install torch torchvision torchaudio
!pip install opencv-python pillow numpy matplotlib
!pip install scikit-image lpips tqdm

import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:",
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
          "GB")

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

!pip install -q gdown


import os

os.makedirs("/content/KLA_dataset", exist_ok=True)

print("Dataset folder created.")


import gdown

train_url = "https://drive.google.com/uc?id=1Ayd88_vLwVh-0of3BzL3v94DF7tTutK9"

gdown.download(
    train_url,
    "/content/KLA_dataset/train.zip",
    quiet=False
)

import os
import zipfile

zip_path = "/content/KLA_dataset/train.zip"

print("File exists:", os.path.exists(zip_path))
print("File size:", round(os.path.getsize(zip_path) / (1024**2), 2), "MB")

with zipfile.ZipFile(zip_path, 'r') as z:
    files = z.namelist()

print("Number of files:", len(files))

print("\nFirst 30 files:")
for f in files[:30]:
    print(f)

with zipfile.ZipFile(zip_path, 'r') as z:
    print("ZIP valid:", z.testzip() is None)

import zipfile
import os

zip_path = "/content/KLA_dataset/train.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    npy_files = [
        f for f in z.namelist()
        if f.lower().endswith(".npy")
    ]

print("Number of .npy files:", len(npy_files))

print("\nFirst 10:")
for f in npy_files[:10]:
    print(f)

print("\nLast 10:")
for f in npy_files[-10:]:
    print(f)

import numpy as np
import zipfile

zip_path = "/content/KLA_dataset/train.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    npy_files = [
        f for f in z.namelist()
        if f.startswith("NoisyLR/") and f.endswith(".npy")
    ]

    sample_file = npy_files[0]

    with z.open(sample_file) as f:
        image = np.load(f)

print("File:", sample_file)
print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum:", image.min())
print("Maximum:", image.max())
print("Mean:", image.mean())
print("Standard deviation:", image.std())

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="gray")
plt.title(f"{sample_file}\nShape: {image.shape}")
plt.axis("off")
plt.show()

import os

print("Files in /content:\n")

for file in os.listdir("/content"):
    if file.endswith(".zip"):
        size_mb = os.path.getsize("/content/" + file) / (1024**2)
        print(f"{file}  →  {size_mb:.2f} MB")

from google.colab import files

uploaded = files.upload()

import os

for file in os.listdir("/content"):
    if file.endswith(".zip"):
        size_mb = os.path.getsize("/content/" + file) / (1024**2)
        print(f"{file} → {size_mb:.2f} MB")

from google.colab import files

uploaded = files.upload()

import os

for file in os.listdir("/content"):
    if file.endswith(".zip"):
        size_mb = os.path.getsize("/content/" + file) / (1024**2)
        print(f"{file} → {size_mb:.2f} MB")

import zipfile

zip_path = "/content/train.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    files = z.namelist()

    print("Number of files:", len(files))

    print("\nFirst 50 files:")
    for f in files[:50]:
        print(f)

    print("\nZIP valid:", z.testzip() is None)

with zipfile.ZipFile(zip_path, "r") as z:
    files = z.namelist()

z.testzip()

import zipfile

zip_path = "/content/train.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    files = z.namelist()

gt_files = [
    f for f in files
    if f.startswith("train/GT/") and f.endswith(".npy")
]

noisy_files = [
    f for f in files
    if f.startswith("train/NoisyLR/") and f.endswith(".npy")
]

print("Ground Truth images:", len(gt_files))
print("NoisyLR images:", len(noisy_files))

gt_ids = {
    f.split("/")[-1]
    for f in gt_files
}

noisy_ids = {
    f.split("/")[-1]
    for f in noisy_files
}

print("GT images:", len(gt_ids))
print("NoisyLR images:", len(noisy_ids))

print("GT without NoisyLR:", len(gt_ids - noisy_ids))
print("NoisyLR without GT:", len(noisy_ids - gt_ids))

if gt_ids == noisy_ids:
    print("\n✅ ALL 3200 IMAGE PAIRS MATCH")
else:
    print("\n⚠️ Some pairs do not match")

import zipfile
import numpy as np
import io

zip_path = "/content/train.zip"

with zipfile.ZipFile(zip_path, "r") as z:

    # Pick one training pair
    sample_name = sorted(gt_ids)[0]

    gt_path = f"train/GT/{sample_name}"
    noisy_path = f"train/NoisyLR/{sample_name}"

    gt_data = np.load(
        io.BytesIO(z.read(gt_path))
    )

    noisy_data = np.load(
        io.BytesIO(z.read(noisy_path))
    )

print("Sample:", sample_name)

print("\n========== GROUND TRUTH ==========")
print("Shape:", gt_data.shape)
print("Data type:", gt_data.dtype)
print("Minimum:", gt_data.min())
print("Maximum:", gt_data.max())
print("Mean:", gt_data.mean())
print("Std:", gt_data.std())

print("\n========== NOISY + LOW RESOLUTION ==========")
print("Shape:", noisy_data.shape)
print("Data type:", noisy_data.dtype)
print("Minimum:", noisy_data.min())
print("Maximum:", noisy_data.max())
print("Mean:", noisy_data.mean())
print("Std:", noisy_data.std())

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(noisy_data, cmap="gray")
plt.title(f"NoisyLR - {sample_name}\n{noisy_data.shape}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gt_data, cmap="gray")
plt.title(f"Ground Truth - {sample_name}\n{gt_data.shape}")
plt.axis("off")

plt.show()

import zipfile
import numpy as np
import io

zip_path = "/content/train.zip"

with zipfile.ZipFile(zip_path, "r") as z:

    sample_name = sorted(gt_ids)[0]

    gt_path = f"train/GT/{sample_name}"
    noisy_path = f"train/NoisyLR/{sample_name}"

    gt_data = np.load(io.BytesIO(z.read(gt_path)))
    noisy_data = np.load(io.BytesIO(z.read(noisy_path)))

print("Sample:", sample_name)

print("\n===== GROUND TRUTH =====")
print("Shape:", gt_data.shape)
print("Data type:", gt_data.dtype)
print("Minimum:", gt_data.min())
print("Maximum:", gt_data.max())
print("Mean:", gt_data.mean())
print("Std:", gt_data.std())

print("\n===== NOISY + LOW RESOLUTION =====")
print("Shape:", noisy_data.shape)
print("Data type:", noisy_data.dtype)
print("Minimum:", noisy_data.min())
print("Maximum:", noisy_data.max())
print("Mean:", noisy_data.mean())
print("Std:", noisy_data.std())

import zipfile
import os

zip_path = "/content/train.zip"
extract_path = "/content/KLA_data"

print("Extracting dataset...")

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_path)

print("Extraction complete.")


import os

print("Contents of KLA_data:")
print(os.listdir("/content/KLA_data"))

print("\nContents of train:")
print(os.listdir("/content/KLA_data/train"))

gt_dir = "/content/KLA_data/train/GT"
noisy_dir = "/content/KLA_data/train/NoisyLR"

gt_files = [
    f for f in os.listdir(gt_dir)
    if f.endswith(".npy")
]

noisy_files = [
    f for f in os.listdir(noisy_dir)
    if f.endswith(".npy")
]

print("GT files:", len(gt_files))
print("NoisyLR files:", len(noisy_files))

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class KLADataset(Dataset):

    def __init__(self, root_dir):
        self.gt_dir = os.path.join(root_dir, "GT")
        self.noisy_dir = os.path.join(root_dir, "NoisyLR")

        self.files = sorted([
            f for f in os.listdir(self.gt_dir)
            if f.endswith(".npy")
        ])

        # Verify every GT image has a matching NoisyLR image
        for filename in self.files:
            noisy_path = os.path.join(self.noisy_dir, filename)

            if not os.path.exists(noisy_path):
                raise FileNotFoundError(
                    f"Missing NoisyLR pair: {filename}"
                )

        print("Dataset initialized successfully.")
        print("Total image pairs:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        gt_path = os.path.join(self.gt_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)

        # Load NumPy images
        gt = np.load(gt_path).astype(np.float32)
        noisy = np.load(noisy_path).astype(np.float32)

        # Convert NumPy → PyTorch
        noisy = torch.from_numpy(noisy)
        gt = torch.from_numpy(gt)

        # Add grayscale channel dimension
        noisy = noisy.unsqueeze(0)
        gt = gt.unsqueeze(0)

        return noisy, gt

dataset = KLADataset("/content/KLA_data/train")

print("\nDataset length:", len(dataset))

noisy, gt = dataset[0]

print("\nNoisy:")
print("Shape:", noisy.shape)
print("Dtype:", noisy.dtype)
print("Min:", noisy.min().item())
print("Max:", noisy.max().item())

print("\nGround Truth:")
print("Shape:", gt.shape)
print("Dtype:", gt.dtype)
print("Min:", gt.min().item())
print("Max:", gt.max().item())

from torch.utils.data import random_split

# Reproducible split
generator = torch.Generator().manual_seed(42)

train_size = int(0.90 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("Total:", len(dataset))
print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))

from torch.utils.data import DataLoader

BATCH_SIZE = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

noisy_batch, gt_batch = next(iter(train_loader))

print("Input batch:")
print("Shape:", noisy_batch.shape)
print("Dtype:", noisy_batch.dtype)

print("\nTarget batch:")
print("Shape:", gt_batch.shape)
print("Dtype:", gt_batch.dtype)

print("\nInput range:")
print(noisy_batch.min().item(), "to", noisy_batch.max().item())

print("\nTarget range:")
print(gt_batch.min().item(), "to", gt_batch.max().item())

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class KLARestorationNet(nn.Module):

    def __init__(self, num_features=64, num_blocks=8):
        super().__init__()

        # Initial feature extraction
        self.head = nn.Conv2d(
            1, num_features, 3, padding=1
        )

        # Residual feature processing
        self.body = nn.Sequential(
            *[
                ResidualBlock(num_features)
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            num_features,
            num_features,
            3,
            padding=1
        )

        # 2× learned upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(
                num_features,
                num_features * 4,
                3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        # Final image reconstruction
        self.tail = nn.Conv2d(
            num_features,
            1,
            3,
            padding=1
        )

    def forward(self, x):

        # Initial features
        x0 = self.head(x)

        # Residual processing
        features = self.body(x0)
        features = self.body_conv(features)

        # Global residual connection
        features = features + x0

        # 2× upsampling
        features = self.upsample(features)

        # Output image
        output = self.tail(features)

        # GT is in [0,1]
        output = torch.sigmoid(output)

        return output

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = KLARestorationNet().to(device)

print("Device:", device)
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

noisy_batch, gt_batch = next(iter(train_loader))

noisy_batch = noisy_batch.to(device)
gt_batch = gt_batch.to(device)

with torch.no_grad():
    output = model(noisy_batch)

print("Input :", noisy_batch.shape)
print("Output:", output.shape)
print("GT    :", gt_batch.shape)

print("\nOutput range:")
print(
    output.min().item(),
    "to",
    output.max().item()
)

import torch
import torch.nn as nn

l1_loss = nn.L1Loss()

def charbonnier_loss(pred, target, epsilon=1e-3):
    diff = pred - target
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    return loss.mean()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4
)

print("Optimizer:", optimizer)

import torch
import torch.nn as nn

def charbonnier_loss(pred, target, epsilon=1e-3):
    diff = pred - target
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    return loss.mean()

print("Charbonnier loss ready.")

def train_one_epoch(model, loader, optimizer, device):

    model.train()
    total_loss = 0.0

    for batch_idx, (noisy, target) in enumerate(loader):

        noisy = noisy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(noisy)

        loss = charbonnier_loss(output, target)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(
                f"Batch {batch_idx+1}/{len(loader)} "
                f"Loss: {loss.item():.6f}"
            )

    return total_loss / len(loader)

def validate(model, loader, device):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for noisy, target in loader:

            noisy = noisy.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(noisy)

            loss = charbonnier_loss(output, target)

            total_loss += loss.item()

    return total_loss / len(loader)

import torch
import torch.nn as nn

def charbonnier_loss(pred, target, epsilon=1e-3):
    diff = pred - target
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    return loss.mean()

def train_one_epoch(model, loader, optimizer, device):

    model.train()
    total_loss = 0.0

    for batch_idx, (noisy, target) in enumerate(loader):

        noisy = noisy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(noisy)

        loss = charbonnier_loss(output, target)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(
                f"Batch {batch_idx+1}/{len(loader)} "
                f"Loss: {loss.item():.6f}"
            )

    return total_loss / len(loader)

def validate(model, loader, device):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for noisy, target in loader:

            noisy = noisy.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(noisy)

            loss = charbonnier_loss(output, target)

            total_loss += loss.item()

    return total_loss / len(loader)


EPOCHS = 5

train_losses = []
val_losses = []

for epoch in range(EPOCHS):

    print("\n" + "=" * 50)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("=" * 50)

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device
    )

    val_loss = validate(
        model,
        val_loader,
        device
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}: "
        f"Train Loss = {train_loss:.6f}, "
        f"Validation Loss = {val_loss:.6f}"
    )


print("dataset:", "dataset" in globals())
print("train_loader:", "train_loader" in globals())
print("val_loader:", "val_loader" in globals())
print("device:", "device" in globals())

import os

print("Files in /content:")
for item in os.listdir("/content"):
    print(item)

!gdown 1Ayd88_vLwVh-0of3BzL3v94DF7tTutK9 -O /content/train.zip


import os

print("File exists:", os.path.exists("/content/train.zip"))

if os.path.exists("/content/train.zip"):
    print("File size:",
          round(os.path.getsize("/content/train.zip") / (1024**3), 2),
          "GB")

import os

size_mb = os.path.getsize("/content/train.zip") / (1024 * 1024)

print(f"train.zip size: {size_mb:.2f} MB")

from google.colab import files

uploaded = files.upload()

import os

print(os.listdir("/content"))

size_mb = os.path.getsize("/content/train.zip") / (1024 * 1024)

print(f"train.zip size: {size_mb:.2f} MB")


import os

for f in os.listdir("/content"):
    if f.lower().endswith(".zip"):
        size_mb = os.path.getsize("/content/" + f) / (1024 * 1024)
        print(f"{f} → {size_mb:.2f} MB")

import zipfile
import os

zip_path = "/content/train.zip"
extract_path = "/content/KLA_data"

print("Extracting 876 MB training dataset...")
print("Please wait...")

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_path)

print("✅ Extraction complete!")


import os

gt_dir = "/content/KLA_data/train/GT"
noisy_dir = "/content/KLA_data/train/NoisyLR"

gt_files = sorted([
    f for f in os.listdir(gt_dir)
    if f.endswith(".npy")
])

noisy_files = sorted([
    f for f in os.listdir(noisy_dir)
    if f.endswith(".npy")
])

print("Ground Truth images:", len(gt_files))
print("NoisyLR images:", len(noisy_files))

print("\nMissing GT pairs:",
      len(set(noisy_files) - set(gt_files)))

print("Missing NoisyLR pairs:",
      len(set(gt_files) - set(noisy_files)))

import os

print(os.listdir("/content"))

import os

print(os.listdir("/content/KLA_data"))

print(os.listdir("/content/KLA_data/train"))

import os

print("Folders/files in /content:")
print(os.listdir("/content"))


import os

for root, dirs, files in os.walk("/content"):
    if "KLA" in root or "kla" in root:
        print(root)

import os

print(os.listdir("/content/drive"))


import os

print(os.listdir("/content"))

import os

print(os.listdir("/content/KLA_data"))


import zipfile

with zipfile.ZipFile("/content/train.zip", "r") as zip_ref:
    print("First 30 files:")
    for name in zip_ref.namelist()[:30]:
        print(name)

import zipfile

with zipfile.ZipFile("/content/train.zip", "r") as zip_ref:
    folders = set()

    for name in zip_ref.namelist():
        parts = name.split("/")
        if len(parts) > 1:
            folders.add(parts[0])

    print("Folders in train.zip:")
    for folder in sorted(folders):
        print(folder)


import os

for root, dirs, files in os.walk("/content/KLA_data"):
    npy_files = [f for f in files if f.endswith(".npy")]

    if npy_files:
        print("\nFolder:", root)
        print("Number of NPY files:", len(npy_files))
        print("Examples:", npy_files[:5])

import os

for root, dirs, files in os.walk("/content/KLA_data"):
    level = root.replace("/content/KLA_data", "").count(os.sep)

    if level <= 2:
        print("  " * level + os.path.basename(root) + "/")

import numpy as np
import os

data_dir = "/content/KLA_data/NoisyLR"

files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

print("Number of NPY files:", len(files))

# Load first image
sample_file = os.path.join(data_dir, files[0])
sample = np.load(sample_file)

print("Sample file:", files[0])
print("Shape:", sample.shape)
print("Data type:", sample.dtype)
print("Minimum:", sample.min())
print("Maximum:", sample.max())
print("Mean:", sample.mean())

import numpy as np
import os

for filename in files[:5]:
    path = os.path.join(data_dir, filename)
    img = np.load(path)

    print(
        filename,
        "shape =", img.shape,
        "dtype =", img.dtype,
        "min =", img.min(),
        "max =", img.max()
    )

import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = "/content/KLA_data/NoisyLR"

files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

# Select one image
file = files[0]
image = np.load(os.path.join(data_dir, file))

plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="gray")
plt.colorbar()
plt.title(f"{file} - {image.shape}")
plt.axis("off")
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax, file in zip(axes.flat, files[:8]):
    image = np.load(os.path.join(data_dir, file))

    ax.imshow(image, cmap="gray")
    ax.set_title(file)
    ax.axis("off")

plt.tight_layout()
plt.show()

import os

print("Files in /content:")
for item in os.listdir("/content"):
    print(item)

import os

for item in os.listdir("/content"):
    path = os.path.join("/content", item)

    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{item:30} {size_mb:.2f} MB")

from google.colab import drive

drive.mount('/content/drive')

import os

print("My Drive:")
for item in os.listdir("/content/drive/MyDrive"):
    print(item)

from google.colab import drive

drive.mount('/content/drive')

import os

print("Contents of /content/drive:")
print(os.listdir("/content/drive"))

import os

for root, dirs, files in os.walk("/content/drive"):
    print(root)
    if root.count("/") > 5:
        dirs[:] = []

import os

for root, dirs, files in os.walk("/content/drive"):
    for file in files:
        if file.lower().endswith((".zip", ".npy", ".tif", ".tiff", ".png", ".jpg", ".mat")):
            print(os.path.join(root, file))

import os

print(os.listdir("/content/drive/MyDrive"))


import os

extensions = (".zip", ".npy", ".mat", ".tif", ".tiff", ".png", ".jpg", ".jpeg")

for root, dirs, files in os.walk("/content/drive/MyDrive"):
    for file in files:
        if file.lower().endswith(extensions):
            print(os.path.join(root, file))


import os

extensions = (".zip", ".npy", ".mat", ".tif", ".tiff", ".png", ".jpg", ".jpeg")

for root, dirs, files in os.walk("/content/drive/MyDrive"):
    for file in files:
        if file.lower().endswith(extensions):
            print(os.path.join(root, file))


import os

extensions = (".zip", ".npy", ".mat", ".tif", ".tiff", ".png", ".jpg", ".jpeg")

for root, dirs, files in os.walk("/content/drive/MyDrive"):
    for file in files:
        if file.lower().endswith(extensions):
            print(os.path.join(root, file))

import os

extensions = (".zip", ".npy", ".mat", ".tif", ".tiff", ".png", ".jpg", ".jpeg")

for root, dirs, files in os.walk("/content/drive/MyDrive"):
    for file in files:
        if file.lower().endswith(extensions):
            print(os.path.join(root, file))

import os

extensions = (".zip", ".npy", ".mat", ".tif", ".tiff", ".png", ".jpg", ".jpeg")

count = 0

for root, dirs, files in os.walk("/content/drive/MyDrive"):
    for file in files:
        if file.lower().endswith(extensions):
            count += 1
            print(f"{count}. {os.path.join(root, file)}")

import os

print(os.listdir("/content/drive/MyDrive"))

from google.colab import drive

drive.mount('/content/drive', force_remount=True)

import os

print(os.path.exists("/content/drive"))
print(os.listdir("/content/drive"))

print(os.listdir("/content/drive/MyDrive"))

import os

for root, dirs, files in os.walk("/content/drive/MyDrive"):
    for file in files:
        if file.lower().endswith(
            (".zip", ".npy", ".mat", ".tif", ".tiff", ".png", ".jpg", ".jpeg")
        ):
            print(os.path.join(root, file))

import zipfile

zip_files = [
    "/content/drive/MyDrive/train.zip",
    "/content/drive/MyDrive/Test_NoisyLR.zip"
]

for zip_path in zip_files:
    print("\n" + "=" * 60)
    print("FILE:", zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()

        print("Number of files:", len(names))
        print("\nFirst 20 files:")

        for name in names[:20]:
            print(name)

import zipfile

for zip_path in zip_files:
    print("\n" + "=" * 60)
    print("FILE:", zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        folders = set()

        for name in z.namelist():
            parts = name.split("/")

            if len(parts) > 1:
                folders.add(parts[0])

        print("Folders:")
        for folder in sorted(folders):
            print("  ", folder)

import zipfile

train_zip = "/content/drive/MyDrive/train.zip"

with zipfile.ZipFile(train_zip, "r") as z:
    print("Files/folders inside train.zip:\n")

    for name in z.namelist()[:100]:
        print(name)

import zipfile

train_zip = "/content/drive/MyDrive/train.zip"

with zipfile.ZipFile(train_zip, "r") as z:
    folders = set()

    for name in z.namelist():
        if name.startswith("train/"):
            parts = name.split("/")

            if len(parts) >= 2:
                folders.add(parts[1])

    print("Contents of train/:")
    for folder in sorted(folders):
        print(folder)

import zipfile
import os

train_zip = "/content/drive/MyDrive/train.zip"
extract_dir = "/content/KLA_train"

with zipfile.ZipFile(train_zip, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

print("Training dataset extracted!")
print(os.listdir(extract_dir))

import os

train_dir = "/content/KLA_train/train"

print("Training folders:")
print(os.listdir(train_dir))

print("\nGT files:")
gt_dir = os.path.join(train_dir, "GT")
print(len(os.listdir(gt_dir)))
print(os.listdir(gt_dir)[:10])

print("\nNoisyLR files:")
noisy_dir = os.path.join(train_dir, "NoisyLR")
print(len(os.listdir(noisy_dir)))
print(os.listdir(noisy_dir)[:10])

import os

gt_files = sorted([
    f for f in os.listdir(gt_dir)
    if f.endswith(".npy")
])

noisy_files = sorted([
    f for f in os.listdir(noisy_dir)
    if f.endswith(".npy")
])

print("GT images:", len(gt_files))
print("NoisyLR images:", len(noisy_files))

print("\nFirst 10 GT:")
print(gt_files[:10])

print("\nFirst 10 NoisyLR:")
print(noisy_files[:10])

print("\nMatching filenames:", gt_files == noisy_files)

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Paths
gt_dir = "/content/KLA_train/train/GT"
noisy_dir = "/content/KLA_train/train/NoisyLR"


class SemiconductorDataset(Dataset):

    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        self.files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        noisy_path = os.path.join(self.noisy_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        noisy = np.load(noisy_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        # Add channel dimension
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt


dataset = SemiconductorDataset(
    noisy_dir,
    gt_dir
)

print("Total images:", len(dataset))

noisy, gt = dataset[0]

print("Noisy shape:", noisy.shape)
print("GT shape:", gt.shape)
print("Noisy dtype:", noisy.dtype)
print("GT dtype:", gt.dtype)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

noisy_batch, gt_batch = next(iter(train_loader))

print("Noisy batch:", noisy_batch.shape)
print("GT batch:", gt_batch.shape)

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: Not detected")

print("Noisy shape:", noisy_batch.shape)
print("GT shape:", gt_batch.shape)

print("Noisy min:", noisy_batch.min().item())
print("Noisy max:", noisy_batch.max().item())

print("GT min:", gt_batch.min().item())
print("GT max:", gt_batch.max().item())

import matplotlib.pyplot as plt

noisy_img = noisy_batch[0, 0].numpy()
gt_img = gt_batch[0, 0].numpy()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(noisy_img, cmap="gray")
plt.title("NoisyLR - 128×128")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gt_img, cmap="gray")
plt.title("GT - 256×256")
plt.axis("off")

plt.show()

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class RestorationUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 2× upsampling: 128×128 → 256×256
        self.up_final = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )

        self.output = nn.Conv2d(
            32, 1, 3, padding=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # 2× super-resolution
        x = self.up_final(d1)

        x = self.output(x)

        return x

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cpu")

model = RestorationUNet().to(device)

test_input = torch.randn(1, 1, 128, 128).to(device)

with torch.no_grad():
    output = model(test_input)

print("Input :", test_input.shape)
print("Output:", output.shape)

from torch.utils.data import Subset, DataLoader

# Use only 100 images for the CPU test
small_train_dataset = Subset(train_dataset, range(100))

small_train_loader = DataLoader(
    small_train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

print("CPU test images:", len(small_train_dataset))
print("CPU test batches:", len(small_train_loader))

import torch.nn as nn

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

print("Loss:", criterion)
print("Optimizer:", optimizer.__class__.__name__)

model.train()

running_loss = 0.0

for batch_idx, (noisy, gt) in enumerate(small_train_loader):

    noisy = noisy.to(device)
    gt = gt.to(device)

    # Clear previous gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(noisy)

    # Calculate loss
    loss = criterion(output, gt)

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    running_loss += loss.item()

    if (batch_idx + 1) % 10 == 0:
        print(
            f"Batch [{batch_idx + 1}/{len(small_train_loader)}] "
            f"Loss: {loss.item():.6f}"
        )

average_loss = running_loss / len(small_train_loader)

print("\nCPU test completed!")
print("Average loss:", average_loss)

import torch

model.eval()

val_loss = 0.0

with torch.no_grad():
    for noisy, gt in val_loader:
        noisy = noisy.to(device)
        gt = gt.to(device)

        output = model(noisy)

        loss = criterion(output, gt)
        val_loss += loss.item()

val_loss /= len(val_loader)

print("Validation Loss:", val_loss)

import math
import torch

def calculate_psnr(pred, target, data_range):
    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return float("inf")

    return 20 * math.log10(data_range / math.sqrt(mse.item()))

gt_min = float("inf")
gt_max = float("-inf")

for _, gt in val_loader:
    gt_min = min(gt_min, gt.min().item())
    gt_max = max(gt_max, gt.max().item())

print("GT minimum:", gt_min)
print("GT maximum:", gt_max)
print("GT data range:", gt_max - gt_min)

import matplotlib.pyplot as plt

model.eval()

with torch.no_grad():
    noisy, gt = next(iter(val_loader))

    noisy = noisy.to(device)
    output = model(noisy)

noisy_img = noisy[0, 0].cpu().numpy()
output_img = output[0, 0].cpu().numpy()
gt_img = gt[0, 0].cpu().numpy()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(noisy_img, cmap="gray")
plt.title("NoisyLR 128×128")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(output_img, cmap="gray")
plt.title("Restored 256×256")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gt_img, cmap="gray")
plt.title("GT 256×256")
plt.axis("off")

plt.tight_layout()
plt.show()

!pip -q install scikit-image

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

model.eval()

total_loss = 0.0
psnr_scores = []
ssim_scores = []

with torch.no_grad():
    for noisy, gt in val_loader:
        noisy = noisy.to(device)
        gt = gt.to(device)

        output = model(noisy)

        # Validation loss
        loss = criterion(output, gt)
        total_loss += loss.item()

        # Convert batch to CPU numpy
        pred_np = output.cpu().numpy()
        gt_np = gt.cpu().numpy()

        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i, 0]
            gt_img = gt_np[i, 0]

            # Use GT dynamic range for PSNR
            data_range = float(gt_img.max() - gt_img.min())

            if data_range == 0:
                data_range = 1.0

            psnr = peak_signal_noise_ratio(
                gt_img,
                pred_img,
                data_range=data_range
            )

            ssim = structural_similarity(
                gt_img,
                pred_img,
                data_range=data_range
            )

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

avg_val_loss = total_loss / len(val_loader)
avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores)

print(f"Validation Loss : {avg_val_loss:.6f}")
print(f"Average PSNR    : {avg_psnr:.4f} dB")
print(f"Average SSIM    : {avg_ssim:.4f}")

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_loss": avg_val_loss,
    "psnr": avg_psnr,
    "ssim": avg_ssim
}

torch.save(
    checkpoint,
    "/content/semiconductor_restoration_checkpoint.pth"
)

print("Checkpoint saved.")

import matplotlib.pyplot as plt

model.eval()

with torch.no_grad():
    noisy, gt = next(iter(val_loader))
    output = model(noisy.to(device))

noisy_img = noisy[0, 0].numpy()
pred_img = output[0, 0].cpu().numpy()
gt_img = gt[0, 0].numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(noisy_img, cmap="gray")
axes[0].set_title("NoisyLR 128×128")
axes[0].axis("off")

axes[1].imshow(pred_img, cmap="gray")
axes[1].set_title("Restored 256×256")
axes[1].axis("off")

axes[2].imshow(gt_img, cmap="gray")
axes[2].set_title("GT 256×256")
axes[2].axis("off")

plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import numpy as np
import os

criterion = nn.L1Loss()

def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0

    for noisy, gt in loader:
        noisy = noisy.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        output = model(noisy)

        loss = criterion(output, gt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for noisy, gt in loader:
            noisy = noisy.to(device)
            gt = gt.to(device)

            output = model(noisy)

            loss = criterion(output, gt)

            total_loss += loss.item()

    return total_loss / len(loader)

device = torch.device("cpu")

model = RestorationUNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

num_epochs = 3

best_val_loss = float("inf")

for epoch in range(num_epochs):

    train_loss = train_one_epoch(
        model,
        small_train_loader,
        optimizer,
        device
    )

    val_loss = validate(
        model,
        val_loader,
        device
    )

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.6f} | "
        f"Validation Loss: {val_loss:.6f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            },
            "/content/best_semiconductor_model.pth"
        )

        print("✅ Best model saved!")

checkpoint = torch.load(
    "/content/best_semiconductor_model.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

print("Best model loaded.")
print("Best validation loss:", checkpoint["val_loss"])
print("Epoch:", checkpoint["epoch"])

print("Epoch:", checkpoint["epoch"])+


print("Epoch:", checkpoint["epoch"])


import matplotlib.pyplot as plt

model.eval()

noisy, gt = val_dataset[0]

with torch.no_grad():
    input_tensor = noisy.unsqueeze(0).to(device)
    prediction = model(input_tensor)

prediction = prediction.squeeze().cpu().numpy()
noisy_img = noisy.squeeze().numpy()
gt_img = gt.squeeze().numpy()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(noisy_img, cmap="gray")
plt.title("NoisyLR - 128×128")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(prediction, cmap="gray")
plt.title("AI Restored - 256×256")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gt_img, cmap="gray")
plt.title("Ground Truth - 256×256")
plt.axis("off")

plt.tight_layout()
plt.show()

import zipfile
import os

test_zip = "/content/drive/MyDrive/Test_NoisyLR.zip"
test_extract = "/content/KLA_test"

with zipfile.ZipFile(test_zip, "r") as z:
    z.extractall(test_extract)

print(os.listdir(test_extract))

test_noisy_dir = "/content/KLA_test/NoisyLR"

test_files = sorted([
    f for f in os.listdir(test_noisy_dir)
    if f.endswith(".npy")
])

print("Number of test images:", len(test_files))
print("First 10:", test_files[:10])

output_dir = "/content/KLA_restored"

os.makedirs(output_dir, exist_ok=True)

print("Output directory:", output_dir)

model.eval()

with torch.no_grad():

    for i, filename in enumerate(test_files):

        path = os.path.join(
            test_noisy_dir,
            filename
        )

        noisy = np.load(path).astype(np.float32)

        tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(device)

        prediction = model(tensor)

        prediction = prediction.squeeze().cpu().numpy()

        save_path = os.path.join(
            output_dir,
            filename
        )

        np.save(save_path, prediction)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(test_files)}")

import os
import numpy as np
import matplotlib.pyplot as plt

output_dir = "/content/KLA_restored"
test_dir = "/content/KLA_test/NoisyLR"

files = sorted([
    f for f in os.listdir(test_dir)
    if f.endswith(".npy")
])

# Select one test image
filename = files[0]

noisy = np.load(os.path.join(test_dir, filename))
restored = np.load(os.path.join(output_dir, filename))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(noisy, cmap="gray")
plt.title(f"NoisyLR - {filename}\n{noisy.shape}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(restored, cmap="gray")
plt.title(f"Restored - {filename}\n{restored.shape}")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Noisy shape:", noisy.shape)
print("Restored shape:", restored.shape)
print("Restored min:", restored.min())
print("Restored max:", restored.max())

import os
import numpy as np
import matplotlib.pyplot as plt

output_dir = "/content/KLA_restored"
test_dir = "/content/KLA_test/NoisyLR"

files = sorted([f for f in os.listdir(test_dir) if f.endswith(".npy")])

filename = files[0]

noisy = np.load(os.path.join(test_dir, filename))
restored = np.load(os.path.join(output_dir, filename))

print("Filename:", filename)
print("Noisy shape:", noisy.shape)
print("Restored shape:", restored.shape)
print("Noisy range:", noisy.min(), "to", noisy.max())
print("Restored range:", restored.min(), "to", restored.max())

from skimage.transform import resize

noisy_display = resize(
    noisy,
    restored.shape,
    preserve_range=True,
    anti_aliasing=False
)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(noisy_display, cmap="gray")
plt.title("NoisyLR (displayed at 256×256)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(restored, cmap="gray")
plt.title("AI Restored")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(restored - noisy_display, cmap="gray")
plt.title("Restoration Difference")
plt.axis("off")

plt.tight_layout()
plt.show()

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import torch

model.eval()

psnr_values = []
ssim_values = []

with torch.no_grad():
    for noisy, gt in val_loader:

        noisy = noisy.to(device)
        prediction = model(noisy)

        prediction = prediction.cpu().numpy()
        gt = gt.numpy()

        for i in range(prediction.shape[0]):

            pred_img = prediction[i, 0]
            gt_img = gt[i, 0]

            data_range = float(gt_img.max() - gt_img.min())

            if data_range <= 0:
                data_range = 1.0

            psnr = peak_signal_noise_ratio(
                gt_img,
                pred_img,
                data_range=data_range
            )

            ssim = structural_similarity(
                gt_img,
                pred_img,
                data_range=data_range
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim)

print(f"Mean PSNR: {np.mean(psnr_values):.4f} dB")
print(f"Mean SSIM: {np.mean(ssim_values):.4f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_loss": float(np.mean(psnr_values)),
    "mean_psnr": float(np.mean(psnr_values)),
    "mean_ssim": float(np.mean(ssim_values))
}, "/content/semiconductor_restoration_final_test.pth")

print("Checkpoint saved successfully.")

val_loss = validate(model, val_loader, device)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_loss": float(val_loss),
    "mean_psnr": float(np.mean(psnr_values)),
    "mean_ssim": float(np.mean(ssim_values))
}, "/content/semiconductor_restoration_final_test.pth")

print("Checkpoint saved successfully.")

import os
import shutil

project_dir = "/content/semiconductor_project"
os.makedirs(project_dir, exist_ok=True)

shutil.copy(
    "/content/best_semiconductor_model.pth",
    project_dir
)

shutil.copy(
    "/content/semiconductor_restoration_final_test.pth",
    project_dir
)

print("Project checkpoints saved in:")
print(project_dir)

from google.colab import files

files.download("/content/semiconductor_restoration_final_test.pth")

print("Baseline model:")
print("Training images: 100 CPU-test subset")
print("Validation images: 640")
print("Input: 128x128")
print("Output: 256x256")

import torch
from torch.utils.data import DataLoader, random_split

# Full dataset
full_dataset = SemiconductorDataset(
    noisy_dir=gt_dir.replace("/GT", "/NoisyLR"),
    gt_dir=gt_dir
)

# 80/20 split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Total:", len(full_dataset))
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

import numpy as np

for name, idx in [("sample 1", 0), ("sample 2", 100), ("sample 3", 500)]:
    noisy, gt = full_dataset[idx]

    print(name)
    print("  Noisy:", noisy.min().item(), "to", noisy.max().item())
    print("  GT   :", gt.min().item(), "to", gt.max().item())

import torch
from torch.utils.data import DataLoader, random_split

full_dataset = SemiconductorDataset(
    noisy_dir="/content/KLA_train/train/NoisyLR",
    gt_dir="/content/KLA_train/train/GT"
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Total:", len(full_dataset))
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

self.output = nn.Conv2d(32, 1, 3, padding=1)

self.output = nn.Sequential(
    nn.Conv2d(32, 1, 3, padding=1),
    nn.Sigmoid()
)

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class RestorationUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final 2× upsampling: 128×128 → 256×256
        self.up_final = nn.ConvTranspose2d(64, 32, 2, stride=2)

        # Output constrained to GT range [0, 1]
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # 2× super-resolution
        x = self.up_final(d1)

        # Final restored image
        x = self.output(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RestorationUNet().to(device)

print("Using device:", device)

test_input = torch.randn(1, 1, 128, 128).to(device)

with torch.no_grad():
    test_output = model(test_input)

print("Input :", test_input.shape)
print("Output:", test_output.shape)
print("Output min:", test_output.min().item())
print("Output max:", test_output.max().item())

import torch

noisy_batch, gt_batch = next(iter(train_loader))

noisy_batch = noisy_batch.to(device)
gt_batch = gt_batch.to(device)

with torch.no_grad():
    prediction = model(noisy_batch)

print("Noisy:", noisy_batch.shape)
print("GT:", gt_batch.shape)
print("Prediction:", prediction.shape)

print("GT range:",
      gt_batch.min().item(),
      "to",
      gt_batch.max().item())

print("Prediction range:",
      prediction.min().item(),
      "to",
      prediction.max().item())

criterion = torch.nn.L1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

model.train()

noisy_batch, gt_batch = next(iter(train_loader))

noisy_batch = noisy_batch.to(device)
gt_batch = gt_batch.to(device)

optimizer.zero_grad()

prediction = model(noisy_batch)

loss = criterion(prediction, gt_batch)

loss.backward()
optimizer.step()

print("One training step completed.")
print("L1 Loss:", loss.item())

# Small CPU experiment
small_train_dataset = torch.utils.data.Subset(
    full_dataset,
    list(range(100))
)

small_train_loader = DataLoader(
    small_train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

criterion = torch.nn.L1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

num_epochs = 3

for epoch in range(num_epochs):

    model.train()
    total_loss = 0.0

    for noisy, gt in small_train_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        prediction = model(noisy)

        loss = criterion(prediction, gt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(small_train_loader)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Training Loss: {avg_loss:.6f}"
    )

import torch

model.eval()

val_loss = 0.0

with torch.no_grad():
    for noisy, gt in val_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        prediction = model(noisy)

        loss = criterion(prediction, gt)

        val_loss += loss.item()

val_loss /= len(val_loader)

print("Validation L1 Loss:", val_loss)

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

psnr_values = []
ssim_values = []

model.eval()

with torch.no_grad():
    for noisy, gt in val_loader:

        noisy = noisy.to(device)

        prediction = model(noisy)

        prediction = prediction.cpu().numpy()
        gt = gt.numpy()

        for i in range(prediction.shape[0]):

            pred_img = prediction[i, 0]
            gt_img = gt[i, 0]

            data_range = gt_img.max() - gt_img.min()

            if data_range <= 0:
                data_range = 1.0

            psnr = peak_signal_noise_ratio(
                gt_img,
                pred_img,
                data_range=data_range
            )

            ssim = structural_similarity(
                gt_img,
                pred_img,
                data_range=data_range
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim)

print("Mean PSNR:", np.mean(psnr_values), "dB")
print("Mean SSIM:", np.mean(ssim_values))

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": val_loss,
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values))
    },
    "/content/improved_baseline_model.pth"
)

print("Improved baseline saved successfully.")


import matplotlib.pyplot as plt
import torch

model.eval()

noisy, gt = val_dataset[0]

with torch.no_grad():
    prediction = model(
        noisy.unsqueeze(0).to(device)
    )

prediction = prediction.squeeze().cpu().numpy()
noisy_img = noisy.squeeze().numpy()
gt_img = gt.squeeze().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(noisy_img, cmap="gray")
axes[0].set_title("NoisyLR (128×128)")
axes[0].axis("off")

axes[1].imshow(prediction, cmap="gray")
axes[1].set_title("Improved Model (256×256)")
axes[1].axis("off")

axes[2].imshow(gt_img, cmap="gray")
axes[2].set_title("Ground Truth (256×256)")
axes[2].axis("off")

plt.tight_layout()
plt.show()

print("Validation L1 Loss:", val_loss)
print("Mean PSNR:", np.mean(psnr_values), "dB")
print("Mean SSIM:", np.mean(ssim_values))

import torch
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Fresh model for the real experiment
model = RestorationUNet().to(device)

# Loss
criterion = torch.nn.L1Loss()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

num_epochs = 30
best_val_loss = float("inf")

os.makedirs("/content/checkpoints", exist_ok=True)

for epoch in range(num_epochs):

    # -------------------------
    # Training
    # -------------------------
    model.train()
    train_loss = 0.0

    for noisy, gt in train_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        prediction = model(noisy)

        loss = criterion(prediction, gt)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for noisy, gt in val_loader:

            noisy = noisy.to(device)
            gt = gt.to(device)

            prediction = model(noisy)

            loss = criterion(prediction, gt)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"LR: {current_lr:.2e}"
    )

    # -------------------------
    # Save best model
    # -------------------------
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            "/content/checkpoints/best_model.pth"
        )

        print("✅ Best model saved.")

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class RestorationUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 2× upsampling: 128×128 → 256×256
        self.up_final = nn.ConvTranspose2d(64, 32, 2, stride=2)

        # GT is 0–1
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        x = self.up_final(d1)
        x = self.output(x)

        return x


print("RestorationUNet is defined.")

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SemiconductorDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        self.files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        noisy = np.load(
            os.path.join(self.noisy_dir, filename)
        ).astype(np.float32)

        gt = np.load(
            os.path.join(self.gt_dir, filename)
        ).astype(np.float32)

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt


noisy_dir = "/content/KLA_train/train/NoisyLR"
gt_dir = "/content/KLA_train/train/GT"

full_dataset = SemiconductorDataset(noisy_dir, gt_dir)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

print("Total:", len(full_dataset))
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))

import zipfile
import os

train_zip = "/content/drive/MyDrive/train.zip"
extract_dir = "/content/KLA_train"

with zipfile.ZipFile(train_zip, "r") as z:
    z.extractall(extract_dir)

print("Extracted successfully.")
print(os.listdir(extract_dir))

from google.colab import drive

drive.mount("/content/drive", force_remount=True)

import os

print(os.listdir("/content/drive/MyDrive"))

print(
    os.path.exists("/content/drive/MyDrive/train.zip")
)

import zipfile
import os

train_zip = "/content/drive/MyDrive/train.zip"
extract_dir = "/content/KLA_train"

with zipfile.ZipFile(train_zip, "r") as z:
    z.extractall(extract_dir)

print("Extraction complete.")
print(os.listdir(extract_dir))

import os

train_dir = "/content/KLA_train/train"
gt_dir = os.path.join(train_dir, "GT")
noisy_dir = os.path.join(train_dir, "NoisyLR")

gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".npy")])
noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith(".npy")])

print("GT images:", len(gt_files))
print("NoisyLR images:", len(noisy_files))
print("Filenames match:", gt_files == noisy_files)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SemiconductorDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        self.files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        noisy = np.load(
            os.path.join(self.noisy_dir, filename)
        ).astype(np.float32)

        gt = np.load(
            os.path.join(self.gt_dir, filename)
        ).astype(np.float32)

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt


full_dataset = SemiconductorDataset(
    noisy_dir,
    gt_dir
)

train_size = 2560
val_size = 640

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

print("Total:", len(full_dataset))
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model = RestorationUNet().to(device)

noisy_batch, gt_batch = next(iter(train_loader))

print("Noisy:", noisy_batch.shape)
print("GT:", gt_batch.shape)

with torch.no_grad():
    prediction = model(noisy_batch.to(device))

print("Prediction:", prediction.shape)

import os
import torch
import torch.nn as nn

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Model
# --------------------------------------------------
model = RestorationUNet().to(device)


# --------------------------------------------------
# Loss
# --------------------------------------------------
criterion = nn.L1Loss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)


# --------------------------------------------------
# Learning-rate scheduler
# --------------------------------------------------
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# --------------------------------------------------
# Training settings
# --------------------------------------------------
num_epochs = 30

best_val_loss = float("inf")

checkpoint_dir = "/content/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

print("Training configuration ready.")
print("Epochs:", num_epochs)
print("Batch size:", train_loader.batch_size)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

import os
import shutil

save_dir = "/content/drive/MyDrive/semiconductor_restoration"
os.makedirs(save_dir, exist_ok=True)

print("Project folder created:")
print(save_dir)

import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import os
import shutil

project_dir = "/content/drive/MyDrive/semiconductor_restoration"
os.makedirs(project_dir, exist_ok=True)

files_to_save = [
    "/content/improved_baseline_model.pth",
    "/content/semiconductor_restoration_final_test.pth"
]

for file_path in files_to_save:
    if os.path.exists(file_path):
        shutil.copy2(file_path, project_dir)
        print("Saved:", os.path.basename(file_path))

print("\nProject folder:")
print(project_dir)

import shutil

shutil.make_archive(
    "/content/drive/MyDrive/semiconductor_restoration/KLA_restored",
    "zip",
    "/content/KLA_restored"
)

print("Restored test images saved to Google Drive.")

import os

print("KLA_restored exists:",
      os.path.exists("/content/KLA_restored"))

if os.path.exists("/content/KLA_restored"):
    print("Files:",
          len(os.listdir("/content/KLA_restored")))

import zipfile
import os

test_zip = "/content/drive/MyDrive/Test_NoisyLR.zip"
test_extract = "/content/KLA_test"

with zipfile.ZipFile(test_zip, "r") as z:
    z.extractall(test_extract)

print(os.listdir(test_extract))

test_noisy_dir = "/content/KLA_test/NoisyLR"
test_files = sorted(
    f for f in os.listdir(test_noisy_dir)
    if f.endswith(".npy")
)

print("Test images:", len(test_files))

output_dir = "/content/KLA_restored"
os.makedirs(output_dir, exist_ok=True)

model.eval()

with torch.no_grad():

    for i, filename in enumerate(test_files):

        path = os.path.join(test_noisy_dir, filename)

        noisy = np.load(path).astype(np.float32)

        tensor = torch.from_numpy(noisy)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(device)

        prediction = model(tensor)

        prediction = prediction.squeeze().cpu().numpy()

        np.save(
            os.path.join(output_dir, filename),
            prediction
        )

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(test_files)}")

import shutil
import os

project_dir = "/content/drive/MyDrive/semiconductor_restoration"
os.makedirs(project_dir, exist_ok=True)

shutil.make_archive(
    os.path.join(project_dir, "KLA_restored"),
    "zip",
    output_dir
)

print("Restored test dataset saved successfully.")

!pip -q install pytorch-msssim

import torch
import torch.nn as nn
from pytorch_msssim import ssim


class RestorationLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):
        l1_loss = self.l1(prediction, target)

        ssim_value = ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True
        )

        ssim_loss = 1.0 - ssim_value

        total_loss = (
            self.alpha * l1_loss +
            self.beta * ssim_loss
        )

        return total_loss, l1_loss, ssim_loss

criterion = RestorationLoss(
    alpha=0.8,
    beta=0.2
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

print("L1 + SSIM training setup ready.")

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model = RestorationUNet().to(device)

print("Model created successfully.")

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class RestorationUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 128x128 -> 256x256
        self.up_final = nn.ConvTranspose2d(64, 32, 2, stride=2)

        # GT is in the range 0-1
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        x = self.up_final(d1)
        x = self.output(x)

        return x


# Create model
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = RestorationUNet().to(device)

print("Model created successfully.")
print("Using device:", device)

# Verify model
test_input = torch.randn(1, 1, 128, 128).to(device)

with torch.no_grad():
    test_output = model(test_input)

print("Input :", test_input.shape)
print("Output:", test_output.shape)
print(
    "Output range:",
    test_output.min().item(),
    "to",
    test_output.max().item()
)

import torch.nn as nn
from pytorch_msssim import ssim


class RestorationLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):
        l1_loss = self.l1(prediction, target)

        ssim_value = ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True
        )

        ssim_loss = 1.0 - ssim_value

        total_loss = (
            self.alpha * l1_loss +
            self.beta * ssim_loss
        )

        return total_loss, l1_loss, ssim_loss


criterion = RestorationLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

print("L1 + SSIM setup ready.")

model.train()

noisy, gt = next(iter(train_loader))

noisy = noisy.to(device)
gt = gt.to(device)

optimizer.zero_grad(set_to_none=True)

prediction = model(noisy)

total_loss, l1_loss, ssim_loss = criterion(
    prediction,
    gt
)

total_loss.backward()
optimizer.step()

print("One L1 + SSIM training step completed.")
print("Total loss:", total_loss.item())
print("L1 loss:", l1_loss.item())
print("SSIM loss:", ssim_loss.item())

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SemiconductorDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        self.files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {noisy_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        noisy = np.load(
            os.path.join(self.noisy_dir, filename)
        ).astype(np.float32)

        gt = np.load(
            os.path.join(self.gt_dir, filename)
        ).astype(np.float32)

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt


noisy_dir = "/content/KLA_train/train/NoisyLR"
gt_dir = "/content/KLA_train/train/GT"

# Check paths
print("Noisy folder exists:", os.path.exists(noisy_dir))
print("GT folder exists:", os.path.exists(gt_dir))

# Create dataset
full_dataset = SemiconductorDataset(
    noisy_dir,
    gt_dir
)

# 80/20 split
train_size = 2560
val_size = 640

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

print("Total:", len(full_dataset))
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import os

train_zip = "/content/drive/MyDrive/train.zip"

print("train.zip exists:", os.path.exists(train_zip))
print("Size (MB):", os.path.getsize(train_zip) / (1024 * 1024))

import zipfile
import os

extract_dir = "/content/KLA_train"

with zipfile.ZipFile(train_zip, "r") as z:
    z.extractall(extract_dir)

print("Extracted:")
print(os.listdir(extract_dir))

from google.colab import drive
import zipfile
import os

# Mount Google Drive
drive.mount("/content/drive", force_remount=True)

# Persistent ZIP location
train_zip = "/content/drive/MyDrive/train.zip"

# Check that the ZIP exists
if not os.path.exists(train_zip):
    raise FileNotFoundError(
        f"train.zip not found at: {train_zip}"
    )

print("train.zip found:", train_zip)
print("Size:", round(os.path.getsize(train_zip) / (1024 * 1024), 2), "MB")

# Extract training dataset
extract_dir = "/content/KLA_train"

with zipfile.ZipFile(train_zip, "r") as z:
    z.extractall(extract_dir)

print("\nExtraction complete.")
print("Contents:", os.listdir(extract_dir))

# Verify dataset folders
train_dir = os.path.join(extract_dir, "train")
noisy_dir = os.path.join(train_dir, "NoisyLR")
gt_dir = os.path.join(train_dir, "GT")

print("\nTrain folder exists:", os.path.exists(train_dir))
print("NoisyLR exists:", os.path.exists(noisy_dir))
print("GT exists:", os.path.exists(gt_dir))

if os.path.exists(noisy_dir):
    print("NoisyLR files:",
          len([f for f in os.listdir(noisy_dir) if f.endswith(".npy")]))

if os.path.exists(gt_dir):
    print("GT files:",
          len([f for f in os.listdir(gt_dir) if f.endswith(".npy")]))

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SemiconductorDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        self.files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        noisy = np.load(
            os.path.join(self.noisy_dir, filename)
        ).astype(np.float32)

        gt = np.load(
            os.path.join(self.gt_dir, filename)
        ).astype(np.float32)

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt


noisy_dir = "/content/KLA_train/train/NoisyLR"
gt_dir = "/content/KLA_train/train/GT"

full_dataset = SemiconductorDataset(noisy_dir, gt_dir)

train_dataset, val_dataset = random_split(
    full_dataset,
    [2560, 640],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

print("Total:", len(full_dataset))
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))

print("Model:", type(model).__name__)
print("Loss:", type(criterion).__name__)
print("Device:", device)

import torch
import torch.nn as nn
from pytorch_msssim import ssim


# ============================================================
# 1. Model
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class RestorationUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 128x128 -> 256x256
        self.up_final = nn.ConvTranspose2d(64, 32, 2, stride=2)

        # GT is 0-1
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        x = self.up_final(d1)
        x = self.output(x)

        return x


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. Create model
# ============================================================

model = RestorationUNet().to(device)


# ============================================================
# 4. L1 + SSIM loss
# ============================================================

class RestorationLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):

        l1_loss = self.l1(prediction, target)

        ssim_value = ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True
        )

        ssim_loss = 1.0 - ssim_value

        total_loss = (
            self.alpha * l1_loss +
            self.beta * ssim_loss
        )

        return total_loss, l1_loss, ssim_loss


criterion = RestorationLoss()


# ============================================================
# 5. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)


print("Model:", type(model).__name__)
print("Loss:", type(criterion).__name__)
print("Optimizer:", type(optimizer).__name__)

!pip -q install pytorch-msssim

from pytorch_msssim import ssim
print("pytorch-msssim installed successfully")

import torch
import torch.nn as nn
from pytorch_msssim import ssim

model.train()

noisy, gt = next(iter(train_loader))

noisy = noisy.to(device)
gt = gt.to(device)

optimizer.zero_grad(set_to_none=True)

prediction = model(noisy)

total_loss, l1_loss, ssim_loss = criterion(
    prediction,
    gt
)

total_loss.backward()
optimizer.step()

print("One L1 + SSIM training step completed.")
print("Total loss:", total_loss.item())
print("L1 loss:", l1_loss.item())
print("SSIM loss:", ssim_loss.item())

!pip -q install pytorch-msssim

import torch
import torch.nn as nn
from pytorch_msssim import ssim

print("SSIM package ready")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class RestorationUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.up_final = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        x = self.up_final(d1)
        return self.output(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RestorationUNet().to(device)

print("Model created")
print("Device:", device)

class RestorationLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):
        l1_loss = self.l1(prediction, target)

        ssim_value = ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True
        )

        ssim_loss = 1.0 - ssim_value

        total_loss = (
            self.alpha * l1_loss +
            self.beta * ssim_loss
        )

        return total_loss, l1_loss, ssim_loss


criterion = RestorationLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

print("Loss and optimizer ready")

print("Model:", type(model).__name__)
print("Loss:", type(criterion).__name__)
print("Device:", device)

model.train()

noisy, gt = next(iter(train_loader))

noisy = noisy.to(device)
gt = gt.to(device)

optimizer.zero_grad(set_to_none=True)

prediction = model(noisy)

total_loss, l1_loss, ssim_loss = criterion(
    prediction,
    gt
)

total_loss.backward()
optimizer.step()

print("One L1 + SSIM training step completed.")
print("Total loss:", total_loss.item())
print("L1 loss:", l1_loss.item())
print("SSIM loss:", ssim_loss.item())

# Use 100 images for CPU experiment
small_train_dataset = torch.utils.data.Subset(
    full_dataset,
    list(range(100))
)

small_train_loader = DataLoader(
    small_train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

num_epochs = 3

for epoch in range(num_epochs):

    model.train()

    total_loss_sum = 0.0
    l1_sum = 0.0
    ssim_sum = 0.0

    for noisy, gt in small_train_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(noisy)

        total_loss, l1_loss, ssim_loss = criterion(
            prediction,
            gt
        )

        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        l1_sum += l1_loss.item()
        ssim_sum += ssim_loss.item()

    n = len(small_train_loader)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Total: {total_loss_sum/n:.6f} | "
        f"L1: {l1_sum/n:.6f} | "
        f"SSIM Loss: {ssim_sum/n:.6f}"
    )

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

model.eval()

val_total = 0.0
psnr_values = []
ssim_values = []

with torch.no_grad():
    for noisy, gt in val_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        prediction = model(noisy)

        total_loss, l1_loss, ssim_loss = criterion(
            prediction,
            gt
        )

        val_total += total_loss.item()

        pred_np = prediction.cpu().numpy()
        gt_np = gt.cpu().numpy()

        for i in range(pred_np.shape[0]):

            pred_img = pred_np[i, 0]
            gt_img = gt_np[i, 0]

            data_range = float(gt_img.max() - gt_img.min())

            if data_range <= 0:
                data_range = 1.0

            psnr = peak_signal_noise_ratio(
                gt_img,
                pred_img,
                data_range=data_range
            )

            ssim_value = structural_similarity(
                gt_img,
                pred_img,
                data_range=data_range
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim_value)

val_total /= len(val_loader)

print("Validation Total Loss:", val_total)
print("Validation PSNR:", np.mean(psnr_values), "dB")
print("Validation SSIM:", np.mean(ssim_values))

import os
import torch

os.makedirs("/content/checkpoints", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_total_loss": float(val_total),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "training_images": 100,
        "epochs": 3,
        "loss": "0.8*L1 + 0.2*(1-SSIM)"
    },
    "/content/checkpoints/l1_ssim_baseline.pth"
)

print("L1 + SSIM baseline saved.")

import shutil
import os

drive_dir = "/content/drive/MyDrive/semiconductor_restoration"
os.makedirs(drive_dir, exist_ok=True)

shutil.copy2(
    "/content/checkpoints/l1_ssim_baseline.pth",
    drive_dir
)

print("Saved to Google Drive.")

import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import torch

device = torch.device("cuda")

print("Using device:", device)
print("GPU:", torch.cuda.get_device_name(0))

# Fresh model for the final experiment
model = RestorationUNet().to(device)

# L1 + SSIM loss
criterion = RestorationLoss(alpha=0.8, beta=0.2)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

# Learning-rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

print("Final model initialized.")

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class RestorationUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 128x128 -> 256x256
        self.up_final = nn.ConvTranspose2d(64, 32, 2, stride=2)

        # GT range is 0-1
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        x = self.up_final(d1)
        x = self.output(x)

        return x


device = torch.device("cuda")
model = RestorationUNet().to(device)

print("Model created successfully.")
print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))

from pytorch_msssim import ssim

class RestorationLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):
        l1_loss = self.l1(prediction, target)

        ssim_value = ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True
        )

        ssim_loss = 1.0 - ssim_value

        total_loss = (
            self.alpha * l1_loss +
            self.beta * ssim_loss
        )

        return total_loss, l1_loss, ssim_loss


criterion = RestorationLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

print("L1 + SSIM setup ready.")

!pip -q install pytorch-msssim


from pytorch_msssim import ssim
print("pytorch-msssim is ready")

import torch
import torch.nn as nn
from pytorch_msssim import ssim

class RestorationLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):
        l1_loss = self.l1(prediction, target)

        ssim_value = ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True
        )

        ssim_loss = 1.0 - ssim_value

        total_loss = (
            self.alpha * l1_loss +
            self.beta * ssim_loss
        )

        return total_loss, l1_loss, ssim_loss


criterion = RestorationLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

print("L1 + SSIM setup ready.")


model.train()

noisy, gt = next(iter(train_loader))

noisy = noisy.to(device, non_blocking=True)
gt = gt.to(device, non_blocking=True)

optimizer.zero_grad(set_to_none=True)

prediction = model(noisy)

total_loss, l1_loss, ssim_loss = criterion(
    prediction,
    gt
)

total_loss.backward()
optimizer.step()

print("GPU test successful.")
print("Input:", noisy.shape)
print("Output:", prediction.shape)
print("Total loss:", total_loss.item())
print("L1 loss:", l1_loss.item())
print("SSIM loss:", ssim_loss.item())
print(
    "GPU memory:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

noisy_dir = "/content/KLA_train/train/NoisyLR"
gt_dir = "/content/KLA_train/train/GT"

class SemiconductorDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        self.files = sorted(
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        noisy = np.load(
            os.path.join(self.noisy_dir, filename)
        ).astype(np.float32)

        gt = np.load(
            os.path.join(self.gt_dir, filename)
        ).astype(np.float32)

        return (
            torch.from_numpy(noisy).unsqueeze(0),
            torch.from_numpy(gt).unsqueeze(0)
        )

full_dataset = SemiconductorDataset(noisy_dir, gt_dir)

train_dataset, val_dataset = random_split(
    full_dataset,
    [2560, 640],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Total:", len(full_dataset))
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

from google.colab import drive
import zipfile
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# 1. Mount Google Drive
# ============================================================
drive.mount("/content/drive", force_remount=True)

train_zip = "/content/drive/MyDrive/train.zip"
extract_dir = "/content/KLA_train"

if not os.path.exists(train_zip):
    raise FileNotFoundError(f"Cannot find: {train_zip}")

# ============================================================
# 2. Extract if needed
# ============================================================
noisy_dir = "/content/KLA_train/train/NoisyLR"
gt_dir = "/content/KLA_train/train/GT"

if not (os.path.exists(noisy_dir) and os.path.exists(gt_dir)):
    print("Extracting train.zip...")

    with zipfile.ZipFile(train_zip, "r") as z:
        z.extractall(extract_dir)

print("NoisyLR folder:", os.path.exists(noisy_dir))
print("GT folder:", os.path.exists(gt_dir))

# ============================================================
# 3. Dataset
# ============================================================
class SemiconductorDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

        self.files = sorted([
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        ])

        if len(self.files) == 0:
            raise RuntimeError("No .npy files found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        noisy = np.load(
            os.path.join(self.noisy_dir, filename)
        ).astype(np.float32)

        gt = np.load(
            os.path.join(self.gt_dir, filename)
        ).astype(np.float32)

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt


full_dataset = SemiconductorDataset(
    noisy_dir,
    gt_dir
)

# ============================================================
# 4. Train / validation split
# ============================================================
train_dataset, val_dataset = random_split(
    full_dataset,
    [2560, 640],
    generator=torch.Generator().manual_seed(42)
)

# ============================================================
# 5. DataLoaders
# ============================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# 6. Verify
# ============================================================
print("\nDataset ready!")
print("Total images:", len(full_dataset))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

noisy, gt = next(iter(train_loader))

print("Noisy:", noisy.shape)
print("GT:", gt.shape)

model.train()

noisy, gt = next(iter(train_loader))

noisy = noisy.to(device, non_blocking=True)
gt = gt.to(device, non_blocking=True)

optimizer.zero_grad(set_to_none=True)

prediction = model(noisy)

total_loss, l1_loss, ssim_loss = criterion(
    prediction,
    gt
)

total_loss.backward()
optimizer.step()

print("GPU test successful")
print("Input :", noisy.shape)
print("GT    :", gt.shape)
print("Output:", prediction.shape)
print("Total loss:", total_loss.item())
print("L1 loss:", l1_loss.item())
print("SSIM loss:", ssim_loss.item())
print(
    "GPU memory:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)

import os
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

num_epochs = 30
checkpoint_dir = "/content/final_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

best_val_loss = float("inf")

# Mixed precision for T4
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):

    model.train()

    train_total = 0.0
    train_l1 = 0.0
    train_ssim_loss = 0.0

    for noisy, gt in train_loader:

        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            prediction = model(noisy)

            total_loss, l1_loss, ssim_loss = criterion(
                prediction,
                gt
            )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_total += total_loss.item()
        train_l1 += l1_loss.item()
        train_ssim_loss += ssim_loss.item()

    train_total /= len(train_loader)
    train_l1 /= len(train_loader)
    train_ssim_loss /= len(train_loader)

    # Validation
    model.eval()

    val_total = 0.0
    psnr_values = []
    ssim_values = []

    with torch.no_grad():

        for noisy, gt in val_loader:

            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                prediction = model(noisy)

                total_loss, _, _ = criterion(
                    prediction,
                    gt
                )

            val_total += total_loss.item()

            pred_np = prediction.float().cpu().numpy()
            gt_np = gt.float().cpu().numpy()

            for i in range(pred_np.shape[0]):

                pred_img = pred_np[i, 0]
                gt_img = gt_np[i, 0]

                data_range = float(
                    gt_img.max() - gt_img.min()
                )

                if data_range <= 0:
                    data_range = 1.0

                psnr_values.append(
                    peak_signal_noise_ratio(
                        gt_img,
                        pred_img,
                        data_range=data_range
                    )
                )

                ssim_values.append(
                    structural_similarity(
                        gt_img,
                        pred_img,
                        data_range=data_range
                    )
                )

    val_total /= len(val_loader)

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    scheduler.step(val_total)

    lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch [{epoch+1}/{num_epochs}]"
        f"\nTrain Loss: {train_total:.6f}"
        f"\nVal Loss:   {val_total:.6f}"
        f"\nPSNR:       {mean_psnr:.4f} dB"
        f"\nSSIM:       {mean_ssim:.4f}"
        f"\nLR:         {lr:.2e}"
    )

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_total,
        "val_loss": val_total,
        "psnr": mean_psnr,
        "ssim": mean_ssim
    }

    torch.save(
        checkpoint,
        os.path.join(checkpoint_dir, "latest_model.pth")
    )

    if val_total < best_val_loss:

        best_val_loss = val_total

        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, "best_model.pth")
        )

        print("✅ Best model saved.")

import shutil
import os

drive_dir = "/content/drive/MyDrive/semiconductor_restoration"
os.makedirs(drive_dir, exist_ok=True)

shutil.copy2(
    "/content/final_checkpoints/best_model.pth",
    drive_dir
)

print("Final best model copied to Google Drive.")


import os
import torch

model_path = "/content/drive/MyDrive/semiconductor_restoration/best_model.pth"

print("Model exists:", os.path.exists(model_path))

if os.path.exists(model_path):
    print(
        "Model size:",
        round(os.path.getsize(model_path) / (1024 * 1024), 2),
        "MB"
    )

from google.colab import files

uploaded = files.upload()

print("\nUploaded files:")
for filename in uploaded:
    print(filename)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# --------------------------------------------------
# Output folder
# --------------------------------------------------
comparison_dir = "/content/visual_comparisons"
os.makedirs(comparison_dir, exist_ok=True)

# --------------------------------------------------
# Use validation samples
# --------------------------------------------------
model.eval()

# Get one validation batch
noisy_batch, gt_batch = next(iter(val_loader))

noisy_batch = noisy_batch.to(device)
gt_batch = gt_batch.to(device)

with torch.no_grad():
    restored_batch = model(noisy_batch)

# --------------------------------------------------
# Generate comparisons
# --------------------------------------------------
num_images = min(8, noisy_batch.shape[0])

for i in range(num_images):

    noisy = noisy_batch[i, 0].cpu().numpy()
    restored = restored_batch[i, 0].cpu().numpy()
    gt = gt_batch[i, 0].cpu().numpy()

    plt.figure(figsize=(15, 5))

    # Noisy
    plt.subplot(1, 3, 1)
    plt.imshow(noisy, cmap="gray")
    plt.title("NoisyLR\n128 × 128")
    plt.axis("off")

    # Restored
    plt.subplot(1, 3, 2)
    plt.imshow(restored, cmap="gray")
    plt.title("AI Restored\n256 × 256")
    plt.axis("off")

    # Ground Truth
    plt.subplot(1, 3, 3)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth\n256 × 256")
    plt.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        comparison_dir,
        f"comparison_{i+1:02d}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

print("Visual comparisons generated.")
print("Saved in:", comparison_dir)

import shutil
import os

drive_comparison_dir = (
    "/content/drive/MyDrive/"
    "semiconductor_restoration/"
    "visual_comparisons"
)

os.makedirs(drive_comparison_dir, exist_ok=True)

for filename in os.listdir(comparison_dir):

    if filename.endswith(".png"):

        shutil.copy2(
            os.path.join(comparison_dir, filename),
            os.path.join(drive_comparison_dir, filename)
        )

print("Visual comparison images saved to Google Drive.")
print(drive_comparison_dir)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Output folder
comparison_dir = "/content/visual_comparisons"
os.makedirs(comparison_dir, exist_ok=True)

# Put model in evaluation mode
model.eval()

# Get one validation batch
noisy_batch, gt_batch = next(iter(val_loader))

noisy_batch = noisy_batch.to(device)
gt_batch = gt_batch.to(device)

# AI restoration
with torch.no_grad():
    restored_batch = model(noisy_batch)

print("Noisy:", noisy_batch.shape)
print("Restored:", restored_batch.shape)
print("GT:", gt_batch.shape)

# Generate comparisons
num_images = min(8, noisy_batch.shape[0])

for i in range(num_images):

    noisy = noisy_batch[i, 0].detach().cpu().numpy()
    restored = restored_batch[i, 0].detach().cpu().numpy()
    gt = gt_batch[i, 0].detach().cpu().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(noisy, cmap="gray")
    plt.title("NoisyLR (Input)\n128 × 128")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(restored, cmap="gray")
    plt.title("AI Restored\n256 × 256")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth\n256 × 256")
    plt.axis("off")

    plt.tight_layout()

    save_path = os.path.join(
        comparison_dir,
        f"comparison_{i+1:02d}.png"
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

print("\nVisual comparison completed.")
print("Saved to:", comparison_dir)
