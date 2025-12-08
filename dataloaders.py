import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


def cache_dataset(dataset, desc="Caching"):
    """
    Iterates through the slow dataset once, loading all data into CPU RAM.
    Returns a fast TensorDataset.
    """
    print(f"[{desc}] Pre-generating data... (this takes ~1-2 mins but saves hours later)")

    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=10)

    all_images = []
    all_labels = []
    all_locs = []
    for imgs, lbls, locs in loader:
        all_images.append(imgs)
        all_labels.append(lbls)
        all_locs.append(locs)

    # Concatenate all batches into single tensors
    train_x = torch.cat(all_images)
    train_y = torch.cat(all_labels)
    train_loc = torch.cat(all_locs)

    print(f"[{desc}] Done. Dataset size: {train_x.shape}")
    return TensorDataset(train_x, train_y, train_loc)


class ClutteredMNIST(Dataset):
    def __init__(
        self,
        base_dataset,
        center_indices,
        clutter_indices=None,
        image_size=64,
        n_clutter=40,
        augment_center=False,
        seed=0,
    ):
        self.base = base_dataset
        self.center_idx = np.asarray(center_indices, dtype=np.int64)
        self.clutter_idx = np.asarray(
            np.arange(len(base_dataset)) if clutter_indices is None else clutter_indices, dtype=np.int64
        )
        self.H = self.W = image_size
        self.n_clutter = n_clutter
        self.augment_center = augment_center
        self.patch_size = 28
        self.min_crop, self.max_crop = 3, 9
        self.min_up, self.max_up = 1.0, 1.5
        self.alpha_thresh = 0.05
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.center_idx)

    def _paste_masked(self, canvas, patch, top, left):
        _, Hc, Wc = canvas.shape
        _, Hp, Wp = patch.shape
        y0, x0 = max(top, 0), max(left, 0)
        y1, x1 = min(top + Hp, Hc), min(left + Wp, Wc)
        if y1 <= y0 or x1 <= x0:
            return
        py0, px0 = y0 - top, x0 - left
        py1, px1 = py0 + (y1 - y0), px0 + (x1 - x0)
        c = canvas[:, y0:y1, x0:x1]
        p = patch[:, py0:py1, px0:px1]
        m = (p > self.alpha_thresh).float()
        canvas[:, y0:y1, x0:x1] = c * (1 - m) + p * m

    def _sample_clutter_patch(self):
        idx = int(self.rng.choice(self.clutter_idx))
        img, _ = self.base[idx]  # (1,28,28)
        _, h, w = img.shape
        s = int(self.rng.randint(self.min_crop, self.max_crop + 1))
        s = min(s, h, w)
        if h > s and w > s:
            y0 = int(self.rng.randint(0, h - s + 1))
            x0 = int(self.rng.randint(0, w - s + 1))
            patch = img[:, y0 : y0 + s, x0 : x0 + s]
        else:
            patch = img
            s = min(h, w)
        up = float(self.rng.uniform(self.min_up, self.max_up))
        tsize = max(6, min(int(round(s * up)), 24))
        patch = F.resize(patch, [tsize, tsize])
        angle = float(self.rng.uniform(-45, 45))
        return F.rotate(patch, angle)

    def __getitem__(self, i):
        i = int(i)
        img_center, label = self.base[int(self.center_idx[i])]
        canvas = torch.zeros(1, self.H, self.W)

        if self.augment_center:
            angle = float(self.rng.uniform(-15, 15))
            scale = float(self.rng.uniform(0.9, 1.1))
        else:
            angle, scale = 0.0, 1.0
        img_center = F.affine(img_center, angle=angle, translate=[0, 0], scale=scale, shear=[0.0, 0.0])

        margin = self.patch_size // 2
        cx = float(self.rng.uniform(margin, self.W - margin))
        cy = float(self.rng.uniform(margin, self.H - margin))
        top_c = int(round(cy - self.patch_size / 2))
        left_c = int(round(cx - self.patch_size / 2))

        for _ in range(self.n_clutter):
            patch = self._sample_clutter_patch()
            ph, pw = patch.shape[1], patch.shape[2]
            left = int(self.rng.randint(0, max(1, self.W - pw + 1)))
            top = int(self.rng.randint(0, max(1, self.H - ph + 1)))
            self._paste_masked(canvas, patch, top, left)

        self._paste_masked(canvas, img_center, top_c, left_c)

        center_xy = torch.tensor([cx / (self.W - 1), cy / (self.H - 1)], dtype=torch.float32)
        return canvas.clamp(0.0, 1.0), torch.tensor(label, dtype=torch.long), center_xy


def get_mnist_cluttered_loaders(
    root="./data",
    batch_size=64,
    val_fraction=0.1,
    image_size=64,
    n_clutter=40,
    cache=False,
    num_workers_train=4,
    num_workers_val=2,
):
    base_train = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    base_test = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())

    n_total = len(base_train)
    n_val = int(val_fraction * n_total)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_total)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = ClutteredMNIST(
        base_train, train_idx, train_idx, image_size=image_size, n_clutter=n_clutter, augment_center=True, seed=0
    )
    val_ds = ClutteredMNIST(
        base_train, val_idx, val_idx, image_size=image_size, n_clutter=n_clutter, augment_center=False, seed=1
    )
    test_idx = np.arange(len(base_test))
    test_ds = ClutteredMNIST(
        base_test, test_idx, test_idx, image_size=image_size, n_clutter=n_clutter, augment_center=False, seed=2
    )

    if cache:
        train_ds_fast = cache_dataset(train_ds, desc="Train")
        val_ds_fast = cache_dataset(val_ds, desc="Val")
        test_ds_fast = cache_dataset(test_ds, desc="Test")
        print("Loaded datasets into RAM.")
        train_loader = DataLoader(
            train_ds_fast, batch_size=batch_size, shuffle=True, num_workers=num_workers_train, prefetch_factor=10
        )
        val_loader = DataLoader(
            val_ds_fast, batch_size=batch_size, shuffle=False, num_workers=num_workers_val, prefetch_factor=10
        )
        test_loader = DataLoader(test_ds_fast, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers_train, prefetch_factor=10
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers_val, prefetch_factor=10
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


class PairedClutteredMNIST(ClutteredMNIST):
    def __getitem__(self, i):
        i = int(i)

        # 1. Retrieve the base digit
        img_center_orig, label = self.base[int(self.center_idx[i])]

        # 2. Create specific seeds to ensure reproducibility and separation
        # We use a unique seed per index 'i' so the dataset is consistent across epochs
        # We separate clutter seed from location seed
        seed_clutter = int(i * 1000 + 1)
        seed_loc1 = int(i * 1000 + 2)
        seed_loc2 = int(i * 1000 + 3)

        # Create a dedicated RNG for the shared elements (clutter + digit aug)
        rng_shared = np.random.RandomState(seed_clutter)

        # A. Prepare the augmented digit (shared transformation)
        if self.augment_center:
            angle = float(rng_shared.uniform(-15, 15))
            scale = float(rng_shared.uniform(0.9, 1.1))
        else:
            angle, scale = 0.0, 1.0

        img_center = F.affine(img_center_orig, angle=angle, translate=[0, 0], scale=scale, shear=[0.0, 0.0])

        # B. Generate the Clutter Canvas
        # We generate the clutter once on a blank canvas
        clutter_canvas = torch.zeros(1, self.H, self.W)

        # We must temporarily swap self.rng because helper methods rely on it
        original_rng = self.rng
        self.rng = rng_shared

        for _ in range(self.n_clutter):
            patch = self._sample_clutter_patch()
            ph, pw = patch.shape[1], patch.shape[2]
            left = int(self.rng.randint(0, max(1, self.W - pw + 1)))
            top = int(self.rng.randint(0, max(1, self.H - ph + 1)))
            self._paste_masked(clutter_canvas, patch, top, left)

        # Restore original rng just in case
        self.rng = original_rng

        # Helper to paste the digit at a specific location
        def create_variant(rng_seed):
            rng_loc = np.random.RandomState(rng_seed)

            # Start with the pre-generated clutter canvas
            # clone() is crucial so we don't overwrite the shared canvas
            canvas = clutter_canvas.clone()

            margin = self.patch_size // 2
            cx = float(rng_loc.uniform(margin, self.W - margin))
            cy = float(rng_loc.uniform(margin, self.H - margin))

            top_c = int(round(cy - self.patch_size / 2))
            left_c = int(round(cx - self.patch_size / 2))

            self._paste_masked(canvas, img_center, top_c, left_c)

            center_xy = torch.tensor([cx / (self.W - 1), cy / (self.H - 1)], dtype=torch.float32)
            return canvas.clamp(0.0, 1.0), center_xy

        # 3. Create the two images
        img1, loc1 = create_variant(seed_loc1)
        img2, loc2 = create_variant(seed_loc2)

        # Return tuple: (img1, loc1, img2, loc2, label)
        return img1, loc1, img2, loc2, torch.tensor(label, dtype=torch.long)


def get_paired_loader(root="./data", batch_size=32, image_size=64, n_clutter=50):
    base_test = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    test_idx = np.arange(len(base_test))

    paired_ds = PairedClutteredMNIST(
        base_test, test_idx, test_idx, image_size=image_size, n_clutter=n_clutter, augment_center=False
    )  # Keep center aug false for cleaner comparison

    return DataLoader(paired_ds, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_mnist_cluttered_loaders(
        root="./data",
        batch_size=16,
        val_fraction=0.1,
        image_size=64,
        n_clutter=50,
        num_workers_train=1,
        num_workers_val=1,
    )
    imgs, labels, centers = next(iter(train_loader))
    print(imgs.shape, labels.shape, centers.shape)

    n_show = 8
    plt.figure(figsize=(12, 4))
    for i in range(n_show):
        img = imgs[i, 0].numpy()
        label = labels[i].item()
        cx_norm, cy_norm = centers[i].tolist()
        H, W = img.shape
        cx, cy = cx_norm * (W - 1), cy_norm * (H - 1)

        ax = plt.subplot(2, n_show // 2, i + 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.scatter([cx], [cy], c="red", s=10)
        ax.set_title(str(label), fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
