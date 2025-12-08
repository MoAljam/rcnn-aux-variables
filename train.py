import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser

from dataloaders import get_mnist_cluttered_loaders
from models import RCNN

N_CLUTTER = 50
CACHE = False
NUM_WORKERS_TRAIN = 4
NUM_WORKERS_VAL = 2
TIMESTEPS = 2
MODULATION_TYPE = "multiplicative"
N_EPOCHS = 2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
CHECKPOINT_INTERVAL = 0  # 0 -> save only at end, -1 (<0) -> no saving
SAVE_PATH = "model_weights/rcnn_model.pth"
parser = ArgumentParser()
# dataset parameters
parser.add_argument("--n_clutter", type=int, default=N_CLUTTER, help="Number of clutter pieces in each image")
parser.add_argument("--cache", type=bool, default=CACHE, help="Whether to cache datasets in RAM")
parser.add_argument(
    "--num_workers_train", type=int, default=NUM_WORKERS_TRAIN, help="Number of workers for train dataloader"
)
parser.add_argument("--num_workers_val", type=int, default=NUM_WORKERS_VAL, help="Number of workers for val dataloader")
# rcnn parameters
parser.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Number of timesteps for RCNN")
parser.add_argument(
    "--modulation_type",
    type=str,
    default=MODULATION_TYPE,
    help="Type of recurrent modulation: 'additive' or 'multiplicative'",
)
# training parameters
parser.add_argument("--n_epochs", type=int, default=N_EPOCHS, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for optimizer")
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=0,
    help="Interval (in epochs) for saving model checkpoints (0=only at end, -1=no saving)",
)
parser.add_argument("--save_path", type=str, default=SAVE_PATH, help="Path to save the trained model weights")


def eval_model(model, data_loader, timesteps, device):
    model.eval()
    correct = np.zeros(timesteps)
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, timesteps=timesteps)
            for t in range(timesteps):
                _, predicted = torch.max(outputs[t], 1)
                correct[t] += (predicted == labels).sum().item()

    accuracy = correct / len(data_loader.dataset) * 100
    return accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    n_epochs,
    learning_rate,
    timesteps,
    device,
    checkpoint_interval=0,  # 0 -> save only at end, -1 (<0) -> no saving
    save_path=None,
):
    if save_path is None:
        save_path = f"model_weights/rcnn_model_{timesteps}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Model weights will be saved to {save_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        model.train()
        total_loss = 0.0
        total_losses = np.zeros(timesteps)
        counter = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, timesteps=timesteps)
            loss = criterion(outputs[0], labels) / timesteps  # NOTE why / timesteps here?
            loss_t = np.zeros(timesteps)
            loss_t[0] = loss.item()
            total_losses[0] += loss.item() * images.size(0)
            for t in range(1, timesteps):  # loss at each timestep
                loss += criterion(outputs[t], labels) / timesteps
                loss_t[t] = criterion(outputs[t], labels).item() / timesteps
                total_losses[t] += loss_t[t] * images.size(0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            print(
                f"Batch {counter+1}/{len(train_loader)}: Loss = {loss.item():.4f}, losses per timestep = {['{:.4f}'.format(l) for l in loss_t]}",
                end="\r",
            )
            # print(f"  Batch {counter+1}/{len(train_loader)}: Loss = {loss.item():.4f}", end='\r')
            counter += 1

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")

        # Validation
        accuracy = eval_model(model, val_loader, timesteps, device)
        print("Validation Accuracy at each timestep: ", ["{:.2f}%".format(acc) for acc in accuracy])

        # Save model checkpoints
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0 and epoch != n_epochs - 1:
            cp_path = save_path.replace(".pth", f"_ep{epoch+1}_ts_{timesteps}_checkpoint.pth")
            torch.save(model.state_dict(), cp_path)
            print(f"Checkpoint saved to {cp_path}")
        elif epoch == n_epochs - 1 and checkpoint_interval >= 0:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")


if __name__ == "__main__":

    args = parser.parse_args()
    # dataset parameters
    n_clutter = args.n_clutter
    cache_here = args.cache
    num_workers_train = args.num_workers_train
    num_workers_val = args.num_workers_val
    # rcnn parameters
    timesteps = args.timesteps
    modulation_type = args.modulation_type
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    # training parameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    checkpoint_interval = args.checkpoint_interval
    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Setting up data loaders...")
    print(f"final models and checkpoints dir: {os.path.dirname(save_path)}")

    train_loader, val_loader, test_loader = get_mnist_cluttered_loaders(
        root="./data", batch_size=batch_size, val_fraction=0.1, image_size=64, n_clutter=n_clutter, cache=cache_here
    )

    print(f"Using device: {device}")
    print("Setting up model...")

    model = RCNN(modulation_type=modulation_type)

    print("Starting training...")

    train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        timesteps=timesteps,
        device=device,
        checkpoint_interval=checkpoint_interval,
        save_path=save_path,
    )
