# %%
import os

import torch

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from models import RCNN
from dataloaders import get_mnist_cluttered_loaders, get_paired_loader, PairedClutteredMNIST
from train import train_model, eval_model
from argparse import ArgumentParser

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

N_CLUTTER = 50
BATCH_SIZE = 256
CACHE = False
NUM_WORKERS_TRAIN = 4
NUM_WORKERS_VAL = 2

TIMESTEPS = 5
MODULATION_TYPE = "multiplicative"
MODEL_PATH = "model_weights_0/rcnn_neuroai_model_5.pth"
PLOT_PATH = "plots/"

parser = ArgumentParser()
# mod = input modulation analysis, decod = decoding analysis, perp = perturbation analysis
parser.add_argument("--mode", type=str, default="all", help="Mode: all, mod, decod, perp")


def plot_input_modulation(
    model, sample_input, sample_label, sample_coords, timesteps, device, num_samples=3, save_path=None
):
    sample_out = model(sample_input.to(device), timesteps=timesteps, return_actvs=True)

    fig, axs = plt.subplots(num_samples, timesteps, figsize=(3 * timesteps, 4 * num_samples))
    fig.suptitle("Input Modulation over Timesteps", fontsize=16)

    for s in range(min(num_samples, sample_input.shape[0])):
        for t in range(timesteps):
            # include true and predicted labels in title
            img = sample_out["input"][t][s].squeeze().detach().cpu().numpy()
            # rescale input to [0,1]
            # img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            label = sample_label[s].item()
            predicted_label = torch.argmax(sample_out["output"][t][s]).item()
            axs[s, t].imshow(img, cmap="gray")
            axs[s, t].set_title(f"T {t+1} | True: {label}, Pred: {predicted_label}")
            axs[s, t].axis("off")
            # highlight target location
            cx, cy = sample_coords[s].numpy()
            axs[s, t].scatter([cx * 64], [cy * 64], s=100, c="red", marker="x")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    # return fig, axs


# collect activations on given data from all layers and timesteps
def collect_activations(model, loader, timesteps, device, max_samples=1200):
    """
    Runs the model on `loader` and collects activations for all layers and timesteps,
    plus labels and continuous center locations.

    Returns:
        acts[layer][t] -> np.array [N, D]
        labels -> np.array [N]
        centers -> np.array [N, 2]  (cx_norm, cy_norm)
    """
    model.eval()

    # acts[layer][t] = [batch1, batch2, ...] -> later concatenated
    acts = {}
    all_labels = []
    all_centers = []

    n_seen = 0

    with torch.no_grad():
        for imgs, labels, centers in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            batch_size = imgs.size(0)
            if n_seen >= max_samples:
                break
            if n_seen + batch_size > max_samples:
                # cut batch to not exceed max_samples
                cut = max_samples - n_seen
                imgs = imgs[:cut]
                labels = labels[:cut]
                centers = centers[:cut]
                batch_size = cut

            activs = model(imgs, timesteps=timesteps, return_actvs=True)

            for key in activs.keys():
                if key not in acts:
                    acts[key] = {t: [] for t in range(timesteps)}
                for t in range(timesteps):
                    a = activs[key][t]
                    a = a.view(a.size(0), -1)  # flatten
                    acts[key][t].append(a.cpu().numpy())

            all_labels.append(labels.cpu().numpy())
            all_centers.append(centers.cpu().numpy())
            n_seen += batch_size

    # concatenate batches
    for layer in acts.keys():
        for t in range(timesteps):
            acts[layer][t] = np.concatenate(acts[layer][t], axis=0)

    labels = np.concatenate(all_labels, axis=0)
    centers = np.concatenate(all_centers, axis=0)

    return acts, labels, centers


def build_auxiliary_labels(labels, centers):
    y_category = labels.astype(np.int64)

    x_norm = centers[:, 0]
    y_norm = centers[:, 1]

    # 2-way: left/right, top/bottom
    y_xloc = (x_norm > 0.5).astype(np.int64)  # 0 = left, 1 = right
    y_yloc = (y_norm > 0.5).astype(np.int64)  # 0 = top, 1 = bottom

    # 2d to 1d unique coords
    coord_1d = x_norm + np.max(x_norm) * y_norm
    # 4-way quadrant: 0 TL, 1 TR, 2 BL, 3 BR
    y_quad = y_xloc + 2 * y_yloc

    variables = {
        "category": {"values": y_category, "type": "categorical"},
        "loc_quadrant": {"values": y_quad, "type": "categorical"},
        "loc_x_y_to_1d": {"values": (coord_1d), "type": "continuous"},
        "x_location": {"values": y_xloc, "type": "continuous"},
        "y_location": {"values": y_yloc, "type": "continuous"},
    }
    return variables


def decode_variable(regressor, X, y, catigorical, train_size=0.8, n_repeats=5, random_state=0):
    rng = np.random.default_rng(random_state)
    n_total = X.shape[0]
    scores = []

    for r in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            stratify=y if catigorical else None,
            random_state=int(rng.integers(0, 100_000)),
        )

        regressor.fit(X_train, y_train)
        score = regressor.score(X_test, y_test)
        scores.append(score)

    scores = np.array(scores)
    return scores.mean(), scores.std()


def decode_all_variables(acts, variables, layers, timesteps, max_features=2000, use_pca=False, n_repeats=5):
    results_mean = {var: {layer: np.zeros(timesteps) for layer in layers} for var in variables}
    results_std = {var: {layer: np.zeros(timesteps) for layer in layers} for var in variables}

    # print shape info
    print("Variables to decode:")
    print({k: [v["values"].shape, v["type"]] for k, v in variables.items()})
    print("Activation shapes:")
    X_reduced = {layer: {} for layer in layers}
    for layer in layers:
        print(f"- Layer {layer:10s}: {acts[layer][0].shape}")
        for t in range(timesteps):
            X = acts[layer][t]
            if X.shape[1] > max_features:
                if use_pca:
                    # randomized SVD is MUCH faster here
                    n_comp = min(max_features, X.shape[0], X.shape[1])
                    pca = PCA(
                        n_components=n_comp,
                        svd_solver="randomized",
                        random_state=0,
                    )
                    X_use = pca.fit_transform(X)
                else:
                    feat_idx = np.random.choice(X.shape[1], size=max_features, replace=False)
                    X_use = X[:, feat_idx]
            else:
                X_use = X
            X_reduced[layer][t] = X_use
        print(f"--> mapped to: {X_reduced[layer][0].shape}")

    for var_name, y in variables.items():
        print(f"\n=== Decoding variable: {var_name} ===")
        catigorical = variables[var_name]["type"] == "categorical"
        if catigorical:
            regressor = LogisticRegression(max_iter=1000, n_jobs=-1)
        else:
            # regressor = LinearRegression(n_jobs=-1)
            regressor = Ridge(alpha=1.0)

        for layer in layers:
            print(f"- Decoding from layer: {layer}")
            for t in range(timesteps):
                X_use = X_reduced[layer][t]
                mean_score, std_score = decode_variable(
                    regressor,
                    X_use,
                    y["values"],
                    catigorical=catigorical,
                    train_size=0.8,
                    n_repeats=n_repeats,
                    random_state=42,
                )
                results_mean[var_name][layer][t] = mean_score
                results_std[var_name][layer][t] = std_score

            score = results_mean[var_name][layer]
            print(f"-Layer {layer:6s}: " + " | ".join([f"t{ti+1}: {a:.1f}" for ti, a in enumerate(score)]))

    return results_mean, results_std, layers, list(variables.keys())


def plot_decoding_results(mean, std, layers, variabl_name, timesteps, save_dir=None, ax=None):
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_plot = True

    markers = ["o", "s", "D", "^", "v", "x", "*"]
    for i, layer in enumerate(layers):
        accs = mean[layer]
        errs = std[layer]
        ax.errorbar(
            np.arange(1, timesteps + 1),
            accs,
            yerr=errs,
            label=layer,
            marker=markers[i % len(markers)],
            capsize=5,
        )

    ax.set_title(f"{variabl_name}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Decoding Accuracy")
    ax.set_xticks(np.arange(1, timesteps + 1))
    ax.set_ylim(0, 1.0)
    ax.grid(True)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"decoding_{variabl_name}.png"), bbox_inches="tight")
    if show_plot:
        ax.legend()
        plt.show(block=False)
        plt.pause(0.001)

    return ax


def perturbation_analysis(model, paired_loader, timesteps, layers_per_ts: dict, device, max_batches=None):
    model.eval()
    base_accuracies_total = []
    pert_accuracies_total = []
    control_accuracies_total = []

    max_batches = min(max_batches, len(paired_loader)) if max_batches is not None else len(paired_loader)
    with torch.no_grad():
        counter = 0
        for base_img, _, perturbed_imgs, _, labels in paired_loader:
            if max_batches is not None and counter >= max_batches:
                break
            counter += 1

            base_img = base_img.to(device)
            perturbed_imgs = perturbed_imgs.to(device)
            labels = labels.to(device)

            base_out = model(base_img, timesteps=timesteps, return_actvs=True)

            perturbed_img_out = model(perturbed_imgs, timesteps=timesteps, return_actvs=True)
            # target activations layer and timestep
            # timesteps start at 0 (index of activations list)
            # layers_per_ts = {"conv2": [0, 1], "conv3": [0, 1]}

            pert_activations = {}
            for t in layers_per_ts.keys():
                pert_activations[t] = {}
                for layer in layers_per_ts[t]:
                    pert_activations[t][layer] = perturbed_img_out[layer][t]

            # create control perturbations inputs dict
            num_control = 10
            pert_activations_control = []
            for n in range(num_control):
                pert_activations_control.append({})
                for t in pert_activations.keys():
                    pert_activations_control[n][t] = {}
                    for layer in pert_activations[t].keys():
                        # original recurrent flow for base image
                        act_orig = base_out[layer][t].detach()
                        # systematic perturbed recurrent flow
                        act_pert = pert_activations[t][layer].detach()

                        # difference vector delta = pert - orig
                        delta = act_pert - act_orig

                        # flatten per sample, permute delta, reshape
                        B = delta.shape[0]
                        flat = delta.view(B, -1)
                        flat_perm = torch.empty_like(flat)
                        for b in range(B):
                            perm = torch.randperm(flat.size(1), device=flat.device)
                            flat_perm[b] = flat[b, perm]
                        delta_perm = flat_perm.view_as(delta)

                        # control recurrent flow: orig + permuted delta
                        act_ctrl = act_orig + delta_perm

                        pert_activations_control[n][t][layer] = act_ctrl

                        # diff_sys = (pert_activations[t][layer] - base_out[layer][t]).view(B, -1).norm(dim=1)
                        # diff_ctrl = (pert_activations_control[0][t][layer] - base_out[layer][t]).view(B, -1).norm(dim=1)
                        # print("mean norm diff sys:", diff_sys.mean().item())
                        # print("mean norm diff ctrl:", diff_ctrl.mean().item())

            # run model with perturbations
            perturbed_out = model(
                base_img,
                timesteps=timesteps,
                return_actvs=True,
                pernurbations_inputs=pert_activations,
            )
            control_outs = []
            for n in range(num_control):
                control_out = model(
                    base_img,
                    timesteps=timesteps,
                    return_actvs=True,
                    pernurbations_inputs=pert_activations_control[n],
                )
                control_outs.append(control_out)

            # get predictions
            base_pred = []
            pert_pred = []
            control_pred = [[] for _ in range(num_control)]
            for t in range(timesteps):
                base_pred.append(torch.argmax(base_out["output"][t], dim=1).cpu().numpy())
                pert_pred.append(torch.argmax(perturbed_out["output"][t], dim=1).cpu().numpy())
                for n in range(num_control):
                    control_pred[n].append(torch.argmax(control_outs[n]["output"][t], dim=1).cpu().numpy())
            # accuracies
            accuracies_base = []
            accuracies_pert = []
            accuracies_control = [[] for _ in range(num_control)]
            for t in range(timesteps):
                acc_base = np.mean(base_pred[t] == labels.numpy())
                acc_pert = np.mean(pert_pred[t] == labels.numpy())
                accuracies_base.append(acc_base)
                accuracies_pert.append(acc_pert)
                for n in range(num_control):
                    acc_control = np.mean(control_pred[n][t] == labels.numpy())
                    accuracies_control[n].append(acc_control)
            avg_accuracies_control = np.mean(np.array(accuracies_control), axis=0)

            base_accuracies_total.append(accuracies_base)
            pert_accuracies_total.append(accuracies_pert)
            control_accuracies_total.append(avg_accuracies_control)

            # print results for this batch
            print(
                f"Batch {counter}/{max_batches}: Base Acc = {accuracies_base[-1]*100:.2f}%, Perturbed Acc = {accuracies_pert[-1]*100:.2f}%, Control Acc = {avg_accuracies_control[-1]*100:.2f}%",
                end="\r",
            )

    return base_accuracies_total, pert_accuracies_total, control_accuracies_total


def compute_functional_importance(base_accuracies_total, pert_accuracies_total, control_accuracies_total, eps=1e-8):
    base = np.asarray(base_accuracies_total)  # [B, T]
    sys = np.asarray(pert_accuracies_total)  # [B, T]
    ctrl = np.asarray(control_accuracies_total)  # [B, T]

    # Functional importance as in Thorat et al. 2021:
    # FI = (Accuracy_control - Accuracy_systematic) / Accuracy_original
    fi_all = (ctrl - sys) / (base + eps)

    fi_mean = fi_all.mean(axis=0)
    fi_std = fi_all.std(axis=0)

    return fi_mean, fi_std, fi_all


# region main


def main():
    pass


# %%
if __name__ == "__main__":

    args = parser.parse_args()
    mode = args.mode.lower()
    # mode = "all"
    print(f"Running analysis mode: {mode}")

    os.makedirs(PLOT_PATH, exist_ok=True)
    print(f"device: {DEVICE}")
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Plots will be saved to: {PLOT_PATH}")

    model = RCNN(modulation_type=MODULATION_TYPE).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    train_loader, test_loader, val_loader = get_mnist_cluttered_loaders(
        root="./data", batch_size=BATCH_SIZE, val_fraction=0.1, image_size=64, n_clutter=N_CLUTTER, cache=CACHE
    )
    # print("sample_inputs shape:", sample_inputs.shape)
    # evaluate model
    # print(model)
    # accuracy = eval_model(model, test_loader, TIMESTEPS, DEVICE)
    # print("Test Accuracy at each timestep: ", ["{:.2f}%".format(acc) for acc in accuracy])

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    # region modulations
    # %%
    # visualization/analysis: input modulations over timesteps
    if mode == "all" or mode == "mod":
        print("\n###input modulations over timesteps...")
        sample_inputs, sample_labels, sample_coords = next(iter(test_loader))

        plot_input_modulation(
            model,
            sample_inputs,
            sample_labels,
            sample_coords,
            timesteps=TIMESTEPS,
            device=DEVICE,
            num_samples=4,
            save_path=os.path.join(PLOT_PATH, "input_modulations.png"),
        )

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    # region decoding
    # %%
    # anaylsis: decoding auxiliary variables
    if mode == "all" or mode == "decod":
        print("\n### decoding auxiliary variables...")

        max_samples = 1000  # you can increase this (e.g. 2000) for more stable estimates
        max_features = 20  # limit number of features for decoding to speed up computation
        use_pca = True
        n_repeats = 5  # number of runs for decoding stability
        sel_layers = ["input", "conv1", "conv2", "conv3", "output", "aux_output"]
        sel_variables = [
            "category",
            "loc_quadrant",
            "loc_x_y_to_1d",
        ]

        print("\nCollecting activations...")
        acts, labels, centers = collect_activations(model, test_loader, TIMESTEPS, DEVICE, max_samples=max_samples)
        print("activations Collected from: ", list(acts.keys()))

        variables = build_auxiliary_labels(labels, centers)
        layers = list(acts.keys())

        # fix layer order for nicer plots if keys are as expected
        sel_layers = sel_layers if sel_layers is not None else layers
        layers = [l for l in layers if l in sel_layers]
        # select variables
        sel_variables = sel_variables if sel_variables is not None else list(variables.keys())
        variables = {k: v for k, v in variables.items() if k in sel_variables}

        # decoding analysis
        print("\nDecoding auxiliary variables...")
        results_mean, results_std, l_names, v_names = decode_all_variables(
            acts,
            variables,
            layers,
            timesteps=TIMESTEPS,
            max_features=max_features,
            use_pca=use_pca,
            n_repeats=5,
        )
        # plot decoding accuracy vs timestep for each variable
        fig, axs = plt.subplots(1, len(variables), figsize=(6 * len(variables), 5))
        fig.suptitle("Decoding Accuracy over Timesteps", fontsize=16)
        for i, var_name in enumerate(variables):
            ax = axs[i] if len(variables) > 1 else axs
            plot_decoding_results(
                results_mean[var_name],
                results_std[var_name],
                layers,
                var_name,
                TIMESTEPS,
                ax=ax,
            )
            if variables[var_name]["type"] == "continuous":
                ax.set_ylabel("R^2 Score")

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"decoding_all_variables.png"), bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.001)

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    # region perpurbations
    # %%
    # analysis: functional importance of of auxiliary variables information in recurrent flow
    # perpurbations test

    if mode == "all" or mode == "perp":
        print("\n### perturbation analysis...")

        max_batches = 5
        # get paired data loader
        paired_loader = get_paired_loader(root="./data", batch_size=BATCH_SIZE, n_clutter=N_CLUTTER)
        print("num of batches: ", len(paired_loader.dataset))
        base_img, base_coords, perturbed_imgs, perturbed_coords, labels = next(iter(paired_loader))

        # example perturbations imgs
        # print("base_img:", base_img.shape)
        # print("base_coords:", base_coords.shape)
        # print("perturbed_imgs:", perturbed_imgs.shape)
        # print("perturbed_coords:", perturbed_coords.shape)
        # print("labels:", labels.shape)
        # visualize some examples
        num_examples = 4
        fig, axs = plt.subplots(num_examples, 2, figsize=(6, 3 * num_examples))
        for i in range(num_examples):
            axs[i, 0].imshow(base_img[i].squeeze(), cmap="gray")
            axs[i, 0].set_title(f"Base Image | Label: {labels[i].item()}")
            axs[i, 0].axis("off")
            axs[i, 1].imshow(perturbed_imgs[i].squeeze(), cmap="gray")
            axs[i, 1].set_title(f"Perturbed Image")
            axs[i, 1].axis("off")
            # highlight target location
            cx, cy = base_coords[i].numpy()
            pcx, pcy = perturbed_coords[i].numpy()
            axs[i, 0].scatter([cx * 64], [cy * 64], s=100, c="red", marker="x")
            axs[i, 1].scatter([cx * 64], [cy * 64], s=100, c="red", marker="x")
            axs[i, 1].scatter([pcx * 64], [pcy * 64], s=100, c="blue", marker="o", facecolors="none", linewidths=2)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"perturbation_examples.png"), bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.001)

        # %%
        layers_per_timesteps = {
            0: [
                "conv2",
                "conv3",
            ],
            1: ["conv2", "conv3"],
        }

        print("Running perturbation analysis...\n", layers_per_timesteps)
        base_acc_total, pert_acc_total, control_acc_total = perturbation_analysis(
            model,
            paired_loader,
            TIMESTEPS,
            layers_per_timesteps,
            DEVICE,
            max_batches=max_batches,
        )
        # %%
        # average over batches for each timestep
        accuracies_base = np.mean(np.array(base_acc_total), axis=0)
        accuracies_pert = np.mean(np.array(pert_acc_total), axis=0)
        accuracies_control = np.mean(np.array(control_acc_total), axis=0)

        # plot results for all timesteps
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("Perurbation Analysis of Recurrent Information Flow\n(conv1 & conv2 to input)", fontsize=16)
        ax = axs[0]
        ax.plot(np.arange(1, TIMESTEPS + 1), accuracies_base * 100, label="Base", marker="o", color="blue")
        ax.plot(np.arange(1, TIMESTEPS + 1), accuracies_pert * 100, label="Perturbed", marker="s", color="green")
        ax.plot(
            np.arange(1, TIMESTEPS + 1), accuracies_control * 100, label="Control (avg)", marker="^", color="orange"
        )
        ax.set_title("Accuracy over Timesteps")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(np.arange(1, TIMESTEPS + 1))
        ax.set_ylim(0, 100)
        ax.grid(True)
        ax.legend()

        # final timestep comparison
        ax = axs[1]
        labels_plot = ["Baseline", "Location\nPerturbed", "Control\nPerturbed (avg)"]
        accuracies_final = [accuracies_base[-1] * 100, accuracies_pert[-1] * 100, accuracies_control[-1] * 100]
        ax.bar(labels_plot, accuracies_final, color=["blue", "green", "orange"])
        ax.set_title("Accuracy at Final Timestep")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        for i, v in enumerate(accuracies_final):
            ax.text(i, v + 1, f"{v:.2f}%", ha="center", fontsize=12)
        # horizontal line for base accuracy
        ax.axhline(y=accuracies_base[-1] * 100, color="red", linestyle="--", label="Baseline Accuracy")
        ax.legend()

        # rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"perurbation_analysis_accuracy.png"), bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.001)

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        # region FI
        # %%
        # compute perburbation from different layers to input
        timesteps_to_plot = np.arange(TIMESTEPS)
        layers_combinations = [["conv2", "conv3"], ["conv2"], ["conv3"]]
        fig, axs = plt.subplots(1, len(layers_combinations), figsize=(6 * len(layers_combinations), 5))
        axs = axs.flatten() if len(layers_combinations) > 1 else [axs]

        fig.suptitle("Functional Importance of Recurrent Flow to Input", fontsize=16)

        for idx, layers in enumerate(layers_combinations):
            ax = axs[idx]
            for i in timesteps_to_plot:
                print(f"Timestep {i+1}, Layers: {layers}")
                l_per_ts = {i: layers}
                base_acc_total_i, pert_acc_total_i, control_acc_total_i = perturbation_analysis(
                    model,
                    paired_loader,
                    TIMESTEPS,
                    l_per_ts,
                    DEVICE,
                    max_batches=max_batches,
                )
                fi_mean_i, fi_std_i, fi_all_i = compute_functional_importance(
                    base_acc_total_i, pert_acc_total_i, control_acc_total_i, eps=1e-8
                )
                ax.errorbar(
                    np.arange(1, TIMESTEPS + 1),
                    fi_mean_i,
                    yerr=fi_std_i,
                    marker="o",
                    capsize=5,
                    label=f"T={i+1}",
                )
            ax.set_title(f"Layers: {' & '.join(layers)}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Functional Importance")
            ax.set_xticks(timesteps_to_plot + 1)
            ax.set_ylim(0, None)
            ax.grid(True)
        # shared legend, ensure it does not overlap with subplots or suptitle
        handles, labels_plot = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels_plot, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"functional_importance_recurrent_flow.png"), bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.001)

# %%
