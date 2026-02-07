"""Plotting utilities for SSL and linear probe logs."""

import os
import torch
import matplotlib.pyplot as plt

def plot_ssl(log_path="logs/ssl_results.pt", out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    logs = torch.load(log_path, map_location="cpu")
    loss = logs["loss"]
    top1 = logs["top1"]
    p_pos = logs["p_pos"]
    cfg = logs.get("config", {})

    # x-axes
    x_loss = list(range(1, len(loss) + 1))
    # metrics were logged every 2 epochs starting at epoch 1 (1,3,5,...)
    x_metrics = list(range(1, 2 * len(top1) + 1, 2))

    title = f"SimCLR SSL | temp={cfg.get('temp','?')} bs={cfg.get('batch_size','?')} lr={cfg.get('lr','?')}"

    # -------------------------
    # Plot 1: Loss only
    # -------------------------
    fig = plt.figure()
    plt.plot(x_loss, loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")
    plt.title(title + " | Loss")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "ssl_loss.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")

    # -------------------------
    # Plot 2: Metrics only
    # -------------------------
    fig = plt.figure()
    plt.plot(x_metrics, top1, label="Top-1 (pos is argmax)")
    plt.plot(x_metrics, p_pos, label="p_pos mean")
    plt.xlabel("Epoch")
    plt.ylabel("Metric (0..1)")
    plt.ylim(0, 1.0)
    plt.title(title + " | Metrics")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "ssl_metrics.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")

def plot_probe(log_path="logs/probe_results.pt", out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    logs = torch.load(log_path, map_location="cpu")
    simclr_loss = logs["simclr_probe_loss"]
    random_loss = logs["random_probe_loss"]
    simclr_acc = logs["simclr_probe_acc"]
    random_acc = logs["random_probe_acc"]
    cfg = logs.get("config", {})

    x = list(range(1, len(simclr_loss) + 1))
    
    # loss curves
    fig = plt.figure()
    plt.plot(x, simclr_loss, label="SimCLR encoder + linear")
    plt.plot(x, random_loss, label="Random encoder + linear")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(f"Linear Probe Train Loss | bs={cfg.get('batch_size','?')} lr={cfg.get('lr','?')}")
    plt.legend()
    plt.tight_layout()
    out_loss = os.path.join(out_dir, "probe_loss.png")
    plt.savefig(out_loss, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved: {out_loss}")

    # accuracy bar plot
    fig = plt.figure()
    plt.bar(["SimCLR", "Random"], [simclr_acc, random_acc])
    plt.ylim(0, 1.0)
    plt.ylabel("Test Top-1 Accuracy")
    plt.title("Linear Probe Test Accuracy")
    plt.tight_layout()
    out_acc = os.path.join(out_dir, "probe_accuracy.png")
    plt.savefig(out_acc, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved: {out_acc}")


def main():
    if os.path.exists("logs/ssl_results.pt"):
        plot_ssl()
    else:
        print("[Skip] logs/ssl_results.pt not found")

    if os.path.exists("logs/probe_results.pt"):
        plot_probe()
    else:
        print("[Skip] logs/probe_results.pt not found")

if __name__ == "__main__":
    main()