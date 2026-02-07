from torchvision.datasets import CIFAR10
from helpers import get_transform, SSLAugmentation, get_device
import torch
import os
from architecture import build_encoder, ProjektionHead
import torch.nn.functional as F
from engine import train_one_epoch, eval_ssl_metrics

def main():
    device = get_device()
    print(device)
    os.makedirs("checkpoints", exist_ok=True)

    lr = 3e-4
    weight_decay = 1e-4
    batch_size = 128
    num_epochs = 100
    temp = 0.2

    dataset = CIFAR10(root="data", train=True, download=True)
    transform = get_transform()
    ssl_dataset = SSLAugmentation(dataset, transform)
    train_loader = torch.utils.data.DataLoader(ssl_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    encoder = build_encoder().to(device)
    projektor = ProjektionHead(512, 512, 128).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projektor.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    print("Start of Training")
    train_losses = []
    top1_list = []
    p_pos_list = []

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(encoder, projektor, train_loader, optimizer, device, temp)
        train_losses.append(avg_loss)
        if epoch % 2 == 0:
            p_pos_mean, top1 = eval_ssl_metrics(encoder, projektor, train_loader, device, temp)
            top1_list.append(top1)
            p_pos_list.append(p_pos_mean)
            print(f"Epoch: {epoch+1} | Train-Loss: {avg_loss:.4f} | Top-1 Pred: {top1:.4f} | Right-Class Likelihood: {p_pos_mean:.4f}")
        else:
            print(f"Epoch: {epoch+1} | Train-Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch +1,
                "encoder": encoder.state_dict(),
                "projektor": projektor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "temp": temp
            }, "checkpoints/latest.pt")
            print("Saved checkpoint: checkpoints/latest.pt")
    
            os.makedirs("logs", exist_ok=True)
            torch.save({
                "loss": train_losses,
                "top1": top1_list,
                "p_pos": p_pos_list,
                "config": {
                    "temp": temp,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "num_epochs": num_epochs
                }
                }, "logs/ssl_results.pt")


if __name__ == "__main__":
    main()
