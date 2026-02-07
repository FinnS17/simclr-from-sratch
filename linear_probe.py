import os
import torch
import torch.nn.functional as F
from architecture import build_encoder
from helpers import get_device
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms

# -------------------
# DATA
# -------------------
def make_loaders(batch_size):
    # transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616)),
    ])

    # datasets
    train_set = CIFAR10(root="data", train=True, download=True, transform=train_transform)
    test_set = CIFAR10(root="data", train=False, download=True, transform= test_transform)

    # loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# -------------------
# Eval
# -------------------
@torch.no_grad()
def eval_accuracy(encoder, linear, test_loader, device):
    encoder.eval()
    linear.eval()
    correct = 0
    total_ex = 0

    for x, y in test_loader:
        x,y = x.to(device), y.to(device)
        features = encoder(x)
        logits = linear(features)
        preds = logits.argmax(dim=1)
        correct_in_batch = (preds == y).sum().item()
        correct += correct_in_batch
        total_ex += y.shape[0]
    return correct / total_ex

# -------------------
# Linear Probe Run
# -------------------
def run_linear_probe(encoder, train_loader, test_loader, device, name, epochs, lr):
    encoder = encoder.to(device)
    encoder.eval()

    # freeze encoder -> no updates
    for p in encoder.parameters():
        p.requires_grad = False

    # classification layer
    linear = nn.Linear(512, 10, bias=True, device=device)
    optimizer = torch.optim.Adam(params=linear.parameters(), lr=lr)
    
    # training
    train_losses = []
    for epoch in range(epochs):
        linear.train()
        running_loss = 0.0
        total_ex = 0

        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features = encoder(x)
            logits = linear(features)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.shape[0]
            total_ex += y.shape[0]
        train_loss = running_loss / total_ex
        train_losses.append(train_loss)
        print(f" Name: {name} | Epoch: {epoch+1} | Train-Loss: {train_loss:.4f}")
    
    final_acc = eval_accuracy(encoder, linear, test_loader, device)
    return final_acc, train_losses

def main():
    device = get_device()
    print(device)
    
    epochs = 20
    lr= 1e-3
    batch_size = 128

    train_loader, test_loader = make_loaders(batch_size=batch_size)

    # 1. SIMCLR:
    ckpt = torch.load("checkpoints/latest.pt", map_location=device)
    simclr_encoder = build_encoder().to(device)
    simclr_encoder.load_state_dict(ckpt["encoder"])
    print("Start Training of Linear Layer after SIMCLR Encoder")
    simclr_final_acc, simclr_train_losses = run_linear_probe(simclr_encoder, train_loader, test_loader, device, "simclr", epochs=epochs, lr=lr)

    # 2. RANDOM Encoder:
    random_encoder = build_encoder().to(device)
    print("\nStart Training of Linear Layer after Random Encoder")
    random_final_acc, random_train_losses = run_linear_probe(random_encoder, train_loader, test_loader, device, "random", epochs=epochs, lr=lr)

    os.makedirs("logs", exist_ok=True)
    torch.save({
        "simclr_probe_loss": simclr_train_losses,
        "random_probe_loss": random_train_losses,
        "simclr_probe_acc": simclr_final_acc,
        "random_probe_acc": random_final_acc,
        "config": {
            "epochs": epochs,
            "lr": lr, 
            "batch_size": batch_size,
            "dataset": "CIFAR10",
            "encoder": "RESNET18"
        }
    }, "logs/probe_results.pt")

    # RESULTS
    print("\n=== Results ===")
    print(f"SIMCLR Test Acc: {simclr_final_acc:.4f}")
    print(f"RANDOM Test Acc: {random_final_acc:.4f}")
    print(f"DELTA Test Accs: {(simclr_final_acc - random_final_acc):.4f}")

if __name__ == "__main__":
    main()