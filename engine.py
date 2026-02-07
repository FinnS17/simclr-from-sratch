"""SimCLR loss/metrics and training loop helpers."""

import torch
import torch.nn.functional as F
import os


def simclr_logits_targets(z1, z2, temp):
    # Math: Cosine Similarity
    # normalize = x / ||x|| -> scales vector to length = 1
    # e.g.:
    # normalize((1,2,3)) * normalize((1,2,3)) = 1 -> same direction
    # normalize((1,2,3)) * normalize((2,4,6)) = 1 -> same direction
    # normalize((1,0,1)) * normalize((0,1,0)) = 0 -> orthogonal
    # normalize((1,2,3)) * normalize((-1,-2,-3)) = -1 -> opposite direction
    z1_norm, z2_norm = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    
    B = z1_norm.shape[0]
    Z = torch.cat([z1_norm, z2_norm], dim=0) # (2B x 128) -> e.g. Z[1] <-> Z[b+1] (same origin, different corruption) -> positive pair
    S = Z @ Z.t() # (2B x 2B) -> all vs all similarity (angle between embeddings of different pictures)

    # scale logits (with temperature -> adjusts sensitivity of softmax)
    logits = S / temp
    N = 2 * B
    
    # mask similarity between same pictures -> logits[i, i]
    mask = torch.eye(N, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(mask, float("-inf"))
    
    # z1 indices [0..B-1] match with z2 indices [B..2B-1]
    # z2 indices [B..2B-1] match back to z1 indices [0..B-1]
    targets = torch.arange(N, device=logits.device) # (128) -> (0,1,2,3..)
    targets = (targets + B) % N
    
    return logits, targets


def train_one_epoch(encoder, projektor, loader, optimizer, device, temp):
    
    encoder.train()
    projektor.train()

    running_loss = 0.0
    total_ex = 0

    for x1, x2 in loader:
        # ( B x 3 x 32 x 32 ) , ( B x 3 x 32 x 32 )
        optimizer.zero_grad()
        x1, x2 = x1.to(device), x2.to(device)
        h1, h2 = encoder(x1), encoder(x2) # (B x 512) - Corruptions Variant 1 / # (B x 512) - Corruptions Variant 2 
        z1, z2 = projektor(h1), projektor(h2) # (B x 128) / (B x 128)
        logits, targets = simclr_logits_targets(z1, z2, temp)  
        loss = F.cross_entropy(logits, targets) # per row: softmax -> - log(p_pos) of targets[i]; average over 2B
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.shape[0]
        total_ex += targets.shape[0]

    avg_loss = running_loss / total_ex
    return avg_loss

@torch.no_grad()
def eval_ssl_metrics(encoder, projektor, loader, device, temp, max_batches=40):
    encoder.eval()
    projektor.eval()

    total_pos_prob = 0.0 # sum softmax probabilities of positive pair
    total_hits = 0 # counts for positive pair with highest probability
    total_ex = 0

    for i, (x1, x2) in enumerate(loader):
        if i >= max_batches:
            break

        x1, x2 = x1.to(device), x2.to(device)
        h1, h2 = encoder(x1), encoder(x2)
        z1, z2 = projektor(h1), projektor(h2)

        logits, targets = simclr_logits_targets(z1, z2, temp)
        N = targets.shape[0]

        # p_pos
        log_probs = F.log_softmax(logits, dim=1)
        p_pos = log_probs[torch.arange(N, device=logits.device), targets].exp()
        total_pos_prob += p_pos.sum().item()

        # top1
        preds = logits.argmax(dim=1)
        total_hits += (preds == targets).sum().item()

        total_ex += N

    p_pos_mean = total_pos_prob / total_ex
    top1 = total_hits / total_ex
    return p_pos_mean, top1




