"""
Looped Latent Controller - Phase 2: Address Head Contrastive Pretraining
Train 3 address heads to cluster semantically similar hidden states.
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, Phase2Config
from model import LoopedLatentController
from dataset import prepare_data
from utils import load_checkpoint, save_checkpoint, Timer

def collect_hidden_states(model, train_loader, config, device):
    """Run frozen model, collect hidden states at evenly spaced positions."""
    model.eval()
    all_hidden = []
    n_collected = 0

    print(f"Collecting {config.n_hidden_states} hidden states...")
    with torch.no_grad():
        for input_ids, _ in train_loader:
            if n_collected >= config.n_hidden_states:
                break
            input_ids = input_ids.to(device)

            text_embeds = model.embed(input_ids)
            S = text_embeds.size(1)
            mask = model.build_attention_mask(S, 0, device)
            x = text_embeds
            for layer in model.layers:
                x = layer(x, model.cos_cache, model.sin_cache, mask)
            x = model.final_norm(x)

            B = x.size(0)
            positions = torch.linspace(0, S - 1, config.positions_per_seq).long()
            for b in range(B):
                for p in positions:
                    all_hidden.append(x[b, p].cpu())
                    n_collected += 1
                    if n_collected >= config.n_hidden_states:
                        break
                if n_collected >= config.n_hidden_states:
                    break

            if n_collected % 50000 < B * config.positions_per_seq:
                print(f"  Collected {n_collected} / {config.n_hidden_states}")

    hidden_tensor = torch.stack(all_hidden[:config.n_hidden_states])
    print(f"Collected {hidden_tensor.shape[0]} hidden states")
    return hidden_tensor

def train_phase2(checkpoint_dir, data_dir='./data_cache'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 2: Address Head Contrastive | Device: {device}")

    mc = ModelConfig()
    pc = Phase2Config()

    model = LoopedLatentController(mc).to(device)
    phase1_best = os.path.join(checkpoint_dir, 'phase1', 'best.pt')
    load_checkpoint(model, None, phase1_best, device)

    # Freeze everything except addr heads
    for name, param in model.named_parameters():
        param.requires_grad = 'addr_heads' in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}")

    tokenizer_path = os.path.join(data_dir, 'tokenizer.json')
    train_ds, _, tokenizer = prepare_data(tokenizer_path, data_dir, seq_len=mc.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2,
                               pin_memory=True, drop_last=True)

    # Collect hidden states
    hidden_cache_path = os.path.join(data_dir, 'phase2_hidden_states.pt')
    if os.path.exists(hidden_cache_path):
        print("Loading cached hidden states...")
        all_hidden = torch.load(hidden_cache_path, weights_only=False)
    else:
        all_hidden = collect_hidden_states(model, train_loader, pc, device)
        torch.save(all_hidden, hidden_cache_path)

    # Optimizer for addr heads only
    addr_params = []
    for head in model.addr_heads:
        addr_params.extend(head.parameters())
    optimizer = torch.optim.AdamW(addr_params, lr=pc.lr)

    timer = Timer()
    N = all_hidden.size(0)

    for step in range(pc.steps):
        indices = torch.randperm(N)[:pc.batch_size]
        batch = all_hidden[indices].to(device)

        batch_norm = F.normalize(batch, p=2, dim=-1)
        sim = torch.mm(batch_norm, batch_norm.t())

        pos_mask = (sim > pc.pos_threshold) & (~torch.eye(pc.batch_size, device=device).bool())
        neg_mask = sim < pc.neg_threshold

        total_loss = torch.tensor(0.0, device=device)

        for k, head in enumerate(model.addr_heads):
            addrs = F.linear(batch, head.weight)

            addr_i = addrs.unsqueeze(1)
            addr_j = addrs.unsqueeze(0)
            dist = (addr_i - addr_j).abs().sum(dim=-1)

            pull = (pos_mask.float() * dist.pow(2)).sum() / max(pos_mask.sum().item(), 1)
            push = (neg_mask.float() * F.relu(pc.margin - dist).pow(2)).sum() / max(neg_mask.sum().item(), 1)

            per_dim_std = addrs.std(dim=0)
            entropy_loss = F.relu(pc.target_dim_std - per_dim_std).mean()

            total_loss = total_loss + pull + push + pc.entropy_weight * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            with torch.no_grad():
                sample_idx = torch.randperm(N)[:1000]
                sample = all_hidden[sample_idx].to(device)
                for k, head in enumerate(model.addr_heads):
                    addrs = F.linear(sample, head.weight)
                    stds = addrs.std(dim=0)
                    print(f"  Head {k} dim stds: {stds.cpu().numpy().round(1)}")

            print(f"Step {step+1}/{pc.steps} | Loss: {total_loss.item():.4f} | "
                  f"Pos: {pos_mask.sum().item()} | Neg: {neg_mask.sum().item()} | "
                  f"Elapsed: {timer.elapsed()/60:.1f}m")

    # Save and freeze
    save_path = os.path.join(checkpoint_dir, 'phase2', 'addr_heads.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'addr_heads_state': [head.state_dict() for head in model.addr_heads],
        'model_state_dict': model.state_dict(),
    }, save_path)
    print(f"Phase 2 complete. Address heads saved to {save_path}")
    return save_path