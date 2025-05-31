# src/prune_and_finetune.py
import torch
import torch.nn as nn
import torch.optim as optim

from data.cifar10 import get_cifar10_dataloaders
from models.vit_cifar import VisionTransformer

def mask_out_heads(model, heads_to_prune):
    """
    Zeroes out Q, K, V projection weights for the specified heads in each layer.
    heads_to_prune: dict mapping layer_idx -> list of head indices to prune.
    """
    for layer_idx, prune_list in heads_to_prune.items():
        # Access the MSA module in that layer
        block = model.blocks[layer_idx]
        attn = block.attn  # MultiHeadSelfAttention instance

        dim     = attn.dim           # e.g. 128
        H       = attn.num_heads     # e.g. 4
        d_h     = attn.head_dim      # e.g. 32
        QKV_dim = 3 * dim            # e.g. 384

        with torch.no_grad():
            # Grab the raw qkv weight: shape [dim, 3*dim]
            W = attn.qkv.weight  # a torch.Tensor

            for h in prune_list:
                # Indices for Q columns
                q_start = h * d_h
                q_end   = (h + 1) * d_h
                # Zero Q
                W[:, q_start:q_end] = 0.0

                # Indices for K columns (offset by dim)
                k_start = dim + h * d_h
                k_end   = dim + (h + 1) * d_h
                W[:, k_start:k_end] = 0.0

                # Indices for V columns (offset by 2*dim)
                v_start = 2 * dim + h * d_h
                v_end   = 2 * dim + (h + 1) * d_h
                W[:, v_start:v_end] = 0.0

            # No need to modify the output projection (attn.proj),
            # because if Q/K → 0, attn_probs will be uniform, and V → 0,
            # so the head’s output is effectively zeroed.

def fine_tune(model, train_loader, test_loader, device, epochs=20, lr=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total

        print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}  Test Acc: {acc*100:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load the model and checkpoint
    model = VisionTransformer(
        img_size=32, patch_size=4, in_chans=3, num_classes=10,
        embed_dim=128, depth=6, num_heads=4, mlp_ratio=4.0
    ).to(device)
    ckpt = torch.load("vit_cifar10_checkpoint.pth", map_location=device)
    model.load_state_dict(ckpt)

    # 2) Define which heads to prune (based on analyze_head_similarity output)
    #    Example: prune head 2 in layer 0, head 1 in layer 1:
    heads_to_prune = {
        0: [2],
        1: [1],
        2: [],
        3: [],
        4: [], 
        5: []
    }

    # 3) Mask out those heads
    mask_out_heads(model, heads_to_prune)

    # 4) Get data loaders (fine-tune on full training set)
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128, data_dir="./data/cifar10")

    # 5) Fine-tune for a few epochs with a small LR
    fine_tune(model, train_loader, test_loader, device, epochs=20, lr=1e-5)

    # 6) Save the pruned & fine-tuned model
    torch.save(model.state_dict(), "vit_cifar10_pruned_finetuned.pth")

if __name__ == "__main__":
    main()
