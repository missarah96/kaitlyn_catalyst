import torch
import torch.nn as nn

class AugmentedSpeciesNet(nn.Module):
    def __init__(self, base_model, original_outputs, target_labels, use_extra_head=True, dropout=0.5):
        super().__init__()
        self.base_model = base_model
        self.use_extra_head = use_extra_head

        if self.use_extra_head:
            hidden_dim = original_outputs // 2

            self.intermediate_head = nn.Linear(original_outputs, hidden_dim)
            self.dropout = nn.Dropout(p=dropout)  # <-- Added dropout
            self.final_head = nn.Linear(hidden_dim, target_labels)

            # === Freeze entire base model ===
            for param in self.base_model.parameters():
                param.requires_grad = False

            # === Unfreeze last two trainable layers: top_conv and dense ===
            for name, module in self.base_model.named_modules():
                if "top_conv" in name or "dense" in name:
                    for param in module.parameters():
                        param.requires_grad = True
                    print(f"Unfroze layer: {name}")

            # Initialize extra heads
            nn.init.xavier_uniform_(self.intermediate_head.weight)
            nn.init.zeros_(self.intermediate_head.bias)
            nn.init.xavier_uniform_(self.final_head.weight)
            nn.init.zeros_(self.final_head.bias)
        else:
            # Unfreeze all layers in base model if using directly
            for param in self.base_model.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] → [B, H, W, C]
        base_logits = self.base_model(x.to(x.device))  # [B, original_outputs]

        if self.use_extra_head:
            x = self.intermediate_head(base_logits)
            x = torch.relu(x)
            x = self.dropout(x)  # <-- Apply dropout here
            return self.final_head(x)
        else:
            return base_logits