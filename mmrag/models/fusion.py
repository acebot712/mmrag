import torch
from torch import nn
from typing import Optional

class CrossModalFusionBlock(nn.Module):
    """
    Fuses image, text, and retrieved document embeddings using configurable fusion:
    - 'attention': multi-head attention (default)
    - 'gated': gated fusion
    - 'transformer': transformer encoder fusion
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, fusion_type: str = 'attention'):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'attention':
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            self.dropout = nn.Dropout(dropout)
        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.Sigmoid()
            )
            self.ff = nn.Linear(embed_dim * 3, embed_dim)
        elif fusion_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.norm = nn.LayerNorm(embed_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(
        self,
        image_emb: torch.Tensor,  # (B, 1, D)
        text_emb: torch.Tensor,   # (B, T, D)
        doc_emb: torch.Tensor     # (B, K, D)
    ) -> torch.Tensor:
        """
        Args:
            image_emb: (batch, 1, embed_dim)
            text_emb: (batch, seq_len, embed_dim)
            doc_emb: (batch, num_docs, embed_dim)
        Returns:
            torch.Tensor: fused embedding (batch, 1, embed_dim)
        """
        if self.fusion_type == 'attention':
            x = torch.cat([image_emb, text_emb, doc_emb], dim=1)  # (B, 1+T+K, D)
            query = image_emb  # (B, 1, D)
            attn_out, _ = self.attn(query, x, x)  # (B, 1, D)
            out = self.norm(attn_out + query)
            out = out + self.dropout(self.ff(out))
            return out  # (B, 1, D)
        elif self.fusion_type == 'gated':
            # Flatten all modalities and apply gating
            x = torch.cat([image_emb, text_emb.mean(dim=1, keepdim=True), doc_emb.mean(dim=1, keepdim=True)], dim=2)  # (B, 1, D*3)
            gate = self.gate(x)
            fused = self.ff(x) * gate
            return fused  # (B, 1, D)
        elif self.fusion_type == 'transformer':
            x = torch.cat([image_emb, text_emb, doc_emb], dim=1)  # (B, 1+T+K, D)
            out = self.transformer(x)
            out = self.norm(out[:, :1, :])  # Take [image] token
            return out  # (B, 1, D) 