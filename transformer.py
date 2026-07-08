import torch.nn as nn
from attention import MultiHeadSelfAttention

class FeedForward(nn.Module):
    def __init__(self, embed_dim=512, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(embed_dim,ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim,embed_dim)
        )
    def forward(self,x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attn=MultiHeadSelfAttention(embed_dim,num_heads)
        self.ln1=nn.LayerNorm(embed_dim)
        self.ff=FeedForward(embed_dim,ff_dim,dropout)
        self.ln2=nn.LayerNorm(embed_dim)
        self.drop=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        x=self.ln1(x+self.drop(self.attn(x,mask)))
        x=self.ln2(x+self.drop(self.ff(x)))
        return x
