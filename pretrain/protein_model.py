import torch
from torch import nn
import torch
import torch.nn.functional as F
from torch import tensor, abs as torch_abs, logical_not, log, clamp

def masked_adt_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = torch.abs(target-input) * mask
    return loss.sum() / mask.sum()

 

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def quantile_loss(
    pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask=mask.float()
    # q = tensor([0.1, 0.25, 0.75, 0.9], device = pred.device)
    q = tensor([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95], device = pred.device)
    bias = pred - truth[:, :, None]
    
    I_over = bias.detach() > 0.
    q_weight = I_over * (1 - q) + logical_not(I_over) * q
    
    q_loss = torch_abs(bias) * q_weight
    q_loss = q_loss.sum(axis = 2) * mask
    return q_loss.sum()/ mask.sum()

        


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()

class MultiheadAttentionBlock(nn.Module): 
    def __init__(self, dim, heads, dropout): 
        super().__init__() 
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, key_padding_mask):
        residual = x 
        if context is None:  
            attn_output, attn_weights = self.attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
            )
        else:  
            attn_output, attn_weights = self.attn(
                query=x,
                key=context,
                value=context,
                key_padding_mask=key_padding_mask,
            )
        x = self.norm(residual + self.dropout(attn_output)) 
        return x, attn_weights

class FeedForward(nn.Module): 
    def __init__(self, dim, ff_mult=4, dropout=0.1): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        ) 
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x 
        x = self.net(x) 
        return self.norm(residual + x)

class PerformerLM_ADT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,                         
        max_seq_len,                        
        dim,                                
        depth,                              
        heads,                              
        dim_head=64,                        
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False,
        ff_chunks=1,
        ff_glu=False,
        emb_dropout=0.,
        ff_dropout=0.1,
        attn_dropout=0.1,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        cross_attend=False,
        no_projection=False,
        tie_embed=False,                  
        g2v_position_emb=True,            
        auto_check_redraw=True,
        qkv_bias=False
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = torch.zeros_like
        self.layer_pos_emb = Always(None)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiheadAttentionBlock(dim=dim, heads=heads, dropout=attn_dropout),
                FeedForward(dim, ff_mult, ff_dropout)
            ]))
        self.dropout = nn.Dropout(emb_dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_out1 = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ) if not tie_embed else None
        self.to_out2 = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ) if not tie_embed else None

    def forward(self, x, other_embedding=None, attn_mask=None, output_attentions=False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'
        x = self.token_emb(x)
        if output_attentions:
            x.requires_grad_()
        x += self.pos_emb(x)
        x = self.dropout(x)
        all_attentions = []
        for attn_block, ff_block in self.layers:
            x, attn_weights = attn_block(
                x,
                context=other_embedding,
                key_padding_mask=attn_mask,
            )
            if output_attentions:
                all_attentions.append(attn_weights.detach())
            x = ff_block(x)
        x = self.norm(x)
        if output_attentions:
            return x, self.to_out1(x), self.to_out2(x), all_attentions
        else:
            return x, self.to_out1(x), self.to_out2(x), None

class BLIP_Pretrain(nn.Module):
    def __init__(self,                 
                 num_tokens=1,
                 num_tokens2=1,
                 rna_max_seq_len=1,
                 adt_max_seq_len=1,
                 g2v_position_emb=True,
                 embed_dim=512,
                 ):
        super().__init__()
        self.adt_model = PerformerLM_ADT(
            num_tokens=num_tokens2,
            dim=embed_dim,
            depth=6,
            max_seq_len=adt_max_seq_len,
            heads=8,
            local_attn_heads=0,
            g2v_position_emb=g2v_position_emb,
            causal=False,
            cross_attend=True
        )

    def forward(self, adt_data, rna_embedding, rna_mask, adt_values, output_atten=False):
        adt_mask = (adt_values != 0)
        adt_embeddings, adt_to_out, adt_to_out_quantiles, adt_gene_atten = self.adt_model(
            adt_data, other_embedding=rna_embedding, attn_mask=rna_mask, output_attentions=output_atten
        )
        return adt_embeddings, adt_to_out, adt_to_out_quantiles, adt_gene_atten, adt_values, adt_mask