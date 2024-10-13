import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional
from robomimic.models.transformers import PositionalEncoding


class PositionalEncoding2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    Copied from DETR repo; can be heavily optimized

    Args:
        num_pos_feats: hidden dimesion (I think)
        temperature: ???
        normalize: whether to normalize embeddings
        scale: used for normalization
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: float = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super(PositionalEncoding2D, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x must be (bsz, ..., hidden_dim, H, W)
        not_mask = torch.ones(x.shape[-2:])[None].to(x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return x + pos


class LearnablePosEncoding2D(nn.Module):
    """Learnable positional encoding for 2D inputs; inspired by DETR paper"""

    def __init__(
        self,
        hidden_dim: int,
        num_pos_feats: int = 64,
        scale: float = 1.0,
    ):
        assert hidden_dim % 2 == 0, "Hidden dimension must be even"
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.scale = scale
        self.row_embed = nn.Parameter(torch.rand(num_pos_feats, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(num_pos_feats, hidden_dim // 2))

    def forward(self, x):
        H, W = x.shape[-2:]
        col = self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1)
        row = self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
        pos = torch.cat([col, row], dim=-1).flatten(0, 1).unsqueeze(1)
        if x.dim() == 5:
            num_imgs = x.shape[1]
            pos = pos.repeat(num_imgs, 1, 1)
            x = x.transpose(1, 2)
        x = x.flatten(2).permute(2, 0, 1)
        return self.scale * x + pos

class Transformer(nn.Module):
    '''
    Basic transformer implementation using torch.nn. Also added option for custom pos embeddings. 
    Made to be as basic as possible but also flexible to be put into ACT.

        d: hidden dimension
        h: number of heads
        d_ff: feed forward dimension
        num_layers: number of layers for encoder and decoder
        L: sequence length
        dropout: dropout rate
        src_vocab_size: size of source vocabulary
        tgt_vocab_size: size of target vocabulary
        pos_encoding_class : nn.Module class defining custom pos encoding

    '''
    def __init__(
        self,
        d : int,
        h : int,
        d_ff : int,
        num_layers : int,
        dropout : float = 0.1,
        L : Optional[int] = None,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        pos_encoding_class: Optional[Callable[..., nn.Module]] = None,
        **pos_encoding_kwargs: Any  # Additional arguments for the custom encoding class
    ):
        super(Transformer, self).__init__()

        self.d = d
        self.h = h
        self.src_embed = nn.Embedding(src_vocab_size, d) if src_vocab_size else None
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d) if tgt_vocab_size else None

        if pos_encoding_class is not None:
            self.src_pos_encoding = pos_encoding_class(**pos_encoding_kwargs)
            self.tgt_pos_encoding = pos_encoding_class(**pos_encoding_kwargs)
        else:
            self.src_pos_encoding = PositionalEncoding(d)
            self.tgt_pos_encoding = PositionalEncoding(d)
        
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=h, dim_feedforward=d_ff, dropout=dropout, batch_first=True
            )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=h, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        
        self.fc = nn.Linear(d, tgt_vocab_size) if tgt_vocab_size else None
        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self, src, tgt):
        # some implementation of mask generation for masked attn.
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, src_len)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3) # (bsz, 1, tgt_len, 1)

        L = tgt.size(1)
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(tgt.device)
        tgt_mask = tgt_mask & ~mask.unsqueeze(0)
        return src_mask, tgt_mask

    def forward(self, src, tgt, auto_masks=False):
        if auto_masks:
            src_mask, tgt_mask = self.generate_mask(src, tgt)
        else:
            src_mask = tgt_mask = None

        if self.src_embed:
            src = self.src_embed(src)
        if self.tgt_embed:
            tgt = self.tgt_embed(tgt)

        src = self.src_pos_encoding(src)
        tgt = self.tgt_pos_encoding(tgt)

        src = self.dropout(src)
        tgt = self.dropout(tgt)

        enc = self.encoder(src, src_key_padding_mask=src_mask)
        dec = self.decoder(
            tgt, enc, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask
        )

        if self.fc:
            dec = self.fc(dec)

        return dec

class StyleEncoder(nn.Module):
    def __init__(
        self,
        act_len: int,
        act_dim: int,
        hidden_dim: int,
        latent_dim: int,
        h: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(StyleEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.act_len = act_len
        self.hidden_dim = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=h, dim_feedforward=d_ff, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
        self.cls_embedding = nn.Parameter(torch.rand(1, hidden_dim))
        self.action_projection = nn.Linear(act_dim, hidden_dim)
        self.qpos_projection = nn.Linear(act_dim, hidden_dim)
        self.latent_projection = nn.Linear(hidden_dim, latent_dim * 2)

        self.pos_encoding = PositionalEncoding(hidden_dim, act_len + 2)
    
    def forward(self, qpos, actions):
        bsz = qpos.shape[0]

        qpos = self.qpos_projection(qpos).unsqueeze(1)  # [bsz, 1, hidden_dim]
        actions = self.action_projection(actions)  # [bsz, act_len, hidden_dim]

        cls = self.cls_embedding.unsqueeze(0).expand(bsz, -1, -1)  # [bsz, 1, hidden_dim]

        x = torch.cat([cls, qpos, actions], dim=1)  # [bsz, act_len + 2, hidden_dim]
        assert x.shape == (bsz, self.act_len + 2, self.hidden_dim)

        x = self.pos_encoding(x.transpose(0, 1))  # [act_len + 2, bsz, hidden_dim]

        x = self.encoder(x)  # [act_len + 2, bsz, hidden_dim]

        x = x[0]  # [bsz, hidden_dim]

        x = self.latent_projection(x)  # [bsz, latent_dim * 2]
        mu, logstd = x.chunk(2, dim=-1)  # [bsz, latent_dim] each
        dist = Normal(mu, logstd.exp())  # Create Normal distribution

        return dist

