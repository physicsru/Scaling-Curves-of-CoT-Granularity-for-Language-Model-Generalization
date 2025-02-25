import torch
import torch.nn as nn
import math
from torch.nn import functional as F
#from modeling_phi3.py import Phi3RotaryEmbedding




class Phi3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin: [batch_size, seq_len, head_dim]
    # position_ids: [batch_size, seq_len]
    
    cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding with gemma->phi3, Gemma->Phi3
class Phi3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=1000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen, rpe=True, rope_theta=10000):
        super().__init__()
        assert d_model % nhead == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(drop)
        self.resid_dropout = nn.Dropout(drop)
        self.register_buffer("bias", torch.tril(torch.ones(maxlen, maxlen)).view(1, 1, maxlen, maxlen))
        self.rpe = rpe
        self.n_head = nhead
        self.n_embd = d_model
        self.head_dim = d_model // nhead
        
        if rpe:
            self.rotary = Phi3RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=maxlen,
                base=rope_theta
            )

    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        # Generate position ids
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Apply RoPE if enabled
        if self.rpe:
            cos, sin = self.rotary(v, position_ids, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size, maxlen, rpe):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.norm = Phi3RMSNorm(d_model)  # Using Phi3's RMSNorm instead of LayerNorm
        self.rpe = rpe
        
        if not rpe:  # Only create positional embeddings if not using RoPE
            pe = torch.zeros(maxlen, d_model).float()
            pe.require_grad = False
            position = torch.arange(0, maxlen).float().unsqueeze(1)
            div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

    def forward(self, x):
        if self.rpe:
            embedding = self.tok_embed(x)
        else:
            embedding = self.tok_embed(x) + self.pe[:, :x.size(1)]
        return self.norm(embedding)

class Block(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen, rpe):
        super().__init__()
        self.ln_1 = Phi3RMSNorm(d_model)  # Using Phi3's RMSNorm
        self.attn = CausalSelfAttention(d_model=d_model, nhead=nhead, drop=drop, maxlen=maxlen, rpe=rpe)
        self.ln_2 = Phi3RMSNorm(d_model)  # Using Phi3's RMSNorm
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(d_model, 4 * d_model),
            c_proj  = nn.Linear(4 * d_model, d_model),
            act     = NewGELU(),
            dropout = nn.Dropout(drop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            embedding = Embedding(d_model=args.dmodel, vocab_size=args.vocab, maxlen=args.maxlen, rpe=args.rpe),
            drop = nn.Dropout(args.drop),
            h = nn.ModuleList([Block(d_model=args.dmodel, nhead=args.head, drop=args.drop, maxlen=args.maxlen, rpe=args.rpe) for _ in range(args.num_layer)]),
            ln_f = Phi3RMSNorm(args.dmodel),  # Using Phi3's RMSNorm
        ))
        self.lm_head = nn.Linear(args.dmodel, args.vocab, bias=True)

    def forward(self, idx):
        b, t = idx.size()
        emb = self.transformer.embedding(idx)
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
        
    def generate(self, idx, start):
        b, t = idx.size()
        tmp_start = start + 0
        while True:
            logits = self.forward(idx)
            idx_new = torch.argmax(logits, dim=2)
            idx[torch.arange(b), tmp_start + 1] = idx_new[torch.arange(b), tmp_start]
            if (torch.sum(idx_new[torch.arange(b), tmp_start] != 2) == 0) or (torch.sum(tmp_start == t - 2) != 0):
                break
            tmp_start[idx_new[torch.arange(b), tmp_start] != 2] += 1
        return idx