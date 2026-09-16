"""PyTorch objective adapters, kept out of `potentials.py` so that module stays NumPy-only.

Exposes `transformer_setting(depth, ...)`, which trains a small decoder-only transformer on a
modular-addition task and presents it through the same `setting` dict contract every benchmark in
`potentials.py` uses, so `simulations.Simulation` subclasses can optimize neural-network weights
with no changes at all: `f_vec: (n, d) -> (n,)` maps n flattened parameter vectors to n losses,
and `f_grad_vec: (n, d) -> (n, d)` maps them to n flattened gradients, where d is the model's
total parameter count.

Why modular addition. It is the standard controlled testbed for small transformers, introduced by
Power et al. (arXiv:2201.02177) and used in the canonical one-layer mod-addition setup of Nanda et
al. (arXiv:2301.05217); more to the point here, it is already established as an instrument for
COMPARING OPTIMIZERS on transformers (arXiv:2504.16041), and its depth axis has been swept within
the same task family (arXiv:2603.25009). Transformer training is a genuinely hard non-convex
problem rather than a toy stand-in for one (arXiv:2310.01082), and small transformers are a
legitimate proxy for difficulties that appear at scale (arXiv:2309.14322). Practically, it needs
no data download, is exactly reproducible, and is small enough to take FULL-BATCH gradients -- so
the comparison keeps the exact-gradient setting every other benchmark in this paper uses, instead
of introducing minibatch noise as a new confound.

Cost note. A Softmin step costs one f and one grad-f evaluation PER PARTICLE, so every step here
is n forward+backward passes. They are batched with torch.func.vmap into a single call rather than
looped, which is what makes n=20 particles tractable; a Python loop over particles would be
roughly n times slower.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


class _Block(nn.Module):
    """Pre-LN transformer block, hand-rolled rather than nn.TransformerEncoderLayer so the
    parameter names and shapes stay simple and stable for flatten/unflatten."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (B, T, self.n_heads, C // self.n_heads)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        # Attention written out by hand rather than via F.scaled_dot_product_attention: the fused
        # kernel has no vmap batching rule (torch warns and falls back to a per-sample loop, which
        # would silently undo the whole point of batching the n particles). At T=3 the explicit
        # form costs nothing anyway.
        att = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~causal, float('-inf')).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyTransformer(nn.Module):
    """Decoder-only transformer over the sequence [a, b, EQ], predicting (a + b) mod p from the
    final position. Depth is the only architectural dimension varied in the experiment; d_model
    and n_heads are held fixed across the sweep so depth is the sole variable."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, depth: int, seq_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(seq_len, d_model))
        self.blocks = nn.ModuleList([_Block(d_model, n_heads) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x)[:, -1, :])


def modular_addition_data(p: int, device):
    """Every (a, b) pair in Z_p x Z_p as the sequence [a, b, EQ], with label (a + b) mod p.
    The EQ token is index p, so the vocabulary size is p + 1."""
    a = torch.arange(p, device=device).repeat_interleave(p)
    b = torch.arange(p, device=device).repeat(p)
    eq = torch.full_like(a, p)
    X = torch.stack([a, b, eq], dim=1)
    Y = (a + b) % p
    return X, Y


def transformer_setting(
    depth: int,
    p: int = 53,
    d_model: int = 64,
    n_heads: int = 2,
    train_frac: float = 0.8,
    target_loss: float = 0.10,
    data_seed: int = 0,
    device=None,
):
    """Build a `setting` dict for training a `TinyTransformer` of the given depth.

    Returns the usual keys (`f_vec`, `f_grad_vec`, `particle_init`, `bounds`, `set`) plus:
      `d`             -- flattened parameter count, i.e. the optimization dimension,
      `particle_init_fn(n, rng) -> (n, d)` -- n independent standard PyTorch inits, the analogue
                         of `initialize_lj_cluster`; the uniform-box `particle_init` below is only
                         a fallback for `Simulation.__init__`'s own draw and is always overridden
                         by the driver via `reset(particle_init=x0)`,
      `target_loss`   -- value-space target for `first_value_time`,
      `chance_loss`   -- log(p), the loss of a uniform predictor, for reference in the paper.

    `set` is None: there is no meaningful coordinate-space target box in weight space, so entry
    time is measured against the loss value, exactly as for Lennard-Jones.

    `device` defaults to CUDA when it is available, and only falls back to CPU when it is not.
    This used to default to CPU unconditionally, which was a silent trap: a caller that forgot to
    pass `device` ran the whole benchmark on one CPU thread even inside a job holding an A40, and
    nothing in the output said so -- the job's own `torch.cuda.is_available()` assertion passed
    while every forward and backward pass ran on the CPU. It cost a GPU allocation before it was
    noticed (the tell was a 50-layer step time on an A40 matching a laptop's). Pass `device`
    explicitly to override; the returned dict reports the choice under `device`.
    """
    device = (torch.device(device) if device is not None
              else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    seq_len = 3
    vocab = p + 1

    X, Y = modular_addition_data(p, device)
    if train_frac < 1.0:
        g = torch.Generator(device='cpu').manual_seed(data_seed)
        perm = torch.randperm(X.shape[0], generator=g).to(device)
        keep = perm[: int(train_frac * X.shape[0])]
        X, Y = X[keep], Y[keep]

    template = TinyTransformer(vocab, d_model, n_heads, depth, seq_len).to(device)
    names, shapes, sizes = [], [], []
    for name, param in template.named_parameters():
        names.append(name)
        shapes.append(tuple(param.shape))
        sizes.append(param.numel())
    total = int(sum(sizes))

    def unflatten(flat):
        out, i = {}, 0
        for name, shape, size in zip(names, shapes, sizes):
            out[name] = flat[i:i + size].view(shape)
            i += size
        return out

    def loss_of(flat, X_, Y_):
        logits = functional_call(template, unflatten(flat), (X_,))
        return F.cross_entropy(logits, Y_)

    # vmap over the particle axis only; the data is shared across particles (in_dims None).
    batched_loss = vmap(loss_of, in_dims=(0, None, None))
    batched_grad = vmap(grad(loss_of), in_dims=(0, None, None))

    def _to_torch(positions):
        return torch.as_tensor(np.ascontiguousarray(positions), dtype=torch.float32, device=device)

    def f_vec(positions):
        with torch.no_grad():
            out = batched_loss(_to_torch(positions), X, Y)
        return out.detach().cpu().numpy().astype(np.float64)

    def f_grad_vec(positions):
        out = batched_grad(_to_torch(positions), X, Y)
        return out.detach().cpu().numpy().astype(np.float64)

    def particle_init_fn(n, rng=None):
        """n independent models under PyTorch's own default initialization, flattened. Seeded from
        `rng` (a numpy Generator or RandomState) when given, so the driver's per-seed NumPy stream
        controls the torch init too and a run stays reproducible from its seed alone."""
        vectors = []
        for _ in range(n):
            if rng is not None:
                draw = (rng.integers(0, 2 ** 31 - 1) if hasattr(rng, 'integers')
                        else rng.randint(0, 2 ** 31 - 1))
                torch.manual_seed(int(draw))
            model = TinyTransformer(vocab, d_model, n_heads, depth, seq_len).to(device)
            with torch.no_grad():
                flat = torch.cat([q.reshape(-1) for q in model.parameters()])
            vectors.append(flat.cpu().numpy())
        return np.stack(vectors).astype(np.float64)

    return {
        'f_vec': f_vec,
        'f_grad_vec': f_grad_vec,
        # Weight-space box. Wide enough never to bind in practice (PyTorch inits sit well inside
        # it and the optimizers used here do not drift anywhere near it), but finite because
        # Simulation.__init__ requires finite bounds to build its vectorization probe.
        'bounds': [[-10.0, 10.0]] * total,
        'particle_init': [[-0.5, 0.5]] * total,
        'set': None,
        'd': total,
        'particle_init_fn': particle_init_fn,
        'target_loss': target_loss,
        'chance_loss': float(np.log(p)),
        'depth': depth,
        'p': p,
        'n_train': int(X.shape[0]),
        # Reported so a caller can print where it is actually running; see the device note above.
        'device': str(device),
    }
