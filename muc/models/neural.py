"""Windowed gradient-trained baselines: small causal Transformer and LSTM.

Both see the last `window` raw tokens (distractors included) behind a BOS token and
predict the next token. Two modes:

online     weights are updated by one SGD/Adam step per event (loss on the last
           prediction, or on all positions of the window: `loss_positions`)
incontext  weights are pretrained offline on streams from *other random automata of the
           same family* (seeds disjoint from tuning/eval) and then frozen; any adaptation
           must happen through the context window. This gives the model prior knowledge
           of the task family that the other models do not get. It is reported as a
           separate model with its pretraining tokens counted as experience.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SeqModel, bits_per_token
from ..tasks import TaskConfig, generate_stream

torch.set_num_threads(1)

PRETRAIN_SEED_BASE = 3000
_PRETRAIN_CACHE: dict = {}


class TinyTransformer(nn.Module):
    def __init__(self, vocab, d_model=16, n_heads=2, n_layers=1, ff_mult=2, window=16):
        super().__init__()
        self.tok = nn.Embedding(vocab + 1, d_model)  # +1: BOS
        self.pos = nn.Embedding(window + 1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * ff_mult, dropout=0.0,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        L = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(self.ln(h))


class TinyLSTM(nn.Module):
    def __init__(self, vocab, emb=16, hidden=16, n_layers=1):
        super().__init__()
        self.tok = nn.Embedding(vocab + 1, emb)
        self.rnn = nn.LSTM(emb, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        h, _ = self.rnn(self.tok(x))
        return self.head(h)


def _n_params(net):
    return sum(p.numel() for p in net.parameters())


class WindowedNeural(SeqModel):
    def __init__(self, vocab, net, window, lr, optimizer="sgd", momentum=0.0,
                 loss_positions="last", online=True, flops_forward=0, seed=0, label="neural"):
        self.V = vocab
        self.BOS = vocab
        self.net = net
        self.window = int(window)
        self.loss_positions = loss_positions
        self.learning = bool(online)
        self.name = label
        self.buf: list = []
        self._cache = None
        self.flops_forward = int(flops_forward)
        self.make_opt(optimizer, lr, momentum)

    def make_opt(self, optimizer, lr, momentum):
        if optimizer == "adam":
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        else:
            self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)

    def reset_context(self):
        self.buf = []
        self._cache = None

    def clear_cache(self):
        self._cache = None

    def _inp(self):
        return torch.tensor([[self.BOS] + self.buf[-self.window:]], dtype=torch.long)

    def predict(self):
        inp = self._inp()
        if self.learning:
            logits = self.net(inp)
        else:
            with torch.no_grad():
                logits = self.net(inp)
        self._cache = (inp, logits)
        return torch.softmax(logits[0, -1].detach().double(), -1).numpy()

    def update(self, token):
        token = int(token)
        if self.learning:
            if self._cache is None:
                self.predict()
            inp, logits = self._cache
            if self.loss_positions == "all":
                tgt = torch.cat([inp[0, 1:], torch.tensor([token])])
                loss = F.cross_entropy(logits[0], tgt)
            else:
                loss = F.cross_entropy(logits[0, -1:], torch.tensor([token]))
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
        self._cache = None
        self.buf.append(token)
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def budget(self):
        n = _n_params(self.net)
        mult = 3 if self.learning else 1  # forward + ~2x for backward
        return {
            "param_bits": int(32 * n),
            "param_bits_used": int(32 * n),
            "n_params": int(n),
            "state_bits": int(self.window * bits_per_token(self.V)),
            "approx_flops_per_event": int(mult * self.flops_forward),
        }


def _transformer_flops(d, L, layers, ff, V):
    per_layer = 2 * (4 * L * d * d) + 2 * (2 * L * L * d) + 2 * (2 * L * d * ff * d)
    return layers * per_layer + 2 * L * d * V


def _lstm_flops(e, h, L, layers, V):
    per_layer = 2 * 4 * h * (e + h) * L
    return layers * per_layer + 2 * L * h * V


def pretrain(model: WindowedNeural, task_cfg: TaskConfig, n_streams=64, events=600,
             steps=600, batch=32, lr=3e-3, seed=PRETRAIN_SEED_BASE):
    """Pretrain on random automata of the family, then freeze (caller sets learning False)."""
    rng = np.random.default_rng(seed)
    plen = max(1, events // len(task_cfg.phases))
    toks = np.stack([generate_stream(task_cfg, PRETRAIN_SEED_BASE + 1 + i, phase_len=plen).tokens
                     for i in range(n_streams)])
    W = model.window
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    T = toks.shape[1]
    for _ in range(steps):
        si = rng.integers(0, n_streams, size=batch)
        st = rng.integers(0, T - W, size=batch)
        seqs = np.stack([toks[i, s:s + W + 1] for i, s in zip(si, st)])
        inp = np.concatenate([np.full((batch, 1), model.BOS), seqs[:, :-1]], axis=1)
        logits = model.net(torch.as_tensor(inp, dtype=torch.long))
        loss = F.cross_entropy(logits.reshape(-1, model.V), torch.as_tensor(seqs.reshape(-1)))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        opt.step()
    model.pretrain_tokens = steps * batch * (W + 1)


def build_neural(kind, vocab, task_cfg: TaskConfig, spec: dict, seed: int):
    """spec keys. Architecture: d_model, n_heads, n_layers, ff_mult / emb, hidden; window.
    Learning: lr, optimizer, momentum, loss_positions. mode: online | incontext.
    Pretraining (incontext only): pre_steps, pre_lr, pre_streams, pre_events, pre_batch."""
    window = int(spec.get("window", 16))
    mode = spec.get("mode", "online")
    # incontext models are pretrained once and shared across eval seeds, so init is fixed
    torch.manual_seed(PRETRAIN_SEED_BASE if mode == "incontext" else seed)
    L = window + 1
    if kind == "transformer":
        d = int(spec.get("d_model", 16))
        arch = dict(d_model=d, n_heads=int(spec.get("n_heads", 2)), n_layers=int(spec.get("n_layers", 1)),
                    ff_mult=int(spec.get("ff_mult", 2)), window=window)
        net = TinyTransformer(vocab, **arch)
        flops = _transformer_flops(d, L, arch["n_layers"], arch["ff_mult"], vocab)
    else:
        e, h, nl = int(spec.get("emb", 16)), int(spec.get("hidden", 16)), int(spec.get("n_layers", 1))
        arch = dict(emb=e, hidden=h, n_layers=nl)
        net = TinyLSTM(vocab, **arch)
        flops = _lstm_flops(e, h, L, nl, vocab)
    label = f"{kind}_{mode}"
    m = WindowedNeural(vocab, net, window, lr=float(spec.get("lr", 0.01)),
                       optimizer=spec.get("optimizer", "sgd"), momentum=float(spec.get("momentum", 0.0)),
                       loss_positions=spec.get("loss_positions", "last"), online=(mode == "online"),
                       flops_forward=flops, seed=seed, label=label)
    if mode == "incontext":
        pre = dict(n_streams=int(spec.get("pre_streams", 64)), events=int(spec.get("pre_events", 600)),
                   steps=int(spec.get("pre_steps", 600)), batch=int(spec.get("pre_batch", 32)),
                   lr=float(spec.get("pre_lr", 3e-3)))
        key = json.dumps([kind, arch, pre, task_cfg.to_dict()], sort_keys=True)
        if key not in _PRETRAIN_CACHE:
            torch.manual_seed(PRETRAIN_SEED_BASE)
            pretrain(m, task_cfg, **pre)
            _PRETRAIN_CACHE[key] = ({k: v.clone() for k, v in net.state_dict().items()}, m.pretrain_tokens)
        sd, ptok = _PRETRAIN_CACHE[key]
        net.load_state_dict(sd)
        m.pretrain_tokens = ptok
        m.learning = False
    return m
