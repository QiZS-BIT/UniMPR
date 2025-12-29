from typing import Optional, Union, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from fmoe import FMoE
from fmoe.gates import NoisyGate
from fmoe.transformer import _Expert


class CustomNoisyGate(NoisyGate):
    def __init__(self, d_model, num_expert, world_size=1, top_k=2):
        super().__init__(d_model, num_expert, world_size, top_k)

    def forward(self, inp):
        clean_logits = inp @ self.w_gate
        if self.training:
            raw_noise_stddev = inp @ self.w_noise
            noise_stddev = (
                self.softplus(raw_noise_stddev) + self.noise_epsilon
            ) * self.training
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.top_k < self.tot_expert and self.training:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        self.set_loss(loss)

        return (
            top_k_indices.contiguous(),
            top_k_gates.contiguous(),
        )


class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
            self,
            num_expert=32,
            d_model=1024,
            d_hidden=4096,
            top_k=2,
            activation=torch.nn.GELU(),
            gate=CustomNoisyGate,
            expert_dp_comm="none",
            expert_rank=0,
            **kwargs
    ):
        def one_expert(d_model):
            return _Expert(1, d_model, d_hidden, activation, rank=0)

        expert = one_expert
        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, gate=gate, top_k=top_k, **kwargs)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)


class MoETransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, num_expert: int = 8, top_k: int = 2,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MoETransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)

        self.moe = FMoETransformerMLP(num_expert=num_expert, d_model=d_model, d_hidden=dim_feedforward, top_k=top_k)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def __setstate__(self, state):
        super(MoETransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.moe(x)
        return self.dropout2(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
