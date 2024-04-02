# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py
# Copyright 2024 The JetMoE team.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only JetMoE2MoE model compatible with HuggingFace weights."""
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.fused_moe import is_hip, get_moe_configs, moe_align_block_size, invoke_fused_moe_kernel
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput


class top_k_gating(nn.Module):
    def __init__(
        self,
        input_size, 
        num_experts, 
        top_k,
    ):
        """
        Initialize the top-k gating mechanism.

        Args:
            input_size (int): Size of the input.
            num_experts (int): Number of experts.
            top_k (int): Number of top experts to select.
            acc_aux_loss (bool): Whether to accumulate auxiliary loss statistics.
            dropout (float): Dropout rate for gating network.
            hidden_size (int): Hidden size of the gating network.
            sample_topk (int): Number of top-k experts to sample during training.
            aux_loss (str): Type of auxiliary loss ('mi' or 'switch').
            gate_type (str): Type of gating mechanism ('mlp', 'linear', or 'gmm').
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        assert top_k <= num_experts
        self.top_k = top_k

        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def extra_repr(self):
        """
        Return extra representation string for the module.
        """
        return 'k={}, num_experts={}'.format(
            self.top_k, self.num_experts)

    def forward(self, x):
        """
        Compute the top-k gating for the input.

        See paper: https://arxiv.org/abs/1701.06538.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_size].
            skip_mask (torch.Tensor): Skip mask tensor (binary) with the same shape as `x`.
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float

        Returns:
            torch.Tensor: Top-k indices.
            torch.Tensor: Top-k gating values.
            torch.Tensor: Probability values for each expert.
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        logits = self.layer(x)
        return logits
    

class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        """
        Initialize the ParallelExperts module.

        Args:
            num_experts (int): Number of experts.
            input_size (int): Size of the input.
            output_size (int): Size of the output.
            bias (bool): Whether to include bias terms.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        self.reset_parameters()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self):
        return 'num_experts={}, input_size={}, output_size={}'.format(
            self.num_experts, self.input_size, self.output_size)

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the model.
        """
        nn.init.uniform_(self.weight, -1. / self.weight.size(1), 1. / self.weight.size(1))

    def forward(self, inputs, expert_size):
        """
        Forward pass of the ParallelExperts module.

        Args:
            inputs (Tensor): Input tensor.
            expert_size: Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            output_list.append(F.linear(input_list[i], self.weight[i]))
        results = torch.cat(output_list, dim=0)
        return results


class MoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    

    Args:
        input_size: integer - size of the input
        head_size: integer - size of the expert's hidden layer
        num_experts: an integer - number of experts
        top_k: an integer - how many experts to use for each batch element
        bias: a boolean - whether to include bias in linear layers
        activation: an activation function to apply to expert's outputs
    """

    def __init__(
        self, 
        input_size, 
        hidden_size, 
        num_experts, 
        top_k,
        bias=True, 
        glu=True,
        ):
        super(MoE, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.glu = glu
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(input_size))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None
        self.output_linear = ParallelExperts(num_experts, hidden_size, input_size)
        self.top_k = min(top_k, self.num_experts)

        self.router = top_k_gating(
            input_size=input_size, 
            num_experts=num_experts, 
            top_k=top_k, 
            )

    def extra_repr(self):
        return 'k={}, e={}'.format(
            self.top_k, self.num_experts)

    def forward(self, x):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        logits = self.router(x)
        y = fused_moe(
            x, self.input_linear.weight, self.output_linear.weight,
            logits, self.top_k, renormalize=True
        )

        if self.bias is not None:
            y = y + self.bias
        return y
    
    def fused_moe_map(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        inplace: bool = False,
        override_config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        This function computes a Mixture of Experts (MoE) layer using two sets of
        weights, w1 and w2, and top-k gating mechanism.

        Parameters:
        - hidden_states (torch.Tensor): The input tensor to the MoE layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - gating_output (torch.Tensor): The output of the gating operation
            (before softmax).
        - topk (int): The number of top-k experts to select.
        - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
        - inplace (bool): If True, perform the operation in-place.
            Defaults to False.
        - override_config (Optional[Dict[str, Any]]): Optional override
            for the kernel configuration.

        Returns:
        - torch.Tensor: The output tensor after applying the MoE layer.
        """
        # Check constraints.
        assert hidden_states.shape[0] == gating_output.shape[0], (
            "Number of tokens mismatch")
        assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
        assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert w1.is_contiguous(), "Expert weights1 must be contiguous"
        assert hidden_states.dtype in [
            torch.float32, torch.float16, torch.bfloat16
        ]
        M, _ = hidden_states.shape
        E, N, _ = w1.shape

        if is_hip():
            # The MoE kernels are not yet supported on ROCm.
            routing_weights = torch.softmax(gating_output,
                                            dim=-1,
                                            dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
        else:
            import vllm._moe_C as moe_kernels

            topk_weights = torch.empty(M,
                                    topk,
                                    dtype=torch.float32,
                                    device=hidden_states.device)
            topk_ids = torch.empty(M,
                                topk,
                                dtype=torch.int32,
                                device=hidden_states.device)
            token_expert_indicies = torch.empty(M,
                                                topk,
                                                dtype=torch.int32,
                                                device=hidden_states.device)
            moe_kernels.topk_softmax(
                topk_weights,
                topk_ids,
                token_expert_indicies,
                gating_output.float(),  # TODO(woosuk): Optimize this.
            )
            del token_expert_indicies  # Not used. Will be used in the future.
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        if override_config:
            config = override_config
        else:
            # First try to load optimal config from the file
            configs = get_moe_configs(E, self.output_linear.weight.shape[2])

            if configs:
                # If an optimal configuration map has been found, look up the
                # optimal config
                config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
            else:
                # Else use the default config
                config = {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 64,
                    'BLOCK_SIZE_K': 32,
                    'GROUP_SIZE_M': 8
                }

                if M <= E:
                    config = {
                        'BLOCK_SIZE_M': 16,
                        'BLOCK_SIZE_N': 32,
                        'BLOCK_SIZE_K': 64,
                        'GROUP_SIZE_M': 1
                    }

        intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                        device=hidden_states.device,
                                        dtype=hidden_states.dtype)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config['BLOCK_SIZE_M'], E)

        invoke_fused_moe_kernel(hidden_states, w1, intermediate_cache1,
                                topk_weights, topk_ids, sorted_token_ids,
                                expert_ids, num_tokens_post_padded, False,
                                topk_ids.shape[1], config)
        
        states = (M, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config)
        
        return intermediate_cache1, states
    
    def fused_moe_reduce(
        self,
        hidden_states: torch.Tensor,
        states: tuple,
        inplace: bool = False,
    ) -> torch.Tensor:
        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert self.output_linear.weight.is_contiguous(), "Expert weights2 must be contiguous"
        assert hidden_states.dtype in [
            torch.float32, torch.float16, torch.bfloat16
        ]

        M, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config = states

        intermediate_cache3 = torch.empty((M, topk_ids.shape[1], self.output_linear.weight.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
        invoke_fused_moe_kernel(hidden_states, self.output_linear.weight, intermediate_cache3,
                            topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, True, 1,
                            config)

        if inplace:
            return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                            dim=1,
                            out=hidden_states)
        return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                        dim=1)


    def map(self, x):
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        logits = self.router(x)
        y, states = self.fused_moe_map(
            x, self.input_linear.weight,
            logits, self.top_k, renormalize=True
        )
        return y, states

    def reduce(self, x, states):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.

        Returns:
            Tensor: Reduced output tensor.
        """
        
        num_token, k, emb_size = x.size()
        assert k == self.top_k
        y = self.fused_moe_reduce(
            x.view(num_token * k, emb_size), states
        )
        y = y.view(num_token, self.input_size)
        return y


class JetMoeAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.kv_channels
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.kv_proj = torch.nn.Linear(
            hidden_size, self.kv_size * 2, bias=False
            )

        self.experts = MoE(
            input_size=hidden_size,
            hidden_size=self.q_size,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            glu=False
        )
        self.top_k = config.moe_top_k

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads * self.top_k,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        num_token, _ = hidden_states.size()

        q, states = self.experts.map(hidden_states)
        q = q.view(num_token, self.top_k, self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2).reshape(num_token, self.num_kv_heads * self.top_k * self.head_dim)
        k, v = self.kv_proj(hidden_states).chunk(2, dim=-1)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        
        attn_output = attn_output.reshape(num_token, self.num_kv_heads, self.top_k, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(num_token, self.top_k, self.kv_size)
        output = self.experts.reduce(attn_output, states)
        return output


class JetMoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attention = JetMoeAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
        )

        self.mlp = MoE(
            input_size=config.hidden_size,
            hidden_size=config.ffn_hidden_size,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            glu=config.glu
        )
            
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class JetMoeModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            JetMoeDecoderLayer(config,
                                 layer_idx,
                                 linear_method=linear_method)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i], attn_metadata,
                                            residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class JetMoeForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = JetMoeModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=False):
            if "rotary_emb.inv_freq" in name:
                continue
            
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            # Skip experts that are not assigned to this worker.
            if (("mlp.experts." in name or "mlp.shared_expert." in name)
                    and name not in params_dict):
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            if name == "model.embed_tokens.weight":
                param = params_dict["lm_head.weight"]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
