# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Rollout with huggingface models.
Patched to emit per-step prefill/decode timing via _PrefillDecodeTimer LogitsProcessor.

Log line format (rank-0 only, once per generate_sequences call):
  [hf_rollout] prefill_ms:XX.X decode_ms:XX.X decode_steps:N prompt_toks:N resp_toks:N
"""

import contextlib
import time

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, LogitsProcessor

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


class _PrefillDecodeTimer(LogitsProcessor):
    """
    Measures time for each generation step inside model.generate().

    Lifecycle:
      - Call reset(t0) right before model.generate() starts.
      - __call__ fires after each forward pass (prefill on step 0, decode on steps 1+).
      - After generate() returns, read prefill_ms / decode_ms / decode_steps.

    Works on any PyTorch backend (CUDA, ROCm, Ascend, MLU …) because:
      - No explicit cuda.synchronize() — logits are already materialized when the
        processor is called, so the GPU has implicitly synced.
      - Only uses time.perf_counter() (wall-clock).
    """

    def __init__(self):
        self.prefill_ms: float = 0.0
        self.decode_ms: float = 0.0
        self.decode_steps: int = 0
        self._t0: float = 0.0
        self._call_count: int = 0

    def reset(self, t0: float) -> None:
        self.prefill_ms = 0.0
        self.decode_ms = 0.0
        self.decode_steps = 0
        self._t0 = t0
        self._call_count = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        now = time.perf_counter()
        elapsed_ms = (now - self._t0) * 1000.0
        if self._call_count == 0:
            self.prefill_ms += elapsed_ms   # first forward = prefill
        else:
            self.decode_ms += elapsed_ms    # subsequent = decode
            self.decode_steps += 1
        self._t0 = now
        self._call_count += 1
        return scores


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        # Skip BaseRollout.__init__: in veRL 0.8.0 it requires (config, model_config, device_mesh)
        # which are server-based args not applicable to standalone HF generation.
        self.config = config
        self.module = module

    # ── stubs for abstract methods required by BaseRollout ────────────────────
    async def release(self): pass
    async def resume(self, tags=None): pass
    async def update_weights(self, weights=None, **kwargs): pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)

        # accumulate timing across mini-batches
        total_prefill_ms = 0.0
        total_decode_ms = 0.0
        total_decode_steps = 0
        total_prompt_toks = 0
        total_resp_toks = 0

        outputs = []
        for chunk in batch_prompts:
            out, td = self._generate_minibatch(chunk)
            outputs.append(out)
            total_prefill_ms   += td["prefill_ms"]
            total_decode_ms    += td["decode_ms"]
            total_decode_steps += td["decode_steps"]
            total_prompt_toks  += td["prompt_toks"]
            total_resp_toks    += td["resp_toks"]

        # emit on rank-0 only to avoid duplicate log lines
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0 and total_prefill_ms > 0:
            print(
                f"[hf_rollout] "
                f"prefill_ms:{total_prefill_ms:.1f} "
                f"decode_ms:{total_decode_ms:.1f} "
                f"decode_steps:{total_decode_steps} "
                f"prompt_toks:{total_prompt_toks} "
                f"resp_toks:{total_resp_toks}",
                flush=True,
            )

        return DataProto.concat(outputs)

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto):
        """Returns (DataProto output, timing_dict)."""
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))

        if not do_sample:
            kwargs = {"do_sample": False, "num_beams": 1}
        elif is_validate:
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,
            }
        else:
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": 1,
            }

        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]           # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        timer = _PrefillDecodeTimer()
        timer.reset(t0=time.perf_counter())

        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=False,
                return_dict_in_generate=True,
                use_cache=True,
                logits_processor=[timer],
            )

        seq = output.sequences
        generated_batch_size = seq.size(0)

        # token counts for throughput calculation
        actual_resp_toks = int((seq[:, prompt_length:] != pad_token_id).sum().item())
        timing_dict = {
            "prefill_ms":   timer.prefill_ms,
            "decode_ms":    timer.decode_ms,
            "decode_steps": timer.decode_steps,
            "prompt_toks":  int(attention_mask.sum().item()),
            "resp_toks":    actual_resp_toks,
        }

        # pad to fixed response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]
        if delta_length > 0:
            delta_tokens = torch.full(
                (generated_batch_size, delta_length), pad_token_id,
                device=seq.device, dtype=seq.dtype,
            )
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:]

        response_length_actual = response.size(1)
        delta_position_id = torch.arange(1, response_length_actual + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        get_torch_device().empty_cache()
        self.module.train()
        return DataProto(batch=batch), timing_dict
