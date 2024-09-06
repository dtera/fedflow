# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import socket
import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from peft import PeftModelForCausalLM, PeftConfig
from prettytable import PrettyTable
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

from fedflow.util import CommProfiler


class FedPeftModelForCausalLM(PeftModelForCausalLM):
    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__(model, peft_config, adapter_name)


class FedPreTrainedModelForCustomer(PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: PretrainedConfig, tokenizer):
        super().__init__(config)
        self.padding_idx = config.pad_token_id

        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        from fedflow.util import PLMStack

        if config.layers_to_transform is None:
            num_layers = config.num_hidden_layers
        else:
            num_layers = len(config.layers_to_transform)

        self.lora_M_stack = PLMStack(num_layers, config.rcd, config.rdc)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.new_gens = []
        self.tokenizer = tokenizer
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            s: socket.socket = None,
            comm_profiler: CommProfiler = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        if s is None:
            print("no socket provided to s")
            return

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inputs_embeds = self.embed_tokens(input_ids)
        send_tensor(s, inputs_embeds, profiler=comm_profiler)  # initial communication

        self.lora_M_stack(s, profiler=comm_profiler)

        hidden_states = recv_tensor(s, buffer_size=int(4096e5))  # final comm
        if self.prefill_end is None:
            # time it for throughput analysis
            self.prefill_end = time.time()
        s.sendall("hi".encode())
        past_kv_shape = recv_tensor(s, buffer_size=1024)
        # hidden_states = outputs[0] # og
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        # stream output
        if self.tokenizer is not None:
            token_id = torch.argmax(logits, -1).item()
            old_full = self.tokenizer.decode(self.new_gens)
            self.new_gens.append(token_id)
            new_full = self.tokenizer.decode(self.new_gens)
            print(new_full[len(old_full):], end="", flush=True)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        bs, h, seq, hi = past_kv_shape.tolist()
        past_key_values = [[torch.zeros((int(bs), int(h), int(seq + 1), int(hi)))]]
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,  # mock on edge side
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        s = kwargs.get("s", None)
        comm_profiler = kwargs.get("comm_profiler", None)
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "s": s,
                "comm_profiler": comm_profiler,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def my_generate(self, *args, **kwargs):
        """simple wrapper to tell cloud when a new query comes in"""
        s = kwargs.get("s", None)
        profiler = kwargs.get("comm_profiler", None)
        speed_profile = kwargs.pop("speed_profile", False)
        if s is None:
            print("no socket")
            return
        s.sendall("new".encode())
        s.recv(1024)
        with torch.no_grad():
            if speed_profile:
                self.prefill_end = None
                self.prefill_start = time.time()
            self.new_gens = []
            outs = self.generate(*args, **kwargs)
            if speed_profile:
                self.decode_end = time.time()
        s.sendall("finish".encode())

        if profiler is not None:
            profiler.get_report()

        if speed_profile:
            self.decode_time = self.decode_end - self.prefill_end
            self.prefill_time = self.prefill_end - self.prefill_start
            self.decode_tokens = outs.shape[1] - kwargs.get("input_ids").shape[1]
            self.prefill_tokens = kwargs.get("input_ids").shape[1]
            print("Throughput Stats")

            table = PrettyTable()

            # 设置列名
            table.field_names = ["Stages", "Tokens", "Time", "TPS"]

            table.align["Stages"] = "l"  # "l" 对应左对齐
            table.align["Tokens"] = "r"  # "r" 对应右对齐
            table.align["Time"] = "r"  # "r" 对应右对齐
            table.align["TPS"] = "r"  # "r" 对应右对齐
            table.add_row(
                [
                    "prefill",
                    self.prefill_tokens,
                    round(self.prefill_time, 2),
                    round(self.prefill_tokens / self.prefill_time, 2),
                ]
            )
            table.add_row(
                [
                    "decode",
                    self.decode_tokens,
                    round(self.decode_time, 2),
                    round(self.decode_tokens / self.decode_time, 2),
                ]
            )
            print(table)

        return outs

    def print_param_count(self):
        m_cnt = 0
        lmh = 0
        emb = 0
        for n, p in self.named_parameters():
            if "lora" in n:
                m_cnt += p.numel()
            elif "lm_head" in n:
                lmh += p.numel()
            elif "emb" in n:
                emb += p.numel()
        total = m_cnt + lmh + emb

        table = PrettyTable()

        # 设置列名
        table.field_names = ["Modules", "Param #", "Param %"]

        # 设置每列的对齐方式
        table.align["Modules"] = "l"  # "l" 对应左对齐
        table.align["Param #"] = "r"  # "r" 对应右对齐
        table.align["Param %"] = "r"  # "r" 对应右对齐
        table.add_row(["word emb", emb, round(emb / total * 100, 2)])
        table.add_row(["SAPLoRA M", m_cnt, round(m_cnt / total * 100, 2)])
        table.add_row(["lm head", lmh, round(lmh / total * 100, 2)])
        print("Param statistics")
        print(table)
