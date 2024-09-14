# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from peft import PeftModelForCausalLM, PeftConfig
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

from fedflow.register import recv_queue, send_queue
from fedflow.util.config_utils import FedPretrainedConfig


class FedPeftModelForCausalLM(PeftModelForCausalLM):
    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__(model, peft_config, adapter_name)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_ids=None,
            **kwargs,
    ):
        inputs_embeds = recv_queue.get()
        outputs = super().forward(input_ids, attention_mask, inputs_embeds, labels, output_attentions,
                                  output_hidden_states, return_dict, task_ids, **kwargs)
        hidden_states = outputs[0]
        send_queue.put(hidden_states[:, -1, :].unsqueeze(1))  # send last
        recv_queue.get()  # prevent 2 consecutive send
        logging.debug(f"kv cache shape{outputs.past_key_values[0][0].shape}")
        send_queue.put(torch.tensor(outputs.past_key_values[0][0].shape, dtype=torch.int16))  # send kv shape


class FedPreTrainedModelForCustomer(PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FedPretrainedConfig, tokenizer):
        super().__init__(config)

        self.lora_config_args = config.lora_config_args
        self.peft_lora_config = config.peft_lora_config
        self.vocab_size = self.lora_config_args.vocab_size
        self.hidden_size = self.lora_config_args.hidden_size
        self.embed_tokens = torch.nn.Embedding(self.vocab_size, self.hidden_size, config.pad_token_id)

        from .tuners import FedLoraMStack
        self.lora_M_stack = FedLoraMStack(self.peft_lora_config.r, self.peft_lora_config.r,
                                          self.peft_lora_config.target_modules,
                                          len(self.peft_lora_config.layers_to_transform))
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.load_state_dict({"weight": recv_queue.get()})
        self.new_gens = []
        self.tokenizer = tokenizer
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_ids=None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        input_embeds = self.embed_tokens(input_ids)
        send_queue.put(input_embeds)  # initial communication

        self.lora_M_stack()

        hidden_states = recv_queue.get()  # final comm
        send_queue.put(torch.zeros(1))

        # hidden_states = outputs[0] # og
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
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

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        past_kv_shape = recv_queue.get()
        bs, h, seq, hi = past_kv_shape.tolist()
        past_key_values = [[torch.zeros((int(bs), int(h), int(seq + 1), int(hi)))]]

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            input_embeds=None,
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
        if input_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": input_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
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
