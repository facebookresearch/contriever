# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
from transformers import BertModel

class Contriever(BertModel):

    def __init__(self, config, **kwargs):
        super().__init__(config, add_pooling_layer=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output['last_hidden_state']
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.)
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        if normalize:
            emb = torch.nn.functionnal.normalize(emb, dim=-1)
        return emb
