from typing import Any, Optional

import torch
from torch import nn
from transformers import MambaConfig

def masked_mean_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """
    # eliminate all values learned from nonexistant timepoints
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()
    data_summed = torch.sum(datatensor * mask_expanded, dim=1)

    # find out number of existing timepoints
    data_counts = mask_expanded.sum(1)
    data_counts = torch.clamp(data_counts, min=1e-9)  # put on min clamp

    # Calculate average:
    averaged = data_summed / (data_counts)

    return averaged


def masked_max_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """
    # eliminate all values learned from nonexistant timepoints
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()

    datatensor[mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    maxed = torch.max(datatensor, 1)[0]

    return maxed

class MambaEmbeddingsForCEHR(nn.Module):
    """Construct the embeddings from concept, token_type, etc., embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(
        self,
        config: MambaConfig,
        device="cpu",
        pooling="mean",
        sensors_count=37,
        static_count=8,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        """Initiate wrapper class for embeddings used in Mamba CEHR classes."""
        super().__init__()
        self.pooling = pooling
        self.device = device
        self.sensors_count = sensors_count
        self.static_count = static_count

        self.sensor_axis_dim_in = 2 * self.sensors_count

        self.sensor_axis_dim = self.sensor_axis_dim_in
        if self.sensor_axis_dim % 2 != 0:
            self.sensor_axis_dim += 1

        self.static_out = self.static_count + 4

        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = config.hidden_size

        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in, self.sensor_axis_dim)

        self.static_embedding = nn.Linear(self.static_count, self.static_out)

        self.scale_back_concat_layer = nn.Linear(
            self.sensor_axis_dim + self.static_out,
            config.hidden_size,
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file.
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        # End copy

    def forward(
        self,
        x,
        static,
        #time,
        sensor_mask,
    ) -> Any:
        """Return the final embeddings of concept ids.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input data (concept_ids) to be embedded.
        inputs_embeds : torch.Tensor
            The embeddings of the input data.
        token_type_ids_batch : torch.Tensor
            The token type IDs of the input data.
        time_stamps : torch.Tensor
            Time stamps of the input data.
        ages : torch.Tensor
            Ages of the input data.
        visit_orders : torch.Tensor
            Visit orders of the input data.
        visit_segments : torch.Tensor
            Visit segments of the input data.
        """
        x_time = torch.clone(x)  # (N, F, T)
        x_time = torch.permute(x_time, (0, 2, 1))  # (N, T, F)
        mask = (
            torch.count_nonzero(x_time, dim=2)
        ) > 0  # mask for sum of all sensors for each person/at each timepoint

        # add indication for missing sensor values
        x_sensor_mask = torch.clone(sensor_mask)  # (N, F, T)
        x_sensor_mask = torch.permute(x_sensor_mask, (0, 2, 1))  # (N, T, F)
        x_time = torch.cat([x_time, x_sensor_mask], axis=2)  # (N, T, 2F) #Binary
        
        x_embeds = self.sensor_embedding(x_time)
        static_embeds = self.static_embedding(static)
        static_embeds = torch.unsqueeze(static_embeds, 1).repeat(1, x_embeds.size(1), 1)

        # if self.pooling == "mean":
        #     x_embeds = masked_mean_pooling(x_embeds, mask)
        # elif self.pooling == "median":
        #     x_embeds = torch.median(x_embeds, dim=1)[0]
        # elif self.pooling == "sum":
        #     x_embeds = torch.sum(x_embeds, dim=1)  # sum on time
        # elif self.pooling == "max":
        #     x_embeds = masked_max_pooling(x_embeds, mask)

        inputs_embeds = torch.cat(
            (x_embeds, static_embeds),
            dim=-1,
        )

        inputs_embeds = self.tanh(self.scale_back_concat_layer(inputs_embeds))
        embeddings = self.dropout(inputs_embeds)

        return self.LayerNorm(embeddings)