"""Embedding layers for the models."""

import math
from typing import Any, Optional

import torch
from torch import nn
from transformers import BigBirdConfig, MambaConfig
from x_transformers import Encoder
import numpy as np


class TimeEmbeddingLayer(nn.Module):
    """Embedding layer for time features."""

    def __init__(self, embedding_size: int, is_time_delta: bool = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta

        self.w = nn.Parameter(torch.empty(1, self.embedding_size))
        self.phi = nn.Parameter(torch.empty(1, self.embedding_size))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, time_stamps: torch.Tensor) -> Any:
        """Apply time embedding to the input time stamps."""
        if self.is_time_delta:
            # If the time_stamps represent time deltas, we calculate the deltas.
            # This is equivalent to the difference between consecutive elements.
            time_stamps = torch.cat(
                (time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]),
                dim=-1,
            )
        time_stamps = time_stamps.float()
        time_stamps_expanded = time_stamps.unsqueeze(-1)
        next_input = time_stamps_expanded * self.w + self.phi

        return torch.sin(next_input)


class VisitEmbedding(nn.Module):
    """Embedding layer for visit segments."""

    def __init__(
        self,
        visit_order_size: int,
        embedding_size: int,
    ):
        super().__init__()
        self.visit_order_size = visit_order_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.visit_order_size, self.embedding_size)

    def forward(self, visit_segments: torch.Tensor) -> Any:
        """Apply visit embedding to the input visit segments."""
        return self.embedding(visit_segments)


class ConceptEmbedding(nn.Module):
    """Embedding layer for event concepts."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        padding_idx: Optional[int] = None,
    ):
        super(ConceptEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_size,
            padding_idx=padding_idx,
        )

    def forward(self, inputs: torch.Tensor) -> Any:
        """Apply concept embedding to the input concepts."""
        return self.embedding(inputs)


class PositionalEmbedding(nn.Module):
    """Positional embedding layer."""

    def __init__(self, embedding_size: int, max_len: int):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, embedding_size, 2).float()
            * -(math.log(10000.0) / embedding_size)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, visit_orders: torch.Tensor) -> Any:
        """Apply positional embedding to the input visit orders."""
        first_visit_concept_orders = visit_orders[:, 0:1]
        normalized_visit_orders = torch.clamp(
            visit_orders - first_visit_concept_orders,
            0,
            self.pe.size(0) - 1,
        )
        return self.pe[normalized_visit_orders]

class PositionalEncodingTF(nn.Module):
    """
    Based on the SEFT positional encoding implementation
    """

    def __init__(self, d_model, max_len=500):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        P_time = P_time.float()

        # create a timescale of all times from 0-1
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        # make a tensor to hold the time embeddings
        times = torch.Tensor(P_time.cpu()).unsqueeze(2)

        # scale the timepoints according to the 0-1 scale
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        # Use a 32-D embedding to represent a single time point
        pe = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1
        )  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        return pe


class BERTEmbeddingsForCEHR(nn.Module):
    """Embeddings for CEHR-BERT."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 128,
        time_embeddings_size: int = 16,
        type_vocab_size: int = 9,
        visit_order_size: int = 3,
        max_len: int = 512,
        layer_norm_eps: float = 1e-12,
        dropout_prob: float = 0.1,
        padding_idx: int = 1,
    ):
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.concept_embedding = ConceptEmbedding(
            num_embeddings=vocab_size,
            embedding_size=embedding_size,
            padding_idx=padding_idx,
        )
        self.token_type_embeddings = nn.Embedding(
            type_vocab_size,
            embedding_size,
        )
        self.time_embedding = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            is_time_delta=True,
        )
        self.age_embedding = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
        )
        self.positional_embedding = PositionalEmbedding(
            embedding_size=embedding_size,
            max_len=max_len,
        )
        self.visit_embedding = VisitEmbedding(
            visit_order_size=visit_order_size,
            embedding_size=embedding_size,
        )
        self.scale_back_concat_layer = nn.Linear(
            embedding_size + 2 * time_embeddings_size,
            embedding_size,
        )  # Assuming 4 input features are concatenated
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        concept_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
    ) -> Any:
        """Apply embeddings to the input features."""
        concept_embed = self.concept_embedding(concept_ids)
        type_embed = self.token_type_embeddings(type_ids)
        time_embed = self.time_embedding(time_stamps)
        age_embed = self.age_embedding(ages)
        positional_embed = self.positional_embedding(visit_orders)
        visit_segment_embed = self.visit_embedding(visit_segments)

        order_sequence_all = torch.arange(
            self.max_len, device=concept_ids.device
        ).expand_as(concept_ids)
        padding_mask = concept_ids == self.padding_idx
        order_sequence = torch.where(
            padding_mask,
            torch.tensor(self.max_len, device=concept_ids.device),
            order_sequence_all,
        )
        global_position_embed = self.positional_embedding(order_sequence)

        embeddings = torch.cat((concept_embed, time_embed, age_embed), dim=-1)
        embeddings = self.tanh(self.scale_back_concat_layer(embeddings))
        embeddings = (
            embeddings
            + type_embed
            + positional_embed
            + visit_segment_embed
            + global_position_embed
        )
        embeddings = self.LayerNorm(embeddings)

        return self.dropout(embeddings)


class BigBirdEmbeddingsForCEHR(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(
        self,
        config: BigBirdConfig,
        time_embeddings_size: int = 16,
        visit_order_size: int = 3,
    ) -> None:
        """Initiate wrapper class for embeddings used in BigBird CEHR classes."""
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.visit_order_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.time_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            is_time_delta=True,
        )
        self.age_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
        )
        self.visit_segment_embeddings = VisitEmbedding(
            visit_order_size=visit_order_size,
            embedding_size=config.hidden_size,
        )
        self.scale_back_concat_layer = nn.Linear(
            config.hidden_size + 2 * time_embeddings_size,
            config.hidden_size,
        )

        self.time_stamps: Optional[torch.Tensor] = None
        self.ages: Optional[torch.Tensor] = None
        self.visit_orders: Optional[torch.Tensor] = None
        self.visit_segments: Optional[torch.Tensor] = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file.
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory.
        self.position_embedding_type = getattr(
            config,
            "position_embedding_type",
            "absolute",
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )
        # End copy

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size

    def cache_input(
        self,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
    ) -> None:
        """Cache values for time_stamps, ages, visit_orders & visit_segments.

        These values will be used by the forward pass to change the final embedding.

        Parameters
        ----------
        time_stamps : torch.Tensor
            Time stamps of the input data.
        ages : torch.Tensor
            Ages of the input data.
        visit_orders : torch.Tensor
            Visit orders of the input data.
        visit_segments : torch.Tensor
            Visit segments of the input data.
        """
        self.time_stamps = time_stamps
        self.ages = ages
        self.visit_orders = visit_orders
        self.visit_segments = visit_segments

    def clear_cache(self) -> None:
        """Delete the tensors cached by cache_input method."""
        del self.time_stamps, self.ages, self.visit_orders, self.visit_segments

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> Any:
        """Return the final embeddings of concept ids using input and cached values."""
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :,
                past_key_values_length : seq_length + past_key_values_length,
            ]

        # Setting the token_type_ids to the registered buffer in constructor
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0],
                    seq_length,
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape,
                    dtype=torch.long,
                    device=self.position_ids.device,
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size**0.5)

        # Using cached values from a prior cache_input call
        time_stamps_embeds = self.time_embeddings(self.time_stamps)
        ages_embeds = self.age_embeddings(self.ages)
        visit_segments_embeds = self.visit_segment_embeddings(self.visit_segments)
        visit_order_embeds = self.visit_order_embeddings(self.visit_orders)

        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        inputs_embeds = torch.cat(
            (inputs_embeds, time_stamps_embeds, ages_embeds),
            dim=-1,
        )
        inputs_embeds = self.tanh(self.scale_back_concat_layer(inputs_embeds))
        embeddings = inputs_embeds + token_type_embeds
        embeddings += position_embeds
        embeddings += visit_order_embeds
        embeddings += visit_segments_embeds

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)

        # Clear the cache for next forward call
        self.clear_cache()

        return embeddings

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
        # Original
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,

        # Modified
        device="cpu",
        pooling="mean",
        sensors_count=37,
        static_count=8,
        num_classes=2,
        layers=1,
        heads=1,
        dropout=0.2,
        attn_dropout=0.2,
        **kwargs
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

        self.attn_layers = Encoder(
            dim=self.sensor_axis_dim,
            depth=layers,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=dropout,
        )

        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in, self.sensor_axis_dim)

        self.static_embedding = nn.Linear(self.static_count, self.static_out)
        self.nonlinear_merger = nn.Linear(
            self.sensor_axis_dim + self.static_out,
            self.sensor_axis_dim + self.static_out,
        )
        self.classifier = nn.Linear(
            self.sensor_axis_dim + self.static_out, num_classes
        )

        self.pos_encoder = PositionalEncodingTF(self.sensor_axis_dim)
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob

        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, x, static, time, sensor_mask, **kwargs,) -> Any:
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

        # make sensor embeddings
        x_time = self.sensor_embedding(x_time)  # (N, T, F)

        # add positional encodings
        pe = self.pos_encoder(time).to(self.device)  # taken from RAINDROP, (N, T, pe)

        # concatenate poolingated attented tensors
        static_embeds = self.static_embedding(static)
        static_embeds = torch.unsqueeze(static_embeds, 1).repeat(1, x_time.size(1), 1)

        inputs_embeds = torch.cat(
            (x_time, static_embeds),
            dim=-1,
        )

        # Apply dropout
        inputs_embeds = self.dropout(inputs_embeds)

        return inputs_embeds
