import torch
from torch import nn
from transformers.models.mamba.modeling_mamba import MambaForCausalLM
from transformers import MambaConfig
from models.embeddings_mamba import MambaEmbeddingsForCEHR

class MambaClassifier(nn.Module):

    def __init__(
        self,
        device="cpu",
        pooling="mean",
        num_classes=2,
        sensors_count=37,
        static_count=8,
        layers=1,
        heads=1,
        dropout=0.2,
        attn_dropout=0.2,
        **kwargs
    ):
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
        
        self.config = MambaConfig(
            vocab_size=32,
        )

        self.embeddings = MambaEmbeddingsForCEHR(
            config=self.config,
        )

        self.post_init()

        self.model = MambaForCausalLM(config=self.config)

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def post_init(self) -> None:
        """Apply weight initialization."""
        self.apply(self._init_weights)

    def forward(self, x, static, time, sensor_mask, labels, **kwargs):

        inputs_embeds = self.embeddings(
            x = x,
            static = static,
            sensor_mask = sensor_mask,
        )

        return self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
        )
