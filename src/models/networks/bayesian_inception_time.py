import torch
import torch.nn as nn
from loguru import logger

from models.bayesian_module import BayesianModule, ConsistentMCDropout
from .mlp import MLP
from .registry import register_model
from .inception import InceptionBlock


class InceptionTime(BayesianModule):
    def __init__(
        self,
        channels_in=1,
        num_classes=5,
        dropout_p=0.0,
        small_head=True,
        n_filters=64,
        bottleneck_channels=64,
    ):
        """
        obtains the InceptionTime model for use as an Encoder.
        """
        super().__init__()
        
        # Build the InceptionTime backbone
        self.resnet = nn.Sequential(
            InceptionBlock(
                in_channels=channels_in,
                n_filters=n_filters,
                bottleneck_channels=bottleneck_channels,
                use_residual=True,
            ),
            InceptionBlock(
                in_channels=4 * n_filters,
                n_filters=n_filters,
                bottleneck_channels=bottleneck_channels,
                use_residual=True,
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        
        self.z_dim = 4 * n_filters
        
        if num_classes != 0:
            if small_head:
                self.classifier = nn.Sequential(
                    ConsistentMCDropout(p=dropout_p), nn.Linear(self.z_dim, num_classes)
                )
            else:
                self.classifier = nn.Sequential(
                    ConsistentMCDropout(p=dropout_p),
                    MLP(self.z_dim, num_classes, hidden_dims=[self.z_dim], bn=True),
                )
        else:
            self.classifier = nn.Identity()

    def det_forward_impl(self, input: torch.Tensor) -> torch.Tensor:
        return self.resnet(input)

    def mc_forward_impl(self, mc_input_BK: torch.Tensor) -> torch.Tensor:
        return self.classifier(mc_input_BK)

    def get_features(self, x):
        out = self.resnet(x)
        return out


@register_model
def inception_time(
    config,
    num_classes: int = 5,
    data_shape=[1, 140],
    **kwargs
) -> InceptionTime:
    
    channels_in = data_shape[0]
    dropout_p = config.model.dropout_p
    small_head = config.model.small_head
    
    n_filters = config.model.n_filters
    bottleneck_channels = config.model.bottleneck_channels

    return InceptionTime(
        channels_in=channels_in,
        num_classes=num_classes,
        dropout_p=dropout_p,
        small_head=small_head,
        n_filters=n_filters,
        bottleneck_channels=bottleneck_channels,
    )
