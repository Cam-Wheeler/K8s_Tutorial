import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A simple CNN that we can use to train for image classification. Based on tiny VVG net https://poloclub.github.io/cnn-explainer/
    Simple conv layers with max pooling every other layer. Uses Relu for activation.
    """

    def __init__(self, input_shape: int, hidden_dim: int, layers: int, num_classes: int) -> None:
        super().__init__()

        input_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_dim,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU()
        )

        self.layers = [input_layer]

        for _ in range(layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        padding='same'),
                    nn.ReLU()
                )
            )
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        padding='same'),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
            )

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

        self.layers.append(classifier)

        self.complete_model = nn.Sequential(*self.layers)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.complete_model(x)

def model_factory(model_name: str = "simple_cnn", **kwargs):
    """
    Factory function to get model by name.
    """
    models = {
        "simple_cnn": SimpleCNN
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)

