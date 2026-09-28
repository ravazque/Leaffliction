"""Convolutional network that classifies leaf images."""
from torch import nn

MIN_SIZE = 32  # four 2x poolings need at least this many pixels


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class LeafCNN(nn.Module):
    """Four conv blocks (32 to 256 filters), global pooling, linear head."""

    def __init__(self, n_classes, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        blocks, cin = [], 3
        for cout in widths:
            blocks.append(_block(cin, cout))
            cin = cout
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(cin, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
