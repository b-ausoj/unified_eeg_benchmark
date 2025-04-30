import torch.nn as nn

class SimpleClassifier(nn.Module):
    """
    Simple classifier for the Unified EEG Benchmark.
    Encoder embedding dim (Maxim): 384.

    Args:
        input_dim (int): Input features (default: 384).
        hidden_dim (int): Hidden layer size (default: 512).
        output_dim (int): Output classes (default: 2).
        dropout (float): Dropout probability (default: 0.1).
    """
    def __init__(self, input_dim=384, hidden_dim=512, output_dim=2, dropout=0.1):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)
    
    