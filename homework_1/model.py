from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding, LSTM


class SeqClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        embed_dim = self.embed.weight.shape[1]
        # TODO: model architecture
        D = 2 if bidirectional else 1
        self.lstm = LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(D * hidden_size, D * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(D * hidden_size, num_class)
        )
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    # reference method: https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        text_len = batch.shape[1]
        # embedding layer
        text_emb = self.embed(batch) # -> (B, L, D * H)
        # LSTM layer
        output, _ = self.lstm(text_emb)
        out_forward = output[:, text_len - 1, :self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size:]
        out_concat = torch.cat(tensors=(out_forward, out_reverse), dim=1) # -> (B, D * H)
        # MLP classification layer
        class_scores = self.mlp(out_concat)

        return class_scores

    def calc_loss(self, scores, labels):
        return self.criterion(scores, labels)