from typing import Dict
from argparse import Namespace

import torch
import torch.nn as nn
from torch.nn import Embedding, LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        rnn_type: str,
        hidden_size: int,
        num_layers: int,
        rnn_dropout: float,
        mlp_dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embed = Embedding.from_pretrained(embeddings, freeze=False) # NOTE: convert PAD to zeros --> not helpful
        embed_dim = self.embed.weight.shape[1]
        # model architecture
        D = 2 if bidirectional else 1
        # TODO: implement RNN type (args.rnn_type)
        RNN = getattr(nn, rnn_type)
        self.lstm = RNN(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=rnn_dropout, bidirectional=bidirectional)
        self.mlp = nn.Sequential(
            nn.Dropout(mlp_dropout),
            nn.Linear(D * hidden_size, D * hidden_size),
            nn.ReLU(), # NOTE: different activation functions --> the same
            nn.Dropout(mlp_dropout),
            nn.Linear(D * hidden_size, num_class)
        )
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    # reference method: https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # implement model forward
        text_len = batch.shape[1]
        # embedding layer
        text_emb = self.embed(batch) # -> (B, L, D * H)
        # LSTM layer
        output, _ = self.lstm(text_emb)
        # TODO: try different features
        out_forward = output[:, text_len - 1, :self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size:]
        out_concat = torch.cat(tensors=(out_forward, out_reverse), dim=1) # -> (B, D * H)
        # MLP classification layer
        class_scores = self.mlp(out_concat)

        return class_scores

    def calc_loss(self, scores, labels):
        return self.criterion(scores, labels)

class SlotTagger(nn.Module):
    def __init__(self, embeddings: torch.tensor, args: Namespace):
        super(SlotTagger, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        D = 2 if args.bidirectional else 1
        RnnModel = getattr(nn, args.rnn_type)
        self.rnn = RnnModel(
            input_size=self.embed.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_first=True,
            dropout=args.rnn_dropout,
            bidirectional=args.bidirectional
        )
        self.mlp = nn.Sequential(
            nn.Dropout(args.mlp_dropout),
            nn.Linear(D * args.hidden_size, D * args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.mlp_dropout),
            nn.Linear(D * args.hidden_size, args.num_class)
        )
        # utilities
        self.pack = lambda batch, seq_lens: pack_padded_sequence(batch, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        self.unpack = lambda rnn_out: pad_packed_sequence(rnn_out, batch_first=True)
        self.separate_seqs = lambda padded, seq_lens: [padded[i, :seq_len] for i, seq_len in enumerate(seq_lens)]
        self.scores_seqs_to_preds = lambda scores_seqs: list(map(lambda t: t.argmax(dim=-1), scores_seqs))

        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    def forward(self, x_batch, seq_lens):
        # embed layer
        x_embedded = self.embed(x_batch)
        # rnn layer
        x_packed = self.pack(x_embedded, seq_lens)
        rnn_out, _ = self.rnn(x_packed)
        rnn_out_padded, seq_lens_ = self.unpack(rnn_out)
        assert torch.all(torch.eq(seq_lens.cpu(), seq_lens_.cpu())).item()
        # mlp layer
        scores_padded = self.mlp(rnn_out_padded)

        return scores_padded
    
    def calc_loss(self, scores_padded, tags_padded):
        return self.criterion(scores_padded.transpose(1, 2), tags_padded)