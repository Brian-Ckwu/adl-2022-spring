from argparse import Namespace

import torch
import torch.nn as nn

from transformers import BertModel
from transformers.tokenization_utils_base import BatchEncoding

class BertIntentClassifier(nn.Module):
    def __init__(self, args: Namespace, num_classes: int) -> None:
        super(BertIntentClassifier, self).__init__()
        # model
        self.bert = BertModel.from_pretrained(args.bertType)
        self.fc = nn.Linear(
            in_features=self.bert.embeddings.word_embeddings.embedding_dim,
            out_features=num_classes
        )
        # loss
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, x: BatchEncoding) -> torch.FloatTensor:
        h = self.bert(**x).last_hidden_state
        h_cls = h[:, 0, :]
        scores = self.fc(h_cls)
        return scores

    def calc_loss(self, y_scores, y_true):
        return self.criterion(y_scores, y_true)

class BertSlotTagger(nn.Module):
    def __init__(self, args: Namespace, num_classes: int) -> None:
        super(BertSlotTagger, self).__init__()
        # model
        self.bert = BertModel.from_pretrained(args.bertType)
        self.fc = nn.Linear(
            in_features=self.bert.embeddings.word_embeddings.embedding_dim,
            out_features=num_classes
        )
        # loss
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, x: BatchEncoding) -> torch.FloatTensor:
        h = self.bert(**x).last_hidden_state
        h_slots = h[:, 1:-1, :]
        scores = self.fc(h_slots)
        return scores