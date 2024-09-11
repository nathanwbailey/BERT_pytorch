"""BERT LM Model."""

import torch

from model_building_blocks import BERTEmbedding, EncoderLayer


class BERT(torch.nn.Module):
    """BERT Model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        heads: int = 12,
        dropout: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Init variables and layers."""
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        feed_forward_hidden = d_model * 4
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size, embed_size=d_model, device=device
        )

        self.encoder_blocks = torch.nn.ModuleList(
            [
                EncoderLayer(d_model, heads, feed_forward_hidden, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, segment_info: torch.Tensor
    ) -> torch.Tensor:
        """Forward Pass."""
        mask = x == 0
        x = self.embedding(x, segment_info)
        for encoder in self.encoder_blocks:
            x = encoder(x, mask)
        return x


class NextSentencePrediction(torch.nn.Module):
    """NSP Layer."""

    def __init__(self, hidden: int) -> None:
        """Init variables and layers."""
        # Predict if the input sequence has 2 consecutive sentences
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        # Only input the [CLS] token
        return self.linear(x[:, 0])


class MaskedLanguageModel(torch.nn.Module):
    """MLM Layer."""

    def __init__(self, hidden: int, vocab_size: int) -> None:
        """Init variables and layers."""
        super().__init__()
        # Predict the masked / random tokens
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        return self.softmax(self.linear(x))


class BERTLM(torch.nn.Module):
    """Bert LM Model."""

    def __init__(self, bert: BERT, vocab_size: int) -> None:
        """Init variables and layers."""
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(
        self, x: torch.Tensor, segment_label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward Pass."""
        x = self.bert(x, segment_label)
        # MLM and NSP at the end
        return self.next_sentence(x), self.mask_lm(x)
