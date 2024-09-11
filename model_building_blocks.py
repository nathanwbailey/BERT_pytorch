"""Building Blocks for the BERT Model."""

import math

import torch


class PositionalEmbedding(torch.nn.Module):
    """Positional Embedding Layer."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 128,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Init Variables and Layers."""
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False  # type: ignore[attr-defined]
        # Generate Positional Embeddings
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))

                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / d_model))
                )
        # Unsqueeze for the batch_size dim
        self.pe = pe.unsqueeze(0).to(device)

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        """Return the Embedding."""
        return self.pe


class BERTEmbedding(torch.nn.Module):
    """BERT Embedding Layer."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        seq_len: int = 64,
        dropout: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Init Variables and Layers."""

        super().__init__()
        self.embed_size = embed_size
        # Do not update the gradients for padding (0)
        self.token = torch.nn.Embedding(
            vocab_size, embed_size, padding_idx=0
        )  # Token embeding
        self.segment = torch.nn.Embedding(
            3, embed_size, padding_idx=0
        )  # Segment label embedding
        self.position = PositionalEmbedding(  # Positional Embedding
            embed_size, seq_len, device=device
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, sequence: torch.Tensor, segment_label: torch.Tensor
    ) -> torch.Tensor:
        """Forward Pass."""
        # Combine the 3 embeddings
        x = (
            self.token(sequence)
            + self.position(sequence)
            + self.segment(segment_label)
        )
        return self.dropout(x)


class FeedForward(torch.nn.Module):
    """Custom Linear Network."""

    def __init__(
        self, d_model: int, middle_dim: int = 2048, dropout: float = 0.1
    ) -> None:
        """Init Variables and Layer."""
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class EncoderLayer(torch.nn.Module):
    """Transformer (Encoder-Style) Block."""

    def __init__(
        self,
        d_model: int = 768,
        heads: int = 12,
        feed_forward_hidden: int = 768 * 4,
        dropout: float = 0.1,
    ) -> None:
        """Init Layers."""
        super().__init__()
        self.layernorm_1 = torch.nn.LayerNorm(d_model)
        self.layernorm_2 = torch.nn.LayerNorm(d_model)
        self.multihead = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, batch_first=True
        )
        self.feedforward = FeedForward(d_model, feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward Pass."""
        # Mask is a key padding mask, so padding is ignored for attention in the keys
        interacted = self.dropout(
            self.multihead(
                query=embeddings,
                key=embeddings,
                value=embeddings,
                key_padding_mask=mask,
            )[0]
        )
        interacted = self.layernorm_1(interacted + embeddings)
        feed_forward_out = self.dropout(self.feedforward(interacted))
        encoded = self.layernorm_2(feed_forward_out + interacted)
        return encoded
