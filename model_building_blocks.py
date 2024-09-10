import torch
import math

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=128, device=torch.device("cpu")):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/d_model)))

                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*(i+1))/d_model)))
        
        self.pe = pe.unsqueeze(0).to(device)

    def forward(self, _):
        return self.pe
    
class BERTEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len=64, dropout=0.1, device=torch.device("cpu")):

        super().__init__()
        self.embed_size = embed_size
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(embed_size, seq_len, device=device)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()
    
    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out
    
class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model=768,
        heads=12,
        feed_forward_hidden = 768*4,
        dropout = 0.1,
    ):
        super().__init__()
        self.layernorm_1 = torch.nn.LayerNorm(d_model)
        self.layernorm_2 = torch.nn.LayerNorm(d_model)
        self.multihead = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=heads)
        self.feedforward = FeedForward(d_model, feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, embeddings, mask):
        interacted = self.dropout(self.multihead(query=embeddings, key=embeddings, value=embeddings)[0])
        interacted = self.layernorm_1(interacted+embeddings)
        feed_forward_out = self.dropout(self.feedforward(interacted))
        encoded = self.layernorm_2(feed_forward_out + interacted)
        return encoded

