import torch
from model_building_blocks import BERTEmbedding
from model_building_blocks import EncoderLayer

class BERT(torch.nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1, device=torch.device('cpu')):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        feed_forward_hidden = d_model * 4
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model, device=device)

        self.encoder_blocks = torch.nn.ModuleList([
            EncoderLayer(d_model, heads, feed_forward_hidden, dropout) for _ in range(n_layers)
        ])
    
    def forward(self, x, segment_info):
        mask = (x == 0)
        x = self.embedding(x, segment_info)
        for encoder in self.encoder_blocks:
            x = encoder(x, mask)
        return x

class NextSentencePrediction(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(torch.nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        return self.softmax(self.linear(x))

class BERTLM(torch.nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)
    
    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)