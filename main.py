"""Pre-Process the Dataset."""
import ast
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from pathlib import Path
from model import BERTLM
from model import BERT
from dataset import BERTDataset
import torch
from train import train_model

MAXLEN = 64

CORPUS_MOVIE_CONV = './datasets/movie_conversations.txt'
CORPUS_MOVIE_LINES = './datasets/movie_lines.txt'

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

with open(CORPUS_MOVIE_CONV, 'r', encoding='iso-8859-1') as c:
    conv = c.readlines()

with open(CORPUS_MOVIE_LINES, 'r', encoding='iso-8859-1') as l:
    lines = l.readlines()

lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    # line_num: line
    lines_dic[objects[0]] = objects[-1]


pairs = [] # type: list[list[str]]

for con in conv:
    ids = con.split(" +++$+++ ")[-1].strip()
    ids = ast.literal_eval(ids)
    for i, j in zip(ids, ids[1:]):
        qa_pairs = []
        first = lines_dic[i].strip()
        second = lines_dic[j].strip()
        # Cap the sentences at a MAXLEN
        qa_pairs.append(' '.join(first.split()[:MAXLEN]))
        qa_pairs.append(' '.join(second.split()[:MAXLEN]))
        pairs.append(qa_pairs)

Path("./data").mkdir(parents=True, exist_ok=True)
text_data = []
file_count = 0

#tqdm = progress bar
for sample in tqdm.tqdm(x[0] for x in pairs):
    text_data.append(sample)
    if len(text_data) == 10000:
        with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1

#Look for all txt files in ./data
paths = [str(x) for x in Path('./data').glob('**/*.txt')]

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
)

tokenizer.train(
    files=paths,
    vocab_size=30_000,
    min_frequency=5,
    limit_alphabet=1000, #limit the alphabet size when creating the tokenizer
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'] # Add the following special tokens to the vocab
)

Path("./bert-it-1").mkdir(parents=True, exist_ok=True)
tokenizer.save_model('./bert-it-1')
bert_tokenizer = BertTokenizer.from_pretrained('./bert-it-1', local_files_only=True) #Only look for local files on machine and not from web

train_data = BERTDataset(pairs, seq_len=MAXLEN, tokenizer=bert_tokenizer)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)

bert_model = BERT(
    vocab_size=len(bert_tokenizer.vocab),
    d_model=768,
    n_layers=2,
    heads=12,
    dropout=0.1
)

model = BERTLM(bert_model, len(bert_tokenizer.vocab))


optimizer = torch.optim.Adam(
    params=filter(lambda param: param.requires_grad, model.parameters()),
    lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
)

loss = torch.nn.NLLLoss(ignore_index=0)

train_model(model=model, num_epochs=100, optimizer=optimizer, loss_function=loss, trainloader=trainloader, device=DEVICE)