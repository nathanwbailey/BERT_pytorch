"""Main file for BERT."""

import ast
from pathlib import Path

import torch
import tqdm
from tokenizers import BertWordPieceTokenizer  # type: ignore[import-untyped]
from transformers import BertTokenizer  # type: ignore[import-untyped]

from dataset import BERTDataset
from model import BERT, BERTLM
from train import train_model

MAXLEN = 64
NUM_EPOCHS = 100
BATCH_SIZE = 64

# Paths to the datasets
CORPUS_MOVIE_CONV = "./datasets/movie_conversations.txt"
CORPUS_MOVIE_LINES = "./datasets/movie_lines.txt"

# Choose the accelerator
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # MPS is Mac Silicon
else:
    DEVICE = torch.device("cpu")

with open(CORPUS_MOVIE_CONV, "r", encoding="iso-8859-1") as conv_file:
    conv = conv_file.readlines()

with open(CORPUS_MOVIE_LINES, "r", encoding="iso-8859-1") as line_file:
    lines = line_file.readlines()

lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    # dict is in the form line_num: line
    lines_dic[objects[0]] = objects[-1]


pairs = []  # type: list[list[str]]

for con in conv:
    # Find the line IDs for each conversation
    ids = con.split(" +++$+++ ")[-1].strip()
    ids = ast.literal_eval(ids)
    # Get all possible line pairs for the IDs listed
    for i, j in zip(ids, ids[1:]):
        qa_pairs = []
        first = lines_dic[i].strip()
        second = lines_dic[j].strip()
        # Cap the sentences at a MAXLEN
        qa_pairs.append(" ".join(first.split()[:MAXLEN]))
        qa_pairs.append(" ".join(second.split()[:MAXLEN]))
        pairs.append(qa_pairs)

Path("./data").mkdir(parents=True, exist_ok=True)
text_data = []
file_count = 0

# tqdm = progress bar
# Write the lines to multiple files
for sample in tqdm.tqdm(x[0] for x in pairs):
    text_data.append(sample)
    if len(text_data) == 10000:
        with open(
            f"./data/text_{file_count}.txt", "w", encoding="utf-8"
        ) as fp:
            fp.write("\n".join(text_data))
        text_data = []
        file_count += 1

# Look for all txt files in ./data
paths = [str(x) for x in Path("./data").glob("**/*.txt")]

# Create a BERT Word Piece tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True,
)

# Train a the Word Piece tokenizer
tokenizer.train(
    files=paths,
    vocab_size=30_000,
    min_frequency=5,
    limit_alphabet=1000,  # limit the alphabet size when creating the tokenizer
    wordpieces_prefix="##",
    special_tokens=[
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[UNK]",
    ],  # Add the following special tokens to the vocab
)

Path("./bert-it-1").mkdir(parents=True, exist_ok=True)
tokenizer.save_model("./bert-it-1")
# Create a tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(
    "./bert-it-1", local_files_only=True
)  # Only look for local files on machine and not from web

# BERT Dataset instance
train_data = BERTDataset(pairs, seq_len=MAXLEN, tokenizer=bert_tokenizer)

trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)

BERT_MODEL = BERT(
    vocab_size=len(bert_tokenizer.vocab),
    d_model=768,
    n_layers=4,
    heads=12,
    dropout=0.1,
    device=DEVICE,
).to(DEVICE)

MODEL = BERTLM(BERT_MODEL, len(bert_tokenizer.vocab)).to(DEVICE)


optimizer = torch.optim.Adam(
    params=filter(lambda param: param.requires_grad, MODEL.parameters()),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

# Loss is negative log likelihood
# 0 is unchanged token so ignore it for the loss
# We only want to focus on predictions of the changed tokens
loss_mlm = torch.nn.NLLLoss(ignore_index=0)
loss_nsp = torch.nn.BCEWithLogitsLoss()

# Train the model
train_model(
    model=MODEL,
    num_epochs=NUM_EPOCHS,
    optimizer=optimizer,
    loss_function_mlm=loss_mlm,
    loss_function_nsp=loss_nsp,
    trainloader=trainloader,
    device=DEVICE,
)
