"""Pre-Process the Dataset."""
import ast
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from pathlib import Path
MAXLEN = 64

CORPUS_MOVIE_CONV = './datasets/movie_conversations.txt'
CORPUS_MOVIE_LINES = './datasets/movie_lines.txt'

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


