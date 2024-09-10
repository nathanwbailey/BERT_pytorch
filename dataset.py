"""Custom Dataset for BERT."""
import torch
import random
import itertools

class BERTDataset(torch.utils.data.Dataset):
    """Custom Dataset Class for BERT."""
    def __init__(self, data_pair, tokenizer, seq_len=64):
        """Init Variables."""
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair
    
    def __len__(self):
        return self.corpus_lines
    
    def get_random_line(self):
        return self.lines[random.randrange(len(self.lines))][1]

    def get_corpus_line(self, item):
        return self.lines[item][0], self.lines[item][1]
    
    def get_sent(self, index):
        # is_next (1, 0) indicates if 2 sentences are consecutive in the original text
        t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            return t1, t2, 1
        return t1, self.get_random_line(), 0
        
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []
        for i, token in enumerate(tokens):
            prob = random.random()
            token_id = self.tokenizer(token)['input_ids'][1:-1]

            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    for _ in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])
                elif prob < 0.9:
                    for _ in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for _ in range(len(token_id)):
                    output_label.append(0)
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        return output, output_label

    
    def __getitem__(self, item):
        # Get a random sentence pair
        t1, t2, is_next_label = self.get_sent(item)

        # Replace random words in sentence with mask / random words
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # Add CLS and SEP tokens to the start and end
        t1 = [self.tokenizer.vocab['[CLS]']] + t1_random + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_random + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]

        bert_input = (t1+t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]

        bert_input.extend(padding)
        bert_label.extend(padding)
        segment_label.extend(padding)

        output = {
            "bert_input": bert_input,
            "bert_label": bert_label,
            "segment_label": segment_label,
            "is_next": is_next_label
        }

        return {key: torch.tensor(value) for key, value in output.items()}

