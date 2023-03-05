import torch
from typing import List
from models.bigram import BigramLanguageModel


def main():
    with open('src/cc_script.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = get_vocab(text)

    # create a mapping from chars to integers
    str_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_str = {i: c for i, c in enumerate(vocab)}
    # encoder: take a string, output a list of integers
    def encode(s): return [str_to_int[c] for c in s]
    # decoder: take a list of integers, output a string
    def decode(l): return ''.join([int_to_str[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]

    batch_size = 4  # how many independent sequences will we process in parallel?
    block_size = 8  # what is the maximum context length for predictions?

    xb, yb = get_batch(batch_size, block_size, train_data)
    m = BigramLanguageModel(len(vocab))
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)


def get_batch(batch_size, block_size, data):
    # generate a small batch of data from inputs x and targets y
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


def get_vocab(text: str) -> List[str]:
    # get all unique chars (the vocabulary)
    chars = sorted(list(set(text)))
    return chars


if __name__ == '__main__':
    main()
