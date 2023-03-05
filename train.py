import torch
from typing import List


def main():
    with open('cc_script.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = get_vocab(text)

    # create a mapping from chars to integers
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}
    # encoder: take a string, output a list of integers
    def encode(s): return [stoi[c] for c in s]
    # decoder: take a list of integers, output a string
    def decode(l): return ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    print(data[:1000])


def get_vocab(text: str) -> List[str]:
    # get all unique chars (the vocabulary)
    chars = sorted(list(set(text)))
    return chars


if __name__ == '__main__':
    main()
