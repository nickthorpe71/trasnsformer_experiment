import torch
import os
from typing import List
from models.bigram import BigramLanguageModel


def main():
    # hyper params
    new_dialog_length = 1000
    block_size = 256  # what is the maximum context length for predictions?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_embedding_dimensions = 384
    num_heads = 6
    num_layers = 6
    dropout = 0.2  # regularization technique for large models
    model_path = "src/trained_models/cc_script.pt"
    # --------------------------------------------

    with open('src/cc_script.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = get_vocab(text)
    int_to_str = {i: c for i, c in enumerate(vocab)}
    def decode(l): return ''.join([int_to_str[i] for i in l])
    
    model = BigramLanguageModel(
        len(vocab), num_embedding_dimensions, block_size, num_heads, num_layers, dropout)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    m = model.to(device)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    model_output = decode(m.generate(idx=context, max_new_tokens=new_dialog_length,
          block_size=block_size)[0].tolist())
    
    print(model_output)

def get_vocab(text: str) -> List[str]:
    # get all unique chars (the vocabulary)
    chars = sorted(list(set(text)))
    return chars


if __name__ == '__main__':
    main()
