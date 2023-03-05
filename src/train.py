import torch
from typing import List, Dict
from models.bigram import BigramLanguageModel

# Steps to break into modules:
# 1. Get hyper params
# 2. Data Loader
# 3. Get encoder and decoder
# 4. Define model
# 5. Define loss function
# 6. Define optimizer
# 7. Train model
# 8. Evaluate model
# 9. Save model

# Other Modules:
# Visualization


def main():
    # hyper params
    batch_size = 64  # how many independent sequences will we process in parallel?
    block_size = 256  # what is the maximum context length for predictions?
    max_iterations = 5000
    eval_interval = 500
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iterations = 200
    num_embedding_dimensions = 384
    num_heads = 6
    num_layers = 6
    dropout = 0.2  # regularization technique for large models
    # dropout of 0.2 means that 20% of the weights will be randomly set to 0 for every forward/backward pass
    # --------------------------------------------

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

    model = BigramLanguageModel(
        len(vocab), num_embedding_dimensions, block_size, num_heads, num_layers, dropout)
    m = model.to(device)

    # -- training --
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for iter in range(max_iterations):
        # every so often, evaluate the model
        if iter % eval_interval == 0:
            losses = estimate_loss(m, eval_iterations, batch_size,
                                   block_size, {'train': train_data, 'val': val_data}, device)
            print(
                f'iter {iter} | train loss {losses["train"]:.4f} | val loss {losses["val"]:.4f}')

        # sample a batch of data
        xb, yb = get_batch(batch_size, block_size, train_data, device)

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(idx=context, max_new_tokens=500,
          block_size=block_size)[0].tolist()))


def get_batch(batch_size, block_size, data, device):
    # generate a small batch of data from inputs x and targets y
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # don't track gradients for this function / greatly speeds up eval
def estimate_loss(model: BigramLanguageModel, eval_iterations: int, batch_size: int, block_size: int, data: Dict[str, torch.tensor], device: str) -> float:
    out = {}
    model.eval()  # puts the model in eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X, Y = get_batch(batch_size, block_size, data[split], device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # puts the model back in train mode
    return out


def get_vocab(text: str) -> List[str]:
    # get all unique chars (the vocabulary)
    chars = sorted(list(set(text)))
    return chars


if __name__ == '__main__':
    main()
