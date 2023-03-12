import torch
import os
import re
from typing import List
from happytransformer import HappyTextToText
from happytransformer import TTSettings

from models.bigram import BigramLanguageModel


def main():
    # hyper params
    new_dialog_length = 10000
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

    # get model output
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    model_output = decode(m.generate(idx=context, max_new_tokens=new_dialog_length,
          block_size=block_size)[0].tolist())
    
    print("Model Output:")
    print(model_output)
    
    # get sentences from output
    sentences = get_dialog_blocks(model_output)
    
    # get grammar checker
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    args = TTSettings(num_beams=5, min_length=1)
    
    # correct sentences
    corrected_sentences = []
    for sentence in sentences:
        if should_skip(sentence) or len(sentence) < 5:
            corrected_sentences.append(sentence.strip())
            continue
        
        start_whitespace = re.search(r'^\s*', sentence).group()

        result = happy_tt.generate_text(sentence, args=args)
        corrected_sentences.append(start_whitespace + result.text)
    
    print("Result:")
    print('\n'.join(corrected_sentences))


def get_vocab(text: str) -> List[str]:
    # get all unique chars (the vocabulary)
    chars = sorted(list(set(text)))
    return chars

def get_sentences(dialog: str) -> List[str]:
    lines = dialog.split('\n')
    with_sentences = []
    gathered_sentence = ""
    for line in lines:
        if (should_skip(line)):
            with_sentences.append(line)
            continue

        gathered_sentence += line
        if (line[-1] == '.' or line[-1] == '?' or line[-1] == '!'):
            with_sentences.append(gathered_sentence)
            gathered_sentence = ""
    return with_sentences

def get_dialog_blocks(dialog: str) -> List[str]:
    lines = dialog.split('\n')
    with_sentences = []
    gathered_sentence = ""
    for line in lines:
        if (should_skip(line)):
            with_sentences.append(gathered_sentence)
            gathered_sentence = ""
            with_sentences.append(line)
            continue

        gathered_sentence += line
    return with_sentences
   
def should_skip(text: str) -> bool:
    return "]" in text or "[" in text or ":" in text or len(text) == 0
        


if __name__ == '__main__':
    main()
