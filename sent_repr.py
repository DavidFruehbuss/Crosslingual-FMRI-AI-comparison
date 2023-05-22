import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pickle
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = 'cpu'

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
model = model.to(device)


for lang in ["EN", "CN", "FR"]:
    print(f"Encoding Language: {lang}")

    # read the aligned words
    with open(f"./text_data/{lang}_aligned_words.pickle", "rb") as f:
        words = pickle.load(f)  # each section contains list of scans; list of words

    # store encodings of each chunk
    hidden_states = []
    for i, sec in enumerate(words):
        encoded_input = tokenizer(sec, return_tensors="pt", padding=True).to(device)

        print(f"Encoding Sec {i}")

        output = model(**encoded_input, output_hidden_states=True)

        h_s = output.hidden_states

        print(f"last hidden state shape: {h_s[-1].shape}")

        hidden_states.append(list(h_s))

    # save the hidden states in a list of tensors (shape: batch_size, sequence_length, hidden_size)
    with open(f"/project/gpuuva021/shared/FMRI-Data/aligned/{lang}_hidden_states.pickle", "wb") as cache_file:
        pickle.dump(hidden_states, cache_file)
