import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pickle, os
import numpy as np
import argparse


def main(lm, device, languages, PATH, OUTPATH, sent_len, BATCH_SIZE):
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = AutoModelForMaskedLM.from_pretrained(lm)
    model = model.to(device)

    os.makedirs(OUTPATH, exist_ok=True)

    for sent_n in sent_len:
        # for lang in ["EN", "FR", "CN"]:
        for lang in languages:
            print(f"Encoding Language: {lang}, using chunk size: {sent_n}")

            # read the aligned words
            with open(
                f"{PATH}/text_data/{lang}_chunk_data_chunk_size_{sent_n}.pickle", "rb"
            ) as f:
                aligned_words = pickle.load(
                    f
                )  # each section contains list of chunks which is a dict with keys sentence, onset, offset, section

            # store encodings of each chunk
            hidden_states_per_section = []
            for i, section in enumerate(aligned_words):
                print(f"Encoding Sec {i}, len(section): {len(section)}")

                assert len(section) > 0, f"Section {i} has 0 chunks \n {section}"

                # get the sentences of the section
                section_sentences = [chunk["sentences"].strip() for chunk in section]

                section_batches = []
                for i in range(0, len(section_sentences), BATCH_SIZE):
                    try:
                        section_batches.append(section_sentences[i : i + BATCH_SIZE])
                    except IndexError:
                        section_batches.append(section_sentences[i:])  # Rest

                print(f"Nr. of batches: {len(section_batches)}")
                print(f"batch 0: {section_batches[0]}")

                # encode the section
                section_h_s = []
                for batch in section_batches:
                    print(f"Encoding batch: {batch}")
                    try:
                        encoded_input = tokenizer(
                            batch,
                            return_tensors="pt",
                            padding="max_length",
                            max_length=150,
                        ).to(device)
                    except IndexError:
                        print(f"=== IndexError: section {i} \n {section} ===")

                    with torch.no_grad():
                        output = model(**encoded_input, output_hidden_states=True)

                    print(f"Hidden state 0 shape: {output.hidden_states[0].shape}")

                    all_layers = torch.stack(output.hidden_states)
                    print(f"Output hidden states len: {len(output.hidden_states)}")
                    print(f"all layers shape: {all_layers.shape}")

                    section_h_s.append(all_layers)

                section_h_s = torch.cat((section_h_s), dim=1)

                section_h_s = section_h_s.to("cpu")

                # RuntimeError: Sizes of tensors must match except in dimension 1.
                # Expected size 76 but got size 57 for tensor number 1 in the list.

                # stack all batches
                print(
                    f"Section shape: {section_h_s.shape}"
                )  # layers, chunks, max length, hidden dim, torch.Size([13, 197, 168, 768])

                hidden_states_per_section.append(section_h_s)

            # stack all sections
            print(f"Hidden states per sec shape: {len(hidden_states_per_section)}")

            # save the hidden states in a list of tensors (sections, batches) (shape: batch_size, sequence_length, hidden_size)
            print(
                f"Saving hidden states of to {OUTPATH}/{lang}_hidden_states_chunk_size_{sent_n}.pickle"
            )
            with open(
                f"{OUTPATH}/{lang}_{lm}_hidden_states_chunk_size_{sent_n}.pickle",
                "wb",
            ) as f:
                pickle.dump(hidden_states_per_section, f)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument("--lm", default="xlm-roberta-base", type=str)
    parser.add_argument(
        "--languages", nargs="+", help="languages to encode", required=True
    )
    parser.add_argument("--sent_len", nargs="+", help="chunk sizes", required=True)

    args = parser.parse_args()
    kwargs = vars(args)

    # SETTINGS
    kwargs["PATH"] = "/project/gpuuva021/shared/FMRI-Data"
    kwargs[
        "OUTPATH"
    ] = "/home/lcur1142/advanced-techniques-computational-semantics/brain/aligned"
    kwargs["BATCH_SIZE"] = 32
    kwargs["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

    main(**kwargs)
