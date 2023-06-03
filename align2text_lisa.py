import tgt
import sys
from funcs import (
    text2fmri,
    extract_sent_list_from_tree_file,
    align_trees_with_csv_annotations,
)
import pickle
import textgrid
from pathlib import Path
import pandas as pd
import pickle


def main():
    # SETTINGS
    PATH = Path("/project/gpuuva021/shared/FMRI-Data")

    # get sentence information from tree files
    sentences = {
        "EN": extract_sent_list_from_tree_file(PATH / "annotation/EN/lppEN_tree.csv"),
        "FR": extract_sent_list_from_tree_file(PATH / "annotation/FR/lppFR_tree.csv"),
        "CN": extract_sent_list_from_tree_file(PATH / "annotation/CN/lppCN_tree.csv"),
    }

    OUTPATH = PATH / "text_data"
    OUTPATH.mkdir(parents=True, exist_ok=True)
    SENT_N = 1  # chunksize: nr. of sentences (of the same section)

    for language in sentences.keys():
        # read word informations from annotation csv file
        word_df = pd.read_csv(
            PATH / f"annotation/{language}/lpp{language}_word_information.csv"
        )

        # make sure to drop rows where the word is ' ' (space)
        # word_df = word_df[word_df["word"] != " "]

        # data is list of sections containing lists of sentences containing dict with keys "sentence", "onset", "offset", "section"
        data = align_trees_with_csv_annotations(
            sentences[language], language, word_df, chunck_size=SENT_N
        )

        # save the data as a pickle file
        with open(
            f"{OUTPATH}/{language}_chunk_data_chunk_size_{SENT_N}.pickle", "wb"
        ) as p:
            pickle.dump(data, p)


if __name__ == "__main__":
    main()
