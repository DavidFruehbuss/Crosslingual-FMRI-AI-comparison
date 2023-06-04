import argparse, pickle
from transformers import pipeline
from pathlib import Path


def get_sentiment_scores(data, sentiment_pipeline):
    all_sentiments = []
    for s, section in enumerate(data):
        print(f"sentence _,0: {data[s][0]['sentences']}")
        chunk_texts = [data[s][i]["sentences"] for i in range(len(data[s]))]

        sentiments = sentiment_pipeline(chunk_texts)
        all_sentiments.append(sentiments)

    # zip with the data
    for s, section in enumerate(data):
        for i, chunk in enumerate(data[s]):
            data[s][i]["sentiment"] = all_sentiments[s][i]

    return all_sentiments


def main(language, sent_len, OUT_FILE):
    if language == "EN" or language == "FR":
        model = "cardiffnlp/twitter-xlm-roberta-base-sentiment" # POS, NEUT, NEG
    elif language == "CN":
        model = "liam168/c2-roberta-base-finetuned-dianping-chinese"

    sentiment_pipeline = pipeline("sentiment-analysis", model=model)

    PATH = Path("/project/gpuuva021/shared/FMRI-Data/text_data")
    file_name = f"{language}_chunk_data_chunk_size_{sent_len}.pickle"

    IN_FILE = PATH / file_name

    with open(IN_FILE, "rb") as f:
        text_data = pickle.load(f)

    # get sentiment for each sentence, integrated into the text_data
    all_sentiments = get_sentiment_scores(text_data, sentiment_pipeline)

    # save the data as a pickle file
    with open(OUT_FILE, "wb") as p:
        pickle.dump(text_data, p)

    print(f"Saved data incl. sentiment to {OUT_FILE}")


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--sent_len", type=int, required=True)

    # Model hyperparameters
    args = parser.parse_args()
    kwargs = vars(args)

    # SETTINGS
    OUT_PATH = Path(
        "/home/lcur1142/advanced-techniques-computational-semantics/brain/text_data"
    )
    OUT_FILE = (
        OUT_PATH
        / f"{kwargs['language']}_chunk_data_chunk_size_{kwargs['sent_len']}_sentiment.pickle"
    )

    kwargs["OUT_FILE"] = OUT_FILE

    main(**kwargs)
