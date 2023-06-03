import pandas as pd
from nltk.tree import Tree


def fmri2words(text_data, Trs, section, delay=5, window=0.2):
    chunks = []
    text = text_data[text_data["section"] == section]
    for tr in range(Trs):
        onset = tr * 2 - delay
        offset = onset + 2
        chunk_data = text[
            (text["onset"] >= onset - window) & (text["offset"] < offset + window)
        ]
        chunks.append(" ".join(list(chunk_data["word"])))
    return chunks


def extract_words_from_tree(tree):
    words = []
    if isinstance(tree, str):  # Base case: leaf node (word)
        return [tree]

    elif isinstance(tree, Tree):
        for subtree in tree:
            words.extend(extract_words_from_tree(subtree))

    return words


def extract_sent_list_from_tree_file(PATH):
    with open(PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sentences = []
    counter = 0
    for i, line in enumerate(lines):
        line = line.strip()
        try:
            tree = Tree.fromstring(line)
        except ValueError:
            try:  # remove last ')'
                tree = Tree.fromstring(line[:-1])

            except ValueError:
                counter += 1
                print(f"=== ValueError: line {i} \n {line} ===")
                continue
        words = extract_words_from_tree(tree)
        sentences.append(words)  # list of list of words

    print(f"Errors during tree parsings: {counter}")
    return sentences


def get_section_data(word_df, section, chunk_size):
    df_by_section = word_df[word_df["section"] == section]

    words, onsets, offsets, sections = (
        df_by_section["word"].tolist(),
        df_by_section["onset"].tolist(),
        df_by_section["offset"].tolist(),
        df_by_section["section"].tolist(),
    )

    # create list of dicts
    section_data = []
    sentence = ""
    temp_onsets, temp_offsets, temp_sections = [], [], []
    for start, word in enumerate(words):
        sentence = sentence + word + " "
        temp_offsets.append(offsets[start])
        temp_onsets.append(onsets[start])
        temp_sections.append(sections[start])

        if word[-1] == "#":
            section_data.append(
                {
                    "sentences": sentence[:-2] + ".",
                    "onset": temp_onsets[0],
                    "offset": temp_offsets[-1],
                    "section": sections[0],
                }
            )
            sentence, temp_onsets, temp_offsets, temp_sections = (
                "",
                [],
                [],
                [],
            )  # reset

    # chunk the section data #TODO: make it work for chunk size > 2
    if chunk_size > 1:
        chunk_data = []
        try:
            for start in range(0, len(section_data), chunk_size):
                # print(f"from {start} to {start + chunk_size}")
                # print(f'get onset from {start}: {section_data[start]["onset"]}')
                # print(
                #     f'Get offset from {start + chunk_size}: {section_data[start + chunk_size]["offset"]}'
                # )

                chunk = ""
                for j in range(chunk_size):
                    chunk += section_data[start + j]["sentences"] + " "

                onset, offset, section = (
                    section_data[start]["onset"],
                    section_data[start + chunk_size - 1]["offset"],
                    section_data[start + chunk_size - 1]["section"],
                )

                chunk_data.append(
                    {
                        "sentences": chunk,
                        "onset": onset,
                        "offset": offset,
                        "section": section,
                    }
                )

        except IndexError:  # add the last chunk in case of chunk size 2
            chunk_data.append(section_data[-1])

        return chunk_data

    else:
        return section_data


def align_trees_with_csv_annotations(sentences, language, word_df, chunck_size=1):
    # replace the last word with the word + #
    for i, sent in enumerate(sentences):
        sentences[i][-1] = sent[-1] + "#"

    # flatten the list of lists of words into a list of words
    words = [item for sublist in sentences for item in sublist]

    print(word_df["word"].tolist()[:10])

    # TODO: check how bad the mismatch is
    if language == "FR":
        print("FR")
        words = words[:-1]
        words[-1] = words[-1] + "#"

    # counter = 0
    # for i, (word, word_csv) in enumerate(zip(words, word_df["word"].tolist())):
    #     if word != word_csv and word != word_csv + "#":
    #         counter += 1
    #         if counter < 10:
    #             print(f"{i} Word mismatch: {word} != {word_csv}")

    #     # if counter > 10:
    #     #     break
    # print(f"Total number of mismatches: {counter}")

    # integrate words back into the dataframe
    assert len(words) == len(
        word_df
    ), f"The number of words of tree ({len(words)}) and csv ({len(word_df)}) does not match"
    # print(len(words), len(word_df))

    word_df["word"] = words

    # keep only relevant columns of the dataframe
    word_df = word_df[["word", "onset", "offset", "section"]]

    # dump as csv
    word_df.to_csv(f"test.csv", index=False)

    # get the number of unique sections
    possible_sections = word_df["section"].unique()

    # extract as lists, for each section individually
    data = []
    for section in possible_sections:
        section_data = get_section_data(word_df, section, chunck_size)

        # add the section's data to the list of all data
        data.append(section_data)

        print(f"{language} Section {section} has {len(section_data)} chunks")

    return data


def text2fmri(textgrid, sent_n, delay=5, lan=None, sentences=None):
    # OLD
    scan_idx = []
    chunks = []
    textgrid = textgrid.tiers
    chunk = ""
    sent_i = 1
    idx_start = int(delay / 2)

    if lan == "EN":
        for interval in textgrid[0].intervals[1:]:
            # print(interval.mark)
            # print(interval.__dict__)
            # different marks depending on the language (EN, CN, FR)
            if (
                interval.mark == "#" or interval.mark == "sil"
            ):  # or interval.mark == "":
                chunk += "."
                if sent_i == sent_n:
                    chunks.append(chunk[1:])
                    idx_end = min(int((interval.maxTime + delay) / 2) + 1, 282)
                    scan_idx.append(slice(idx_start, idx_end))
                    sent_i = 0
                    chunk = ""
                    idx_start = idx_end - 1
                sent_i += 1
                continue
            chunk += " " + interval.mark
        return chunks, scan_idx
