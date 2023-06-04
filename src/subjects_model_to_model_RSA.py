from nilearn import datasets, image, masking, plotting, regions
from nilearn.input_data import NiftiMasker
from sklearn.decomposition import PCA
from nilearn.image import new_img_like
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import pickle
import json
import math
from pathlib import Path


def rsa_function(
    file_path,
    data_type="FMRI",
    N_components=280,
    layer=-1,
    text_bins=None,
    interp=False,
    interp_size=61,
):
    """
    Computes the sentence by sentence RSA with cosine similarity.
    """

    if data_type == "FMRI":
        nii_img = image.load_img(file_path)

        if text_bins == None:
            nii_data = nii_img.get_fdata()
            time_series = nii_data.transpose(3, 0, 1, 2).reshape(nii_data.shape[3], -1)

        else:
            # extract and mean the fmri scans for each sentence chunck
            nii_data = np.array(
                [
                    image.mean_img(image.index_img(nii_img_1, sent["idx"])).get_fdata()
                    for sent in text_bins[0]
                ]
            )
            time_series = nii_data.reshape(nii_data.shape[0], -1)

        N_components = min(N_components, len(time_series))
        pca = PCA(N_components)
        pca.fit(time_series)
        # print('Explained variance:', pca.explained_variance_ratio_)

        data_matrix = pca.transform(time_series)

    elif data_type == "AI":
        # Expected input: torch.Size([93, 59, 768])
        with open(file_path, "rb") as f:
            hidden_activations = pickle.load(f)

        # input shape is [sections, layers, chunks, words, hidden_dim]
        h_a = hidden_activations[0][layer]  # last hidden_activations for first section

        hidden_activations = h_a.numpy()
        data_matrix = np.mean(hidden_activations, axis=1)

    # discard first 8 seconds to account for FMRI warmup
    data_matrix = data_matrix[4:, :]

    if interp:
        # interpolation
        # Calculate the interpolation factor
        interp_factor = interp_size / len(data_matrix)

        # Generate indices for interpolation
        indices = np.arange(interp_size) / interp_factor

        # Perform linear interpolation
        RSA_interp = np.empty((interp_size, data_matrix.shape[1]))
        for dim in range(data_matrix.shape[1]):
            RSA_interp[:, dim] = np.interp(
                indices, np.arange(len(data_matrix)), data_matrix[:, dim]
            )

        data_matrix = RSA_interp

    # compute the RSA as sentence by sentence cosine sim matrix
    RSA = sp.spatial.distance.cdist(data_matrix, data_matrix, metric="cosine")
    RSA = RSA / RSA.max()

    return RSA


def main():
    """
    compute the RSAs for all languages and all layers (first section only)
    """

    models = [
        "EN_bert-base-uncased",
        "CN_bert-base-chinese",
        "FR_Geotrend/bert-base-fr-cased",
    ]
    RSAs_models = {}

    for model in models:
        print(f"Computing RSAs for {model}")
        file_path = f"./aligned/{model}hidden_states_chunk_size_2.pickle"
        for layer in range(13):
            key = model + "_" + str(layer)

            print(f"Computing RSAs for {key}")
            RSAs_models[key] = rsa_function(
                file_path, "AI", layer=layer, interp=True, interp_size=61
            )

    RSAs_list = {}
    for key, value in RSAs_models.items():
        RSAs_list[key] = value.tolist()

    with open("./RSAs/RSAs_Monolingual_Models.json", "w") as f:
        json.dump(RSAs_list, f)


if __name__ == "__main__":
    main()
