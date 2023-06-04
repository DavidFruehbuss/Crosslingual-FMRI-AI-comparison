# Crosslingual-FMRI-AI-comparison

Comparing the representations of human FMRI recordings and large language model's layer activations in large language models across three languages. This repositiory contains the code and additional material for our submission.

## Structure

The code-base is structured as follows:

- [Analysis.ipynb](Analysis.ipynb): contains the main analyi
- [pipeline.job](pipeline.job)`: Contains the preprocessing pipeline to obtain the hidden representations from the monolingual and multilingual language models, and perform calculation intense RSA computations.
- [src/](src/): contains all the neccessary functions to run the analysis. The functions are split into the following files:
  - [src/align2text.py](src/align2text.py): contains functions that align the tree files with the csv files and save the data as a pickle file
  - [src/sent_repr.py](src/sent_repr.py): contains the functions to obtain the hidden representations from the language models
  - [src/get_sentiments.py](src/get_sentiments.py): contains the code neccessary to obtain the sentiment scores for the text segments
  - [src/subjects_model_to_model_RSA.py](src/subjects_model_to_model_RSA.py): contains the main RSA function for inbetween LM RSA analysis
  - [src/functs.py](src/functs.py): contains helper functions

### Additional Material

- The poster can be found [here](Poster.pdf).
- The paper can be found [here](Paper.pdf).
