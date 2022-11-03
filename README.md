Cross-lingual approaches to fine-grained emotion detection
==============

We explore cross-lingual aproaches to fine-grained emotion detection
as well as creating fine-grained emotion datasets for Spanish.


Requirements to run the experiments
--------
- python3
- sklearn
- scipy


Usage
--------

The 'models' folder contains the main model.

Reports Pearson correlation and p-values for models trained on source language data and tested on source (SRC-SRC) and then tested on target language data (SRC-TRG)

```
python3 models/model.py --src_dataset dataset/en --trg_dataset dataset/es --emotion anger --features ngrams

src_dataset done
trg_dataset done
Training SVR...
SRC-SRC: 0.47 (0.00)
SRC-TRG: 0.06 (0.25)
```

model.py has the following arguments:

```
optional arguments:
  -h, --help            show this help message and exit
  -sd SRC_DATASET, --src_dataset SRC_DATASET
  -td TRG_DATASET, --trg_dataset TRG_DATASET
  -emo EMOTION, --emotion EMOTION
  -f FEATURES [FEATURES ...], --features FEATURES [FEATURES ...]
```

--src_dataset should point to a repository with the subrepos 'train', 'dev', and 'test'. The default is dataset/en

--trg_dataset should point to a repository with the subrepo 'test'. The default is dataset/es

--emotion is one of the four emotions: anger, fear, joy, sadness

  The optional features are:
  1. 'ngrams': token ngrams (1-4),
  2. 'char_ngrams': character ngrams (3-5),
  3. 'NRC_sent': lexicon features from the English NRC sentiment lexicon (see paper)
  4. 'NRC_emo': lexicon features from the English NRC emotion lexicon
  5. 'NRC_hash': lexicon features from the English NRC hashtag emotion lexicon


To Do
---------

Notice that directly applying a model without any transfer method leads to very poor results (0.06 compared to 0.47).

Use (LibreTranslate)[https://libretranslate.com/] to create two new versions of the data: 1) Spanish translations of the original English train/dev data and 2) English translations of the Spanish test data. Which approach works best?

Use an available multiLingual language model (mBERT, XLM Robert, etc) and fine-tune this on the original English train/dev data and then test directly on the Spanish test data. How well does this compare to the other approaches?


If you use this code, please cite the following paper:
-------
```
@inproceedings{NavasAlejo2020,
  author =  "Navas Alejo, Irean
        and  Badia, Toni
        and  Barnes, Jeremy",
  title =   "Cross-lingual Emotion Intensity Prediction",
  booktitle =   "Proceedings of the Third Workshop on Computational Modeling of People’s Opinions, Personality, and Emotions in Social Media (PEOPLES 2020)",
  year =    "2020",
  publisher =   "Association for Computational Linguistics",
  pages =   "",
  location =    "Barcelona, Spain"
}
```


License
-------

Copyright (C) 2022, Jeremy Barnes

Licensed under the terms of the Creative Commons CC-BY public license
