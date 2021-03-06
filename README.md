# PolAQ (Polish Answers and Questions) - a master thesis

This project includes code that was created for my master thesis.
Its purpose was to create a new polish question answering
dataset based on english [*SQuAD* 1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset.
It consists of 2 smaller, independent projects:

- polaq_create
- polaq_test

The final results of the thesis are 5 ***PolAQ*** datasets:

- ***polaq_dataset_depth_0*** - 9890 question-answer pairs
- ***polaq_dataset_depth_0_combined*** - 10 495 question-answer pairs
- ***polaq_dataset_depth_1*** - 12 135 question-answer pairs
- ***polaq_dataset_depth_1_combined*** - 14 416 question-answer pairs
- ***polaq_dataset_manual*** - 5535 question-answer pairs

There is also a small, manually labelled test dataset ***polaq_dataset_test*** (164 questions).

All datasets were made available to use and can be obtained from [here](polaq_test/data).


## polaq_create

All code used for creating new ***PolAQ*** datasets with
a use of NLP tools like `gensim` or `spaCy`.

Please use `poetry` to install dependencies.

Requirements:

- `python` 3.7 or higher
- `poetry` 0.12 or higher (to install: `pip install poetry`)



## polaq_test

I tested here all 5 datasets. They were used as a training data (each of them independently)
for [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md).
Then each model tried to answer questions from a test set.
The results can be found in [output](polaq_test/output) folder.

Use provided bash scripts to initialize project,
download BERT model, train it and predict results.

Requirements:

- `python` 3.7 or higher

### Final score

The table below presents final score for all 5 datasets on the test set.
Both metrics (*EM* and *F1*) were explained in a [SQuAD paper](https://arxiv.org/abs/1606.05250).


|                             | EM (Exact Match) |   F1   |
| :-------------------------- |:----------------:| :-----:|
| *polaq\_manual*             |      71.9%       | 75.65% |
| *polaq\_depth\_0\_combined* |      71.34%      | 79.67% |
| *polaq\_depth\_0*           |      72.56%      | 80.39% |
| *polaq\_depth\_1\_combined* |      70.12%      | 77.12% |
| *polaq\_depth\_1*           |      69.51%      | 75.05% |
