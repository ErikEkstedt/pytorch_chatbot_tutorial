# Chatbot Tutorial

This is a restructuring of the 
[PyTorch chatbot tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html).


Almost everything is the same except some functions have been split and rename for clarity
(for myself at least). The goal was to make the long form interactive script structure
more like a regular deep learning project with separate scripts and directories.


Structure:
```bash
.
├── README.md
├── chatbot_tutorial.py
├── data
│   ├── lines_of_pairs_movie_lines.txt
│   └── pairs_voc_trimmed_min4_pairs.pt
├── dataset.py
├── models.py
└── preprocess.py
```

## Data

The data is downloaded and preprocessed as the original tutorial describes. 

### [preprocess.py](preprocess.py)

This script assumes that the zipped data is downloaded and extracted. Then writes a csv
file were each row contains a context-response pair separated by a delimiter ('\t' by
default)

###  [dataset.py](dataset.py)

Contains a function `load_pairs_trim_and_save(min_count=4)` which reads the preprocess csv
file, trims the data by removing words used less than `min_count` and saves(torch.save)
the list of training pairs and the corresponding vocabulary.


The number of pairs left after trim.
```bash
min_count   : kept      : #pairs
--------------------------------
3           : 88.9%     : 196772
4           : 85.5%     : 189173
5           : 82.1%     : 181592
6           : 79.4%     : 175622
7           : 76.9%     : 170092
8           : 74.6%     : 165050
```

TODO:

Also contains the dataset and dataloader used for training.
