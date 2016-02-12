## HW1 - alignment

Our approach for the alignment is to run Model 2 with the following additional modifications.

1. Initial probabilities
    - Since we had Model 1's resulting probabilities, we adopted the learned probabilities as our initial parameters for Model 2. We found out that the best number of EM iterations to get this initial probabilities was 3.

2. Most frequent words
    - Some common words like `der` in German are more likely to appear in the corpus, and at the same time more likely to be aligned unambiguously. We extracted 500 most frequent German and English word lists, went through Google Translate to get translation pairs with corresponding languages. Finally we appended the resulting pairs multiple times in the end of corpus in order for the model to learn the probable alignment. We found that 5 times gave us the best result.

3. Biasing diagonal entries
    - To favor diagonal alignments, we introduced alpha as an multiplier to the probabilities if the alignments go diagonally. Therefore diagonal entries are promoted by alpha times more than other alignments using this initialization.

## Team
- Adhiguna Surya Kuncoro (AustinTexas)
- Hiroaki Hayashi (rooa)

## Details
### Run code

Example output will be like following:

```
$ python align_model2.py -b [bitext] -r 5 -i 15 -a 3 -p [pretrained p_e_given_f in pickle format] > output.txt
```
The options:

| Options     | Description     |
| :------------- | :------------- |
| b       | bitext file to train (, dev, and test)       |
| r       | Repetition of most frequent words' pairs       |
| i       | Number of EM iterations       |
| a       | Bias parameter for diagonal entries       |
| p       | Initial probability for Model 2 (trained on Model 1) in pickled form   |


### File description

- `data/english_most_freq_translated_pair_500.txt`
    - 500 most frequent words in English, translated to German
- `data/german_most_freq_translated_pair_500.txt`
    - 500 most frequent words in German, translated to English
- `outputs/`
    - contains previous outputs for backup
- `align_model1.py`
    - IBM Model1 implementation, done by rooa
- `align_model2.py`
    - IBM Model2 implementation, done by AustinTexas

### MISC

- Probability of a English word given a German word, trained on Model 1
    - Number of EM iterations: 3
    - Available [here](https://cmu.box.com/s/5o61qod7ut6q5hwdv9d64v9w8ryes38b)

### When importing pickle file..

Example code would be like this:

```python
import cPickle as pickle
from collections import defaultdict

def dd():
    return defaultdict(float)

with open("path/to/file", "rb") as f:
    prob = pickle.load(f)
```

**Since the pickled data relies on the dd function above, import or define `dd` function first, and load it.**
