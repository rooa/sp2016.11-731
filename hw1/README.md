### Run code

```
$ python align_model1.py -r 5 > output.txt
```

### Details

- `data/english_most_freq_translated_pair_500.txt`
    - 500 most frequent words in English, translated to German
- `data/german_most_freq_translated_pair_500.txt`
    - 500 most frequent words in German, translated to English
- `outputs/`
    - contains previous outputs for backup
- `align_model1.py`
    - IBM Model1 implementation

### Resource

- [Probability of a English word given a German word](https://cmu.box.com/s/b8va7hp45dkk0qktn9d9e1oy1jklu4x8)

```python
import cPickle as pickle
from collections import defaultdict

def dd():
    return defaultdict(float)

with open("path/to/file", "rb") as f:
    prob = pickle.load(f)
```

Since the pickled data relies on a particular function, import or define `dd` function first, and load it.
