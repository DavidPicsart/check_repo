import collections
import datetime
import re
from multiprocessing import Pool
import nltk
import numpy as np
import pandas as pd
from icecream import ic
from nltk.corpus import stopwords
from tqdm import tqdm
from .evaluate import dist
from .evaluate import sim


class Data:
    """A set of static functions used for text data manipulation and processing"""

    moderators_tags_data = None
    whitelist_data = None

    @staticmethod
    def contains_non_ascii(s: str) -> bool:
        """Checks if there are non-ascii characters in text"""
        return any(ord(i) > 127 for i in s)

    @staticmethod
    def parallelize_dataframe(df, func, num_partitions=1, num_cores=1):
        """Parallelize dataframe with several processes on a dataframe"""
        df_split = np.array_split(df, num_partitions)
        pool = Pool(num_cores)
        df = pd.concat(pool.map(func, df_split), ignore_index=True)
        pool.close()
        pool.join()

        return df

    @staticmethod
    def compute_recency(data_slc):
        """Cumpute recency weight of the query"""
        current_date = datetime.datetime.now()
        data_slc['age'] = data_slc['date'].apply(
            lambda x: (current_date -
                       datetime.datetime.strptime(x, '%Y-%m-%d')).days
        )
        data_slc['age'] = data_slc['age'] - data_slc['age'].min()
        return data_slc

    @staticmethod
    def concat_recency(data_slc):
        """Concatinate recency weights to queries"""
        data_slc['text'] = data_slc.apply(
            lambda x: list(map(
                lambda y: str(x['age']) + '_' + str(y),
                filter(None, re.split('[^a-zA-Z0-9][ ]?', x['text']))
            )),
            axis=1
        )

        return data_slc

    @staticmethod
    def create_syntaxtic_contextual_mapping(dictionary, model, sim_threshold=0.4):
        """Create mapping of semantically and syntaxtically similar words from dictionary"""

        def simm(x, y):
            sm = sim(x, y, dictionary, model)

            return sm if max(len(x), len(y)) < 7 else sm + 0.1

        def distt(x, y):
            return (dist(x, y) == 1 and max(len(x), len(y)) <= 7) or \
                   (dist(x, y) <= 2 and max(len(x), len(y)) == 8) or \
                   (dist(x, y) <= 3 and max(len(x), len(y)) >= 9)

        sim_words = {}

        ic(' -  [*] Merging close words from dictionary')
        cleared_words = []
        dkeys = list(dictionary.keys())

        for i, nw in tqdm(enumerate(dkeys)):
            for cw in cleared_words:
                if nw in dictionary and cw in dictionary \
                        and simm(nw, cw) > sim_threshold and distt(nw, cw):
                    sim_words[i] = cw

                    break
            else:
                cleared_words.append(nw)

        return sim_words

    @staticmethod
    def build_dataset(words, n_words, dictionary=None, date_weight=False):
        """Build dictionary, reverse dictionary, word frequency and indexed coprpus"""

        count = [['UNK', -1]]

        if date_weight:
            count.extend(collections.Counter(
                w.split('_')[1] for w in words).most_common(n_words - 1))
        else:
            count.extend(collections.Counter(
                words).most_common(n_words - 1))

        if not dictionary:
            ic(' -  [*] Creating Dictionary...')
            dictionary = dict()

            for word, _ in count:
                dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0

        for word in tqdm(words):
            w = word.split('_')[1] if date_weight else word

            if w in dictionary:
                index = dictionary[w]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            d = word.split('_')[0] + '_' + \
                str(index) if date_weight else str(index)
            data.append(d)
        print('Unknown words: %s' % unk_count)
        print('Corpus length: %s' % len(data))

        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        assert (len(dictionary) == len(reversed_dictionary))

        return data, count, dictionary, reversed_dictionary

    @staticmethod
    def is_valid_word(word, stop_words=None, whitelist=None, moderators_tags=None):
        """Checks if the word is valid for training or inference"""

        if whitelist and word in whitelist:
            return True

        if moderators_tags and word in moderators_tags:
            return False

        if stop_words and word in stop_words:
            return False

        if Data.contains_non_ascii(word):
            return False

        if len(word) <= 2:
            return False

        if word.isdigit():
            return False

        return True

    @staticmethod
    def clean_text(texts, lower=True):
        """Cleans given text from not valid words and characters"""
        tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w^~]+')
        cleaned_texts = []
        for text in texts:
            words = tokenizer.tokenize(text)
            stop_words = set(stopwords.words('english'))
            cleaned_text = [word for word in words if
                            Data.is_valid_word(word,
                                               stop_words=stop_words,
                                               whitelist=Data.whitelist_data,
                                               moderators_tags=Data.moderators_tags_data)]

            cleaned_texts.append(' '.join(cleaned_text) if cleaned_text else '')
        return pd.Series(cleaned_texts)

    @staticmethod
    def collect_data(raw_data, vocabulary_size=10000, lower=False,
                     dictionary=None, phrases=False, date_weight=False, dictionary_to_remove_sim=None,
                     model_to_remove_sim=None):
        """Tokenizes data and passes for dictionary generation"""

        if lower:
            raw_data = raw_data.lower()
        tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w^~]+')
        vocabulary = tokenizer.tokenize(raw_data)

        data, count, dictionary, reverse_dictionary = Data.build_dataset(vocabulary,
                                                                         vocabulary_size,
                                                                         dictionary=dictionary,
                                                                         date_weight=date_weight)

        return data, count, dictionary, reverse_dictionary
