import re
from nltk.corpus import stopwords

class WordSeparator:
    """
    Separate joined words based on Viterbi algorithm
    """

    def __init__(self, word_count: dict, stopword_lang='english'):
        self.word_count = word_count
        self.stop_words = set(s for s in stopwords.words(stopword_lang) if len(s) > 1)
        self.max_word_length = max(map(len, word_count))
        self.total = float(sum(word_count.values()))

    def word_prob(self, word):
        if word in self.word_count:
            return self.word_count.get(word) / self.total
        elif word in self.stop_words:
            return 1
        else:
            return 0

    def words(self, text): return re.findall('[a-z]+', text.lower())

    def separate(self, text):
        probs, lasts = [1.0], [0]

        for i in range(1, len(text) + 1):
            prob_k, k = max((probs[j] * self.word_prob(text[j:i]), j)
                            for j in range(max(0, i - self.max_word_length), i))
            probs.append(prob_k)
            lasts.append(k)
        words = []
        i = len(text)

        while 0 < i:
            words.append(text[lasts[i]:i])
            i = lasts[i]
        words.reverse()

        return words, probs[-1]