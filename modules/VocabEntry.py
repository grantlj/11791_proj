#   the utility class for vocabulary
from collections import Counter
from itertools import chain

class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 0
        self.word2id['<unk>'] = 0

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    #   this is another thing we want
    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    #   this is the thing we want.
    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self,sent_ids):
        ret_words=[]
        for ind in sent_ids:
            ret_words.append(self.id2word[ind])
        return ret_words

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2, lower_case=False,whitelist=None):
        vocab_entry = VocabEntry()

        if lower_case:
            for sent_id in range(0, len(corpus)):
                for w_id in range(0, len(corpus[sent_id])):
                    corpus[sent_id][w_id] = corpus[sent_id][w_id].lower()

        #   word freq less than something, just go away
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]

        #   add while list
        for token in whitelist:
            if not token in top_k_words:
                top_k_words.append(token)

        for word in top_k_words:
            vocab_entry.add(word)


        return vocab_entry