import numpy as np
from scipy.sparse import bsr_matrix


class BagWord(object):
    def __init__(self):
        self.word_set = set()
        self.word_index = None

    def add_line(self, line):
        words = line.split()
        for word in words:
            self.word_set.add(word)

    def transform(self):
        self.word_index = {v: k for k, v in enumerate(self.word_set)}

    def transform_line(self, line):
        if not self.word_index:
            self.transform()
        line_vector = np.zeros((len(self.word_index)))
        # skip word if not seen
        indices = [self.word_index.get(w) for w in line.split() if w in self.word_index]
        line_vector[indices] = 1
        return line_vector

    def transform_lines(self, lines):
        if not self.word_index:
            self.transform()
        x = np.zeros((len(lines), len(self.word_index)))
        for i in range(len(lines)):

            x[i, :] = np.squeeze(self.transform_line(lines[i]))
        return x