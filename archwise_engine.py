import re
import math
import json
from collections import Counter

class ArchWiseEngine:
    def __init__(self):
        self.documents = []
        self.responses = []
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []

    def _tokenize(self, text):
        clean = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        return [w for w in clean.split() if w]

    def train(self, corpus_path="corpus.txt"):
        self.documents = []
        self.responses = []
        raw_lines = []

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "::" not in line:
                    continue
                patterns_part, reply = line.split("::", 1)
                patterns = [p.strip() for p in patterns_part.split("|") if p.strip()]
                reply = reply.strip()
                for pat in patterns:
                    tokens = self._tokenize(pat)
                    if tokens:
                        self.documents.append(tokens)
                        self.responses.append(reply)

        total_docs = len(self.documents)
        df = Counter()
        all_words = set()
        for doc in self.documents:
            unique_words = set(doc)
            for w in unique_words:
                df[w] += 1
                all_words.add(w)

        self.vocab = {word: idx for idx, word in enumerate(sorted(list(all_words)))}
        self.idf = {word: math.log((1 + total_docs) / (1 + df[word])) + 1 for word in self.vocab}

        self.doc_vectors = [self._vectorize(doc) for doc in self.documents]

    def _vectorize(self, tokens):
        tf = Counter(tokens)
        vec = [0.0] * len(self.vocab)
        for word, count in tf.items():
            if word in self.vocab:
                idx = self.vocab[word]
                vec[idx] = count * self.idf[word]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _cosine_similarity(self, vec_a, vec_b):
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def save(self, filepath="model.json"):
        data = {
            "vocab": self.vocab,
            "idf": self.idf,
            "responses": self.responses,
            "doc_vectors": self.doc_vectors
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filepath="model.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.responses = data["responses"]
        self.doc_vectors = data["doc_vectors"]

    def generate(self, prompt):
        tokens = self._tokenize(prompt)
        if not tokens:
            return "Please say something so I can understand."

        query_vec = self._vectorize(tokens)
        if sum(query_vec) == 0:
            return "I have not learned those words yet. You can train me by adding them to corpus.txt!"

        best_score = -1.0
        best_idx = -1
        for i, dvec in enumerate(self.doc_vectors):
            sim = self._cosine_similarity(query_vec, dvec)
            if sim > best_score:
                best_score = sim
                best_idx = i

        if best_score < 0.2:
            return "I am not quite sure what you mean. Could you rephrase that?"

        return self.responses[best_idx]

if __name__ == "__main__":
    engine = ArchWiseEngine()
    engine.train("corpus.txt")
    engine.save("model.json")
    print("ArchWise successfully trained and saved to model.json.")
