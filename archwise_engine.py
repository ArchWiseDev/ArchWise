import re
import math
import json
import random
from collections import Counter

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20
}

class ArchWiseEngine:
    def __init__(self):
        self.documents = []
        self.responses = []
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []
        self.assistant_fallbacks = [
            "I'm not completely sure about that. Could you rephrase or ask in another way?",
            "I don't have enough information on that topic at the moment. What else can I help you with?",
            "I didn't quite catch that. Feel free to provide more context so I can assist you better."
        ]

    def _tokenize(self, text):
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return [w for w in clean.split() if w]

    def _try_arithmetic(self, text):
        norm = text.lower().replace("what's", "what is").replace("whats", "what is")
        tokens = self._tokenize(norm)
        
        converted = []
        for t in tokens:
            if t in WORD_NUMBERS:
                converted.append(str(WORD_NUMBERS[t]))
            else:
                converted.append(t)
        
        reconstructed = " ".join(converted)
        
        # Addition
        match_add = re.search(r"(\d+)\s*(?:\+|\bplus\b)\s*(\d+)", reconstructed)
        if match_add:
            a, b = int(match_add.group(1)), int(match_add.group(2))
            return f"{a} + {b} = **{a + b}**"

        # Subtraction
        match_sub = re.search(r"(\d+)\s*(?:\-|\bminus\b)\s*(\d+)", reconstructed)
        if match_sub:
            a, b = int(match_sub.group(1)), int(match_sub.group(2))
            return f"{a} - {b} = **{a - b}**"

        # Multiplication
        match_mul = re.search(r"(\d+)\s*(?:\*|\btimes\b|\bmultiplied by\b)\s*(\d+)", reconstructed)
        if match_mul:
            a, b = int(match_mul.group(1)), int(match_mul.group(2))
            return f"{a} × {b} = **{a * b}**"

        return None

    def train(self, corpus_path="corpus.txt"):
        self.documents = []
        self.responses = []

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "::" not in line:
                    continue
                patterns_part, reply = line.split("::", 1)
                patterns = [p.strip() for p in patterns_part.split("|") if p.strip()]
                reply = reply.strip().replace(r"\n", "\n")
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
        calc_result = self._try_arithmetic(prompt)
        if calc_result:
            return calc_result

        tokens = self._tokenize(prompt)
        if not tokens:
            return "How can I help you today?"

        query_vec = self._vectorize(tokens)
        if sum(query_vec) == 0:
            return random.choice(self.assistant_fallbacks)

        best_score = -1.0
        best_idx = -1
        for i, dvec in enumerate(self.doc_vectors):
            sim = self._cosine_similarity(query_vec, dvec)
            if sim > best_score:
                best_score = sim
                best_idx = i

        if best_score < 0.2:
            return random.choice(self.assistant_fallbacks)

        return self.responses[best_idx]

if __name__ == "__main__":
    engine = ArchWiseEngine()
    engine.train("corpus.txt")
    engine.save("model.json")
    print("ArchWise Knowledge Base compiled successfully.")
