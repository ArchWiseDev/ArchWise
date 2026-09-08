import re
import math
import json
import random
from collections import Counter

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20,
    "hundred": 100
}

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "can", "you", "tell", "me", "please", "would", "could", "is", "it",
    "of", "and", "or", "that", "this"
}

GRAMMAR_PATTERNS = [
    (r"\b(he|she|it)\s+go\b", r"\1 goes"),
    (r"\b(he|she|it)\s+dont\b", r"\1 doesn't"),
    (r"\b(he|she|it)\s+does\s+not\s+has\b", r"\1 does not have"),
    (r"\b(they|we|you)\s+is\b", r"\1 are"),
    (r"\b(they|we|you)\s+was\b", r"\1 were"),
    (r"\b(i)\s+is\b", r"I am"),
    (r"\b(i)\s+are\b", r"I am"),
    (r"\btheir\s+(going|coming|here|running)\b", r"they're \1"),
    (r"\byour\s+(welcome|right|wrong)\b", r"you're \1"),
    (r"\bcould\s+of\b", r"could have"),
    (r"\bshould\s+of\b", r"should have"),
    (r"\bwould\s+of\b", r"would have"),
    (r"\ba\s+([aeiou]\w+)", r"an \1"),
    (r"\ban\s+([^aeiou\s]\w+)", r"a \1"),
]

def stem_word(w):
    suffixes = ("ing", "ly", "ed", "ous", "ies", "es", "s", "ment")
    for s in suffixes:
        if w.endswith(s) and len(w) > len(s) + 2:
            return w[:-len(s)]
    return w

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class ArchWiseEngine:
    def __init__(self):
        self.documents = []
        self.responses = []
        self.vocab = {}
        self.raw_vocab = set()
        self.idf = {}
        self.doc_vectors = []
        self.assistant_fallbacks = [
            "I haven't indexed that specific concept yet. You can ask me about grammar rules, language mechanics, or mathematics!",
            "I'm not certain how to answer that with my current knowledge. Try rephrasing or asking for a grammatical explanation.",
            "That query is outside my current trained parameters, but I am continuously learning."
        ]

    def _tokenize(self, text):
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        raw_tokens = [w for w in clean.split() if w]
        corrected = []
        for w in raw_tokens:
            if w in self.raw_vocab or len(w) < 4:
                corrected.append(w)
            else:
                closest = min(self.raw_vocab, key=lambda target: levenshtein(w, target)) if self.raw_vocab else w
                if levenshtein(w, closest) <= 2:
                    corrected.append(closest)
                else:
                    corrected.append(w)
        return [stem_word(w) for w in corrected if w not in STOPWORDS]

    def _summarize(self, text):
        match = re.match(r"^summarize:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        body = match.group(1).strip()
        sentences = [s.strip() for s in re.split(r"[.!?]+", body) if s.strip()]
        if len(sentences) <= 1:
            return f"**Summary**: {body}"

        # Score sentences by non-stopword token count
        word_counts = Counter([stem_word(w.lower()) for w in re.findall(r"\b\w+\b", body) if w.lower() not in STOPWORDS])
        scored = []
        for s in sentences:
            tokens = [stem_word(w.lower()) for w in re.findall(r"\b\w+\b", s) if w.lower() not in STOPWORDS]
            score = sum(word_counts[t] for t in tokens) / (len(tokens) + 1)
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [item[1] for item in scored[:2]]
        return f"**Summary**:\n" + "\n".join([f"- {s}." for s in top_sentences])

    def _correct_grammar(self, text):
        trigger = re.match(r"^(?:fix|correct|proofread|grammar check):\s*(.*)", text, re.IGNORECASE)
        if not trigger:
            return None
        target = trigger.group(1).strip()
        corrected = target
        for pattern, replacement in GRAMMAR_PATTERNS:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

        if corrected:
            corrected = corrected[0].upper() + corrected[1:]
            if not corrected.endswith((".", "!", "?")):
                corrected += "."

        return f"**Original**: *\"{target}\"*\n\n**Corrected**: *\"{corrected}\"*"

    def _try_arithmetic(self, text):
        norm = text.lower().replace("what's", "what is").replace("whats", "what is")
        tokens = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", norm).split() if w]
        converted = [str(WORD_NUMBERS[t]) if t in WORD_NUMBERS else t for t in tokens]
        reconstructed = " ".join(converted)

        match_add = re.search(r"(\d+)\s*(?:\+|\bplus\b)\s*(\d+)", reconstructed)
        if match_add:
            a, b = int(match_add.group(1)), int(match_add.group(2))
            return f"{a} + {b} = **{a + b}**"

        match_sub = re.search(r"(\d+)\s*(?:\-|\bminus\b)\s*(\d+)", reconstructed)
        if match_sub:
            a, b = int(match_sub.group(1)), int(match_sub.group(2))
            return f"{a} - {b} = **{a - b}**"

        match_mul = re.search(r"(\d+)\s*(?:\*|\btimes\b|\bmultiplied by\b)\s*(\d+)", reconstructed)
        if match_mul:
            a, b = int(match_mul.group(1)), int(match_mul.group(2))
            return f"{a} × {b} = **{a * b}**"

        match_div = re.search(r"(\d+)\s*(?:\/|\bdivided by\b)\s*(\d+)", reconstructed)
        if match_div:
            a, b = int(match_div.group(1)), int(match_div.group(2))
            if b == 0:
                return "Division by zero is mathematically undefined."
            return f"{a} / {b} = **{a / b:.2f}**"

        return None

    def train(self, corpus_path="corpus.txt"):
        self.documents = []
        self.responses = []
        self.raw_vocab = set()

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "::" not in line:
                    continue
                patterns_part, reply = line.split("::", 1)
                patterns = [p.strip() for p in patterns_part.split("|") if p.strip()]
                reply = reply.strip().replace(r"\n", "\n")
                for pat in patterns:
                    clean_words = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", pat.lower()).split() if w]
                    self.raw_vocab.update(clean_words)
                    tokens = [stem_word(w) for w in clean_words if w not in STOPWORDS]
                    if tokens:
                        self.documents.append(tokens)
                        self.responses.append(reply)

        total_docs = len(self.documents)
        df = Counter()
        all_stems = set()
        for doc in self.documents:
            for w in set(doc):
                df[w] += 1
                all_stems.add(w)

        self.vocab = {stem: idx for idx, stem in enumerate(sorted(list(all_stems)))}
        self.idf = {stem: math.log((1.0 + total_docs) / (1.0 + df[stem])) + 1.0 for stem in self.vocab}
        self.doc_vectors = [self._vectorize(doc) for doc in self.documents]

    def _vectorize(self, tokens):
        tf = Counter(tokens)
        vec = [0.0] * len(self.vocab)
        for stem, count in tf.items():
            if stem in self.vocab:
                idx = self.vocab[stem]
                w_tf = 1.0 + math.log(count) if count > 0 else 0.0
                vec[idx] = w_tf * self.idf[stem]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _cosine_similarity(self, vec_a, vec_b):
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def save(self, filepath="model.json"):
        data = {
            "vocab": self.vocab,
            "raw_vocab": list(self.raw_vocab),
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
        self.raw_vocab = set(data["raw_vocab"])
        self.idf = data["idf"]
        self.responses = data["responses"]
        self.doc_vectors = data["doc_vectors"]

    def generate(self, prompt):
        # 1. Summarization
        summary = self._summarize(prompt)
        if summary:
            return summary

        # 2. Grammar Correction Check
        grammar_eval = self._correct_grammar(prompt)
        if grammar_eval:
            return grammar_eval

        # 3. Arithmetic Check
        calc_result = self._try_arithmetic(prompt)
        if calc_result:
            return calc_result

        # 4. Semantic Search
        tokens = self._tokenize(prompt)
        if not tokens:
            return "How can I assist you today?"

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

        if best_score < 0.18:
            return random.choice(self.assistant_fallbacks)

        return self.responses[best_idx]

if __name__ == "__main__":
    engine = ArchWiseEngine()
    engine.train("corpus.txt")
    engine.save("model.json")
    print(f"ArchWise v0.3 Compiled: Multi-task engine active across {len(engine.documents)} patterns.")
