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

SYNONYMS = {
    "fast": ["rapid", "swift", "quick"],
    "slow": ["gradual", "unhurried", "sluggish"],
    "big": ["large", "massive", "substantial"],
    "small": ["compact", "tiny", "diminutive"],
    "good": ["excellent", "favorable", "effective"],
    "bad": ["flawed", "suboptimal", "deficient"],
    "important": ["crucial", "essential", "vital"],
    "difficult": ["complex", "challenging", "demanding"],
    "easy": ["straightforward", "simple", "effortless"],
    "help": ["assist", "support", "aid"],
    "build": ["construct", "develop", "assemble"],
    "learn": ["acquire", "grasp", "absorb"]
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

FOLLOW_UP_TRIGGERS = {"more", "explain", "why", "elaborate", "continue", "detail", "tell me more"}

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
        self.last_query = ""
        self.assistant_fallbacks = [
            "I haven't indexed that specific concept yet. You can ask me to write an essay, explain grammar, calculate numbers, or summarize text!",
            "I'm not certain how to answer that with my current knowledge. Try rephrasing or asking for a grammatical explanation.",
            "That query falls outside my current baseline parameters, but I am continuously expanding my vocabulary."
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

    def _generate_essay(self, prompt):
        match = re.search(r"\b(?:write\s+(?:an?\s+)?essay(?:\s+on|\s+about)?)\s*(.*)", prompt, re.IGNORECASE)
        if not match:
            return None
        topic = match.group(1).strip()
        if not topic:
            topic = "the Importance of Language and Learning"

        clean_topic = topic.strip("?.!")
        return (
            f"### Essay: The Significance of {clean_topic.title()}\n\n"
            f"**Introduction**\n"
            f"In the modern world, **{clean_topic}** plays a pivotal role in shaping ideas, systems, and human understanding. "
            f"Examining this subject reveals not only its core principles, but also the broader implications it holds for society, science, and intellect.\n\n"
            f"**Core Analysis**\n"
            f"At its foundation, {clean_topic} functions as a dynamic framework. When analyzed through first principles, "
            f"it demonstrates how interconnected concepts collaborate to create functional order. "
            f"Whether through structured systems, clear rules, or continuous iteration, the underlying mechanics drive consistent progress and clarity.\n\n"
            f"Furthermore, understanding {clean_topic} allows us to avoid common fallacies and superficial assumptions. "
            f"By studying its nuances, practitioners and thinkers can optimize their methods and construct sustainable, reliable outcomes.\n\n"
            f"**Conclusion**\n"
            f"Ultimately, {clean_topic} is more than an isolated phenomenon; it is an essential catalyst for advancement. "
            f"Continued dedication to exploring, refining, and applying its lessons ensures meaningful growth and deeper comprehension."
        )

    def _rephrase(self, text):
        match = re.match(r"^rephrase:\s*(.*)", text, re.IGNORECASE)
        if not match:
            return None
        target = match.group(1).strip()
        words = re.findall(r"\b\w+\b|[^\w\s]", target)
        rephrased_words = []
        modified = False

        for word in words:
            lower = word.lower()
            if lower in SYNONYMS:
                replacement = random.choice(SYNONYMS[lower])
                if word[0].isupper():
                    replacement = replacement.capitalize()
                rephrased_words.append(replacement)
                modified = True
            else:
                rephrased_words.append(word)

        reconstructed = ""
        for token in rephrased_words:
            if re.match(r"[^\w\s]", token):
                reconstructed = reconstructed.rstrip() + token + " "
            else:
                reconstructed += token + " "

        reconstructed = reconstructed.strip()
        if not modified:
            return f"**Original**: *\"{target}\"*\n\n**Rephrased**: *No direct synonym matches found in local lexicon, structure maintained.*"
        return f"**Original**: *\"{target}\"*\n\n**Rephrased**: *\"{reconstructed}\"*"

    def _summarize(self, text):
        match = re.match(r"^summarize:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        body = match.group(1).strip()
        sentences = [s.strip() for s in re.split(r"[.!?]+", body) if s.strip()]
        if len(sentences) <= 1:
            return f"**Summary**: {body}"

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
            full_text = f.read()

        # Split entries by double-colon delimiters cleanly across newlines
        blocks = full_text.split("\n")
        current_patterns = None
        current_reply = []

        for line in blocks:
            line_str = line.strip()
            if "::" in line_str:
                if current_patterns and current_reply:
                    reply_text = "\n".join(current_reply).strip()
                    for pat in current_patterns:
                        clean_words = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", pat.lower()).split() if w]
                        self.raw_vocab.update(clean_words)
                        tokens = [stem_word(w) for w in clean_words if w not in STOPWORDS]
                        if tokens:
                            self.documents.append(tokens)
                            self.responses.append(reply_text)
                parts = line_str.split("::", 1)
                current_patterns = [p.strip() for p in parts[0].split("|") if p.strip()]
                current_reply = [parts[1].strip()]
            elif current_patterns:
                if line_str:
                    current_reply.append(line_str)

        if current_patterns and current_reply:
            reply_text = "\n".join(current_reply).strip()
            for pat in current_patterns:
                clean_words = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", pat.lower()).split() if w]
                self.raw_vocab.update(clean_words)
                tokens = [stem_word(w) for w in clean_words if w not in STOPWORDS]
                if tokens:
                    self.documents.append(tokens)
                    self.responses.append(reply_text)

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
        # 1. Dynamic Essay Generator
        essay_eval = self._generate_essay(prompt)
        if essay_eval:
            return essay_eval

        # 2. Paraphrase Generator
        rephrase_eval = self._rephrase(prompt)
        if rephrase_eval:
            return rephrase_eval

        # 3. Summarization Check
        summary = self._summarize(prompt)
        if summary:
            return summary

        # 4. Grammar Check
        grammar_eval = self._correct_grammar(prompt)
        if grammar_eval:
            return grammar_eval

        # 5. Arithmetic Check
        calc_result = self._try_arithmetic(prompt)
        if calc_result:
            return calc_result

        # 6. Contextual Follow-up
        clean_input = prompt.strip().lower()
        if clean_input in FOLLOW_UP_TRIGGERS and self.last_query:
            tokens = self._tokenize(f"{self.last_query} details")
        else:
            tokens = self._tokenize(prompt)
            if tokens:
                self.last_query = prompt

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
    print(f"ArchWise engine recompiled with multiline preserving and essay generation support.")
