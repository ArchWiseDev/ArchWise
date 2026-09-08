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
    "of", "and", "or", "that", "this", "do", "does", "did", "i"
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

class ArchWiseEngine:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.responses = []
        self.doc_len = []
        self.avgdl = 0.0
        self.vocab = set()
        self.idf = {}
        self.doc_freqs = []
        self.last_query = ""
        self.assistant_fallbacks = [
            "I haven't indexed that specific concept yet. Feel free to ask about grammar rules, science, arithmetic, or writing tools!",
            "I don't have enough verified information on that topic yet. Could you try rephrasing or asking another question?",
            "That concept falls outside my current baseline training."
        ]

    def _tokenize(self, text):
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = [stem_word(w) for w in clean.split() if w and w not in STOPWORDS]
        return tokens

    def _generate_essay(self, prompt):
        match = re.search(r"\b(?:write\s+(?:an?\s+)?essay(?:\s+on|\s+about)?)\s*(.*)", prompt, re.IGNORECASE)
        if not match:
            return None
        topic = match.group(1).strip()
        if not topic:
            topic = "the Importance of Knowledge and Learning"

        clean_topic = topic.strip("?.!")
        return (
            f"### Essay: The Significance of {clean_topic.title()}\n\n"
            f"**Introduction**\n"
            f"In the modern world, **{clean_topic}** plays a pivotal role in shaping ideas, systems, and human understanding. "
            f"Examining this subject reveals not only its core principles, but also the broader implications it holds for society, science, and intellect.\n\n"
            f"**Core Analysis**\n"
            f"At its foundation, {clean_topic} functions as a dynamic framework. When analyzed systematically, "
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

        with open(corpus_path, "r", encoding="utf-8") as f:
            full_text = f.read()

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
                tokens = [stem_word(w) for w in clean_words if w not in STOPWORDS]
                if tokens:
                    self.documents.append(tokens)
                    self.responses.append(reply_text)

        self.doc_len = [len(doc) for doc in self.documents]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1.0
        self.doc_freqs = [Counter(doc) for doc in self.documents]

        total_docs = len(self.documents)
        df = Counter()
        for doc in self.documents:
            for w in set(doc):
                df[w] += 1
                self.vocab.add(w)

        # Probabilistic BM25 IDF
        self.idf = {w: math.log((total_docs - df[w] + 0.5) / (df[w] + 0.5) + 1.0) for w in self.vocab}

    def _bm25_score(self, query_tokens, doc_idx):
        score = 0.0
        doc_freq = self.doc_freqs[doc_idx]
        dl = self.doc_len[doc_idx]

        for token in query_tokens:
            if token not in doc_freq:
                continue
            freq = doc_freq[token]
            idf = self.idf.get(token, 0.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            score += idf * (freq * (self.k1 + 1.0)) / denom
        return score

    def save(self, filepath="model.json"):
        data = {
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "doc_len": self.doc_len,
            "vocab": list(self.vocab),
            "idf": self.idf,
            "responses": self.responses,
            "documents": self.documents
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filepath="model.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.k1 = data["k1"]
        self.b = data["b"]
        self.avgdl = data["avgdl"]
        self.doc_len = data["doc_len"]
        self.vocab = set(data["vocab"])
        self.idf = data["idf"]
        self.responses = data["responses"]
        self.documents = data["documents"]
        self.doc_freqs = [Counter(doc) for doc in self.documents]

    def generate(self, prompt):
        # 1. Essay Generation
        essay_eval = self._generate_essay(prompt)
        if essay_eval:
            return essay_eval

        # 2. Rephrase Engine
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

        # 6. Conversational Follow-up
        clean_input = prompt.strip().lower()
        if clean_input in FOLLOW_UP_TRIGGERS and self.last_query:
            tokens = self._tokenize(f"{self.last_query}")
        else:
            tokens = self._tokenize(prompt)
            if tokens:
                self.last_query = prompt

        if not tokens:
            return "How can I assist you today?"

        best_score = 0.0
        best_idx = -1
        for i in range(len(self.documents)):
            score = self._bm25_score(tokens, i)
            if score > best_score:
                best_score = score
                best_idx = i

        # Strict Relevance Floor: Must have meaningful BM25 score and at least one shared token
        matched_tokens = set(tokens).intersection(set(self.documents[best_idx])) if best_idx != -1 else set()
        if best_score < 1.2 or not matched_tokens:
            return random.choice(self.assistant_fallbacks)

        return self.responses[best_idx]

if __name__ == "__main__":
    engine = ArchWiseEngine()
    engine.train("corpus.txt")
    engine.save("model.json")
    print("ArchWise Precision Engine trained with BM25 (Levenshtein warping removed).")
