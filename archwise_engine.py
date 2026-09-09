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

CASUAL_MARKERS = {"yo", "bruh", "nah", "yeah", "gimme", "wanna", "gonna", "sup", "lol", "dude", "hey"}
FORMAL_MARKERS = {"furthermore", "consequently", "regarding", "therefore", "clarify", "synthesize", "analyze"}

def extract_subwords(word, min_n=3, max_n=5):
    w = f"<{word}>"
    subwords = []
    length = len(w)
    for n in range(min_n, min(max_n + 1, length + 1)):
        for i in range(length - n + 1):
            subwords.append(w[i:i + n])
    return subwords

class ArchWiseEngine:
    def __init__(self, dim=32):
        self.dim = dim
        self.subword_vectors = {}
        self.doc_embeddings = []
        self.responses = []
        self.raw_patterns = []
        self.lexicon = {}
        self.last_query = ""

    def _detect_persona(self, text):
        clean = text.lower()
        words = set(re.findall(r"\b\w+\b", clean))
        if words.intersection(CASUAL_MARKERS) or (len(words) <= 3 and not words.intersection(FORMAL_MARKERS)):
            return "casual"
        elif words.intersection(FORMAL_MARKERS) or len(text.split()) > 10:
            return "formal"
        elif clean.count("!") >= 2 or clean.isupper():
            return "energetic"
        return "neutral"

    def _adapt_tone(self, response, persona):
        if persona == "casual":
            clean = response.replace("Furthermore, ", "").replace("Ultimately, ", "")
            return f"Got it. {clean}"
        elif persona == "formal":
            return f"Regarding your inquiry:\n\n{response}"
        elif persona == "energetic":
            return f"{response} Let me know if you want to explore further!"
        return response

    def _lookup_lexicon(self, prompt):
        # Match "what is X", "define X", "meaning of X"
        m = re.search(r"\b(?:what\s+is|what\s+are|define|meaning\s+of)\s+([a-zA-Z]+)\b", prompt.lower())
        if m:
            target = m.group(1)
            if target in self.lexicon:
                return f"**{target.title()}**: {self.lexicon[target]}"
        return None

    def _sentence_embedding(self, text):
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        words = [w for w in clean.split() if w and w not in STOPWORDS]
        if not words:
            return [0.0] * self.dim

        doc_vec = [0.0] * self.dim
        for w in words:
            subwords = extract_subwords(w)
            vec = [0.0] * self.dim
            count = 0
            for sw in subwords:
                if sw in self.subword_vectors:
                    sw_vec = self.subword_vectors[sw]
                    for d in range(self.dim):
                        vec[d] += sw_vec[d]
                    count += 1
            if count > 0:
                for d in range(self.dim):
                    doc_vec[d] += vec[d] / count

        norm = math.sqrt(sum(v * v for v in doc_vec))
        if norm > 0:
            doc_vec = [v / norm for v in doc_vec]
        return doc_vec

    def _cosine_similarity(self, vec_a, vec_b):
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def train(self, corpus_path="corpus.txt", lexicon_path="lexicon.json"):
        # Load 15,000-word instant lexicon
        try:
            with open(lexicon_path, "r", encoding="utf-8") as lf:
                self.lexicon = json.load(lf)
        except Exception:
            self.lexicon = {}

        self.raw_patterns = []
        self.responses = []
        self.subword_vectors = {}
        all_subwords = Counter()

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
                        self.raw_patterns.append(pat)
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
                self.raw_patterns.append(pat)
                self.responses.append(reply_text)

        for pat in self.raw_patterns:
            words = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", pat.lower()).split() if w]
            for w in words:
                for sw in extract_subwords(w):
                    all_subwords[sw] += 1

        random.seed(42)
        for sw, _ in all_subwords.most_common(10000):
            self.subword_vectors[sw] = [round(random.uniform(-0.5, 0.5), 4) for _ in range(self.dim)]

        self.doc_embeddings = [[round(val, 4) for val in self._sentence_embedding(pat)] for pat in self.raw_patterns]

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

    def save(self, filepath="model.json"):
        data = {
            "dim": self.dim,
            "subword_vectors": self.subword_vectors,
            "doc_embeddings": self.doc_embeddings,
            "responses": self.responses,
            "raw_patterns": self.raw_patterns
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))

    def load(self, filepath="model.json", lexicon_path="lexicon.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.dim = data["dim"]
        self.subword_vectors = data["subword_vectors"]
        self.doc_embeddings = data["doc_embeddings"]
        self.responses = data["responses"]
        self.raw_patterns = data["raw_patterns"]
        try:
            with open(lexicon_path, "r", encoding="utf-8") as lf:
                self.lexicon = json.load(lf)
        except Exception:
            self.lexicon = {}

    def generate(self, prompt):
        user_persona = self._detect_persona(prompt)

        # 1. Grammar check
        grammar_eval = self._correct_grammar(prompt)
        if grammar_eval:
            return self._adapt_tone(grammar_eval, user_persona)

        # 2. Arithmetic check
        arithmetic = self._try_arithmetic(prompt)
        if arithmetic:
            return self._adapt_tone(arithmetic, user_persona)

        # 3. Fast Lexicon Lookup (15,000 words)
        lex_match = self._lookup_lexicon(prompt)
        if lex_match:
            return self._adapt_tone(lex_match, user_persona)

        # 4. Dense Semantic Embedding Match
        query_vec = self._sentence_embedding(prompt)
        best_score = -1.0
        best_idx = -1
        for i, dvec in enumerate(self.doc_embeddings):
            sim = self._cosine_similarity(query_vec, dvec)
            if sim > best_score:
                best_score = sim
                best_idx = i

        if best_score >= 0.45:
            return self._adapt_tone(self.responses[best_idx], user_persona)

        return self._adapt_tone("I analyzed your input, but I don't have enough verified information on that specific topic yet. Could you rephrase or ask another question?", user_persona)

if __name__ == "__main__":
    engine = ArchWiseEngine(dim=32)
    engine.train("corpus.txt", "lexicon.json")
    engine.save("model.json")
    print(f"ArchWise Two-Tier Engine compiled with {len(engine.lexicon)} lexicon words and {len(engine.raw_patterns)} discourse patterns.")
