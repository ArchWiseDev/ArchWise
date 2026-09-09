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

def extract_subwords(word, min_n=3, max_n=5):
    """Deconstructs words into character n-grams to deduce meanings of unknown words."""
    w = f"<{word}>"
    subwords = []
    length = len(w)
    for n in range(min_n, min(max_n + 1, length + 1)):
        for i in range(length - n + 1):
            subwords.append(w[i:i + n])
    return subwords

class ArchWiseEngine:
    def __init__(self, dim=64):
        self.dim = dim
        self.subword_vectors = {}
        self.doc_embeddings = []
        self.responses = []
        self.raw_patterns = []
        self.last_query = ""
        self.assistant_fallbacks = [
            "I'm analyzing the context of your query, but I don't have a confident deduction for that topic yet.",
            "That concept falls outside my current baseline parameters, though I am analyzing its linguistic structure.",
            "I couldn't derive sufficient semantic confidence for that statement. Could you provide additional context?"
        ]

    def _get_word_vector(self, word):
        """Generates an embedding for known OR unknown words via subword synthesis."""
        subwords = extract_subwords(word)
        vec = [0.0] * self.dim
        found = 0
        for sw in subwords:
            if sw in self.subword_vectors:
                sw_vec = self.subword_vectors[sw]
                for d in range(self.dim):
                    vec[d] += sw_vec[d]
                found += 1

        if found == 0:
            # Fallback pseudorandom stable projection based on hash for completely novel tokens
            seed = sum(ord(c) for c in word)
            random.seed(seed)
            return [random.uniform(-0.1, 0.1) for _ in range(self.dim)]

        # Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _sentence_embedding(self, text):
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        words = [w for w in clean.split() if w and w not in STOPWORDS]
        if not words:
            return [0.0] * self.dim

        doc_vec = [0.0] * self.dim
        for w in words:
            w_vec = self._get_word_vector(w)
            for d in range(self.dim):
                doc_vec[d] += w_vec[d]

        norm = math.sqrt(sum(v * v for v in doc_vec))
        if norm > 0:
            doc_vec = [v / norm for v in doc_vec]
        return doc_vec

    def _cosine_similarity(self, vec_a, vec_b):
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def train(self, corpus_path="corpus.txt"):
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

        # Collect subword frequencies
        for pat in self.raw_patterns:
            words = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", pat.lower()).split() if w]
            for w in words:
                for sw in extract_subwords(w):
                    all_subwords[sw] += 1

        # Initialize subword coordinate embeddings
        random.seed(42)
        for sw in all_subwords:
            self.subword_vectors[sw] = [random.uniform(-0.5, 0.5) for _ in range(self.dim)]

        # Precompute document vectors
        self.doc_embeddings = [self._sentence_embedding(pat) for pat in self.raw_patterns]

    def _generate_essay(self, prompt):
        match = re.search(r"\b(?:write\s+(?:an?\s+)?essay(?:\s+on|\s+about)?)\s*(.*)", prompt, re.IGNORECASE)
        if not match:
            return None
        topic = match.group(1).strip() or "the Importance of Knowledge and Learning"
        clean_topic = topic.strip("?.!")
        return (
            f"### Essay: The Significance of {clean_topic.title()}\n\n"
            f"**Introduction**\n"
            f"In the modern world, **{clean_topic}** plays a pivotal role in shaping ideas, systems, and human understanding.\n\n"
            f"**Core Analysis**\n"
            f"At its foundation, {clean_topic} functions as a dynamic framework. When analyzed systematically, "
            f"it demonstrates how interconnected concepts collaborate to create functional order.\n\n"
            f"**Conclusion**\n"
            f"Ultimately, {clean_topic} is an essential catalyst for advancement. "
            f"Continued exploration ensures deeper comprehension."
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

        reconstructed = "".join([t if re.match(r"[^\w\s]", t) else " " + t for t in rephrased_words]).strip()
        if not modified:
            return f"**Original**: *\"{target}\"*\n\n**Rephrased**: *No direct synonym matches found in local lexicon.*"
        return f"**Original**: *\"{target}\"*\n\n**Rephrased**: *\"{reconstructed}\"*"

    def _summarize(self, text):
        match = re.match(r"^summarize:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        body = match.group(1).strip()
        sentences = [s.strip() for s in re.split(r"[.!?]+", body) if s.strip()]
        if len(sentences) <= 1:
            return f"**Summary**: {body}"
        return f"**Summary**:\n" + "\n".join([f"- {s}." for s in sentences[:2]])

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
            json.dump(data, f)

    def load(self, filepath="model.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.dim = data["dim"]
        self.subword_vectors = data["subword_vectors"]
        self.doc_embeddings = data["doc_embeddings"]
        self.responses = data["responses"]
        self.raw_patterns = data["raw_patterns"]

    def generate(self, prompt):
        essay = self._generate_essay(prompt)
        if essay:
            return essay

        rephrase = self._rephrase(prompt)
        if rephrase:
            return rephrase

        summary = self._summarize(prompt)
        if summary:
            return summary

        grammar = self._correct_grammar(prompt)
        if grammar:
            return grammar

        arithmetic = self._try_arithmetic(prompt)
        if arithmetic:
            return arithmetic

        clean_input = prompt.strip().lower()
        if clean_input in FOLLOW_UP_TRIGGERS and self.last_query:
            query_text = f"{self.last_query}"
        else:
            query_text = prompt
            self.last_query = prompt

        query_vec = self._sentence_embedding(query_text)
        if sum(query_vec) == 0:
            return random.choice(self.assistant_fallbacks)

        best_score = -1.0
        best_idx = -1
        for i, dvec in enumerate(self.doc_embeddings):
            sim = self._cosine_similarity(query_vec, dvec)
            if sim > best_score:
                best_score = sim
                best_idx = i

        # Dense embedding confidence threshold
        if best_score < 0.45:
            return random.choice(self.assistant_fallbacks)

        return self.responses[best_idx]

if __name__ == "__main__":
    engine = ArchWiseEngine(dim=64)
    engine.train("corpus.txt")
    engine.save("model.json")
    print(f"ArchWise Dense Vector Space compiled with {len(engine.subword_vectors)} subword embeddings.")
