import re
import math
import json
import random
import ast
import operator
from collections import Counter

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "hundred": 100
}

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "can", "you", "tell", "me", "please", "would", "could", "is", "it",
    "of", "and", "or", "that", "this", "do", "does", "did", "i"
}

SAFE_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos
}

EMOTION_CHAMBERS = {
    "curiosity": [
        "Questions like this are genuinely fun to pull apart.",
        "Now that is an interesting angle to analyze.",
        "Let's look at the underlying mechanics here:"
    ],
    "dry_wit": [
        "Alright, let's tackle this systematically.",
        "Short answer incoming; let's break it down.",
        "Fair warning, there's a lot of detail under the hood here:"
    ],
    "philosophical": [
        "It's fascinating how much nuance hides behind a seemingly simple query.",
        "Looking at this from first principles reveals a lot about the structure:",
        "Let's deconstruct the core concepts:"
    ]
}

def safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Non-numeric")
    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            if op_type == ast.Div and right == 0:
                raise ZeroDivisionError("Division by zero")
            if op_type == ast.Pow and (right > 100 or left > 10000):
                raise ValueError("Exponent too large")
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError("Unsupported operator")
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError("Unsupported unary")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            fname = node.func.id.lower()
            if fname == "sqrt" and len(node.args) == 1:
                val = safe_eval(node.args[0])
                if val < 0: raise ValueError("Negative sqrt")
                return math.sqrt(val)
            elif fname == "abs" and len(node.args) == 1:
                return abs(safe_eval(node.args[0]))
    raise ValueError("Invalid math syntax")

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
        self.synsets = {}
        self.math_kb = {}
        self.geography = {}
        self.omnibus = {}

    def _spontaneous_emotion_wrapper(self, text, prompt):
        intrigue_words = {"why", "how", "strange", "paradox", "origin", "universe", "secret", "deep", "solve"}
        prompt_words = set(re.findall(r"\b\w+\b", prompt.lower()))
        
        if not (prompt_words.intersection(intrigue_words) or random.random() < 0.25):
            return text

        mood = "curiosity" if prompt_words.intersection(intrigue_words) else random.choice(["dry_wit", "philosophical"])
        prefix = random.choice(EMOTION_CHAMBERS[mood])
        return f"*{prefix}*\n\n{text}"

    def _normalize_prompt(self, prompt):
        norm = prompt.strip().lower()
        norm = re.sub(r"\bwhat['’]?s\b", "what is", norm)
        norm = re.sub(r"\bwho['’]?s\b", "who is", norm)
        norm = re.sub(r"\bwhere['’]?s\b", "where is", norm)
        return norm

    def _lookup_omnibus(self, normalized_prompt):
        m = re.search(r"\b(?:what\s+is|tell\s+me\s+about|about|define)\s+([a-zA-Z0-9\s]+)\b", normalized_prompt)
        target = m.group(1).strip() if m else normalized_prompt
        
        if target in self.omnibus:
            return self.omnibus[target]

        words = target.split()
        for w in words:
            if w in self.omnibus:
                return self.omnibus[w]
        return None

    def _lookup_geography(self, normalized_prompt):
        m_capital = re.search(r"\bcapital\s+of\s+([a-zA-Z\s]+)\b", normalized_prompt)
        if m_capital:
            target = m_capital.group(1).strip()
            if target in self.geography:
                data = self.geography[target]
                return f"The capital of **{data.get('name', target.title())}** is **{data.get('capital', 'Unknown')}**."

        for country_key, data in self.geography.items():
            if re.search(r"\b" + re.escape(country_key) + r"\b", normalized_prompt):
                if "capital" in normalized_prompt:
                    return f"The capital of **{data.get('name', country_key.title())}** is **{data.get('capital', 'Unknown')}**."
                return data.get("summary", "")
        return None

    def _lookup_math_kb(self, prompt):
        clean = prompt.lower().strip()
        for key, val in self.math_kb.items():
            if key in clean:
                return val
        return None

    def _evaluate_expression(self, text):
        norm = text.lower().replace("what is", "").replace("calculate", "").replace("solve", "").strip()
        norm = norm.replace("times", "*").replace("multiplied by", "*").replace("divided by", "/").replace("plus", "+").replace("minus", "-").replace("^", "**")

        tokens = norm.split()
        converted = [str(WORD_NUMBERS[t]) if t in WORD_NUMBERS else t for t in tokens]
        expr = "".join(converted)

        if not re.search(r"[\d\+\-\*\/\^\%]", expr):
            return None

        clean_expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\%\,\s_a-zA-Z]", "", expr).strip()
        try:
            tree = ast.parse(clean_expr, mode="eval")
            res = safe_eval(tree.body)
            if isinstance(res, float) and res.is_integer():
                res = int(res)
            elif isinstance(res, float):
                res = round(res, 6)
            return f"**Result**: `{clean_expr}` = **{res}**"
        except Exception:
            return None

    def _lookup_lexicon(self, normalized_prompt):
        m = re.search(r"\b(?:what\s+is|define|meaning\s+of|who\s+is)\s+([a-zA-Z]+)\b", normalized_prompt)
        if m:
            target = m.group(1).lower()
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

    def train(self, corpus_path="corpus.txt", lexicon_path="lexicon.json", synsets_path="synsets.json", math_path="math_knowledge.json", geo_path="geography.json", omnibus_path="omnibus.json"):
        for path, attr in [(lexicon_path, "lexicon"), (synsets_path, "synsets"), (math_path, "math_kb"), (geo_path, "geography"), (omnibus_path, "omnibus")]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    setattr(self, attr, json.load(f))
            except Exception:
                setattr(self, attr, {})

        self.raw_patterns = []
        self.responses = []
        self.subword_vectors = {}
        all_subwords = Counter()

        with open(corpus_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        current_patterns = None
        current_reply = []

        for line in full_text.split("\n"):
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
            elif current_patterns and line_str:
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

    def load(self, filepath="model.json", lexicon_path="lexicon.json", synsets_path="synsets.json", math_path="math_knowledge.json", geo_path="geography.json", omnibus_path="omnibus.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.dim = data["dim"]
        self.subword_vectors = data["subword_vectors"]
        self.doc_embeddings = data["doc_embeddings"]
        self.responses = data["responses"]
        self.raw_patterns = data["raw_patterns"]

        for path, attr in [(lexicon_path, "lexicon"), (synsets_path, "synsets"), (math_path, "math_kb"), (geo_path, "geography"), (omnibus_path, "omnibus")]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    setattr(self, attr, json.load(f))
            except Exception:
                setattr(self, attr, {})

    def generate(self, prompt):
        norm = self._normalize_prompt(prompt)

        # 1. Omnibus Knowledge Check (Periodic Elements, Astronomy, Linux, Currencies)
        omni_match = self._lookup_omnibus(norm)
        if omni_match:
            return self._spontaneous_emotion_wrapper(omni_match, prompt)

        # 2. Geography Knowledge Base Lookup
        geo_match = self._lookup_geography(norm)
        if geo_match:
            return self._spontaneous_emotion_wrapper(geo_match, prompt)

        # 3. Math Formula & AST Calculation
        math_fact = self._lookup_math_kb(norm)
        if math_fact:
            return self._spontaneous_emotion_wrapper(math_fact, prompt)

        math_eval = self._evaluate_expression(norm)
        if math_eval:
            return self._spontaneous_emotion_wrapper(math_eval, prompt)

        # 4. English Lexicon Lookup
        lex_match = self._lookup_lexicon(norm)
        if lex_match:
            return self._spontaneous_emotion_wrapper(lex_match, prompt)

        # 5. Dense Subword Vector Match
        query_vec = self._sentence_embedding(norm)
        best_score = -1.0
        best_idx = -1
        for i, dvec in enumerate(self.doc_embeddings):
            sim = self._cosine_similarity(query_vec, dvec)
            if sim > best_score:
                best_score = sim
                best_idx = i

        if best_score >= 0.52:
            return self._spontaneous_emotion_wrapper(self.responses[best_idx], prompt)

        return (
            f"I analyzed your inquiry about **'{prompt}'**, but I don't have enough verified data indexed on this topic yet. "
            f"Try asking about chemical elements, astronomy, Linux commands, world geography, math, or definitions."
        )

if __name__ == "__main__":
    engine = ArchWiseEngine(dim=32)
    engine.train("corpus.txt", "lexicon.json", "synsets.json", "math_knowledge.json", "geography.json", "omnibus.json")
    engine.save("model.json")
    print("ArchWise Omnibus Engine compiled successfully.")
