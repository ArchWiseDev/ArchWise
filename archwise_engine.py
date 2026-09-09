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

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}

def safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Non-numeric constant")
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
        raise ValueError(f"Unsupported operation: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError("Unsupported unary operator")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            if func_name == "sqrt" and len(node.args) == 1:
                val = safe_eval(node.args[0])
                if val < 0:
                    raise ValueError("Negative square root")
                return math.sqrt(val)
            elif func_name == "abs" and len(node.args) == 1:
                return abs(safe_eval(node.args[0]))
    raise ValueError("Invalid mathematical syntax")

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
        self.last_query = ""

    def _detect_persona(self, text):
        clean = text.lower()
        words = set(re.findall(r"\b\w+\b", clean))
        if words.intersection({"yo", "bruh", "nah", "yeah", "gimme", "wanna", "gonna", "sup", "lol", "dude"}):
            return "casual"
        elif words.intersection({"furthermore", "consequently", "regarding", "therefore", "clarify", "synthesize"}):
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
            return f"{response} Let's solve more!"
        return response

    def _lookup_math_kb(self, prompt):
        clean = prompt.lower().strip()
        for key, val in self.math_kb.items():
            if key in clean:
                return val
        return None

    def _evaluate_expression(self, text):
        norm = text.lower()
        norm = norm.replace("what is", "").replace("calculate", "").replace("solve", "").strip()
        norm = norm.replace("times", "*").replace("multiplied by", "*")
        norm = norm.replace("divided by", "/").replace("plus", "+").replace("minus", "-")
        norm = norm.replace("^", "**")

        # Convert word numbers to digits
        tokens = norm.split()
        converted = [str(WORD_NUMBERS[t]) if t in WORD_NUMBERS else t for t in tokens]
        expr_candidate = "".join(converted)

        # Allow basic functions like sqrt
        if not re.search(r"[\d\+\-\*\/\^\%]", expr_candidate):
            return None

        clean_expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\%\,\s_a-zA-Z]", "", expr_candidate).strip()

        try:
            tree = ast.parse(clean_expr, mode="eval")
            res = safe_eval(tree.body)
            if isinstance(res, float) and res.is_integer():
                res = int(res)
            elif isinstance(res, float):
                res = round(res, 6)
            return f"**Result**: `{clean_expr}` = **{res}**"
        except ZeroDivisionError:
            return "Division by zero is mathematically undefined."
        except Exception:
            return None

    def _lookup_lexicon(self, prompt):
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

    def train(self, corpus_path="corpus.txt", lexicon_path="lexicon.json", synsets_path="synsets.json", math_path="math_knowledge.json"):
        try:
            with open(lexicon_path, "r", encoding="utf-8") as lf:
                self.lexicon = json.load(lf)
        except Exception:
            self.lexicon = {}

        try:
            with open(synsets_path, "r", encoding="utf-8") as sf:
                self.synsets = json.load(sf)
        except Exception:
            self.synsets = {}

        try:
            with open(math_path, "r", encoding="utf-8") as mf:
                self.math_kb = json.load(mf)
        except Exception:
            self.math_kb = {}

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

    def load(self, filepath="model.json", lexicon_path="lexicon.json", synsets_path="synsets.json", math_path="math_knowledge.json"):
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
        try:
            with open(synsets_path, "r", encoding="utf-8") as sf:
                self.synsets = json.load(sf)
        except Exception:
            self.synsets = {}
        try:
            with open(math_path, "r", encoding="utf-8") as mf:
                self.math_kb = json.load(mf)
        except Exception:
            self.math_kb = {}

    def generate(self, prompt):
        user_persona = self._detect_persona(prompt)

        # 1. Grammar check
        grammar_eval = self._correct_grammar(prompt)
        if grammar_eval:
            return self._adapt_tone(grammar_eval, user_persona)

        # 2. Math Formula / Principle Knowledge Base Lookup
        math_fact = self._lookup_math_kb(prompt)
        if math_fact:
            return self._adapt_tone(math_fact, user_persona)

        # 3. Dynamic AST Math Evaluation
        math_eval = self._evaluate_expression(prompt)
        if math_eval:
            return self._adapt_tone(math_eval, user_persona)

        # 4. Fast Lexicon Lookup
        lex_match = self._lookup_lexicon(prompt)
        if lex_match:
            return self._adapt_tone(lex_match, user_persona)

        # 5. Dense Subword Vector Match
        query_vec = self._sentence_embedding(prompt)
        best_score = -1.0
        best_idx = -1
        for i, dvec in enumerate(self.doc_embeddings):
            sim = self._cosine_similarity(query_vec, dvec)
            if sim > best_score:
                best_score = sim
                best_idx = i

        if best_score >= 0.42:
            return self._adapt_tone(self.responses[best_idx], user_persona)

        return self._adapt_tone("I analyzed your input, but I don't have enough verified information on that specific topic yet. Could you rephrase or ask another question?", user_persona)

if __name__ == "__main__":
    engine = ArchWiseEngine(dim=32)
    engine.train("corpus.txt", "lexicon.json", "synsets.json", "math_knowledge.json")
    engine.save("model.json")
    print("ArchWise math-enabled AST engine compiled successfully.")
