import re
import math
import json
import random
import ast
import operator
import urllib.request
import urllib.parse
from html.parser import HTMLParser
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
    "of", "and", "or", "that", "this", "do", "does", "did", "i", "whos", "who"
}

OPINION_KEYWORDS = {
    "best", "greatest", "worst", "favorite", "better", "top", "great", "coolest"
}

SAFE_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos
}

def safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)): return node.value
        raise ValueError
    elif isinstance(node, ast.BinOp):
        left, right = safe_eval(node.left), safe_eval(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            if op_type == ast.Div and right == 0: raise ZeroDivisionError
            return SAFE_OPERATORS[op_type](left, right)
    return None

def extract_subwords(word, min_n=3, max_n=5):
    w = f"<{word}>"
    subwords = []
    length = len(w)
    for n in range(min_n, min(max_n + 1, length + 1)):
        for i in range(length - n + 1):
            subwords.append(w[i:i + n])
    return subwords

class DeepHTMLPageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ignore = False
        self.in_title = False
        self.in_heading = False
        self.title = ""
        self.headings = []
        self.paragraphs = []
        self.current_buf = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg", "path"):
            self.ignore = True
        elif tag == "title":
            self.in_title = True
            self.current_buf = []
        elif tag in ("h1", "h2", "h3"):
            self.in_heading = True
            self.current_buf = []
        elif tag in ("p", "li"):
            self.current_buf = []

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg", "path"):
            self.ignore = False
        elif tag == "title" and self.in_title:
            self.in_title = False
            self.title = "".join(self.current_buf).strip()
        elif tag in ("h1", "h2", "h3") and self.in_heading:
            self.in_heading = False
            h_text = "".join(self.current_buf).strip()
            if h_text and len(h_text) > 3:
                self.headings.append(h_text)
        elif tag in ("p", "li"):
            p_text = "".join(self.current_buf).strip()
            if p_text and len(p_text) > 25:
                self.paragraphs.append(p_text)

    def handle_data(self, data):
        if not self.ignore:
            clean = re.sub(r"\s+", " ", data)
            if clean.strip():
                self.current_buf.append(clean)

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

    def _normalize(self, text):
        norm = text.strip().lower()
        norm = re.sub(r"[^\w\s\+\-\*\/\^\.]", " ", norm)
        return re.sub(r"\s+", " ", norm).strip()

    def _sentence_embedding(self, text):
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        words = [w for w in clean.split() if w and w not in STOPWORDS]
        if not words: return [0.0] * self.dim
        doc_vec = [0.0] * self.dim
        for w in words:
            subwords = extract_subwords(w)
            vec = [0.0] * self.dim
            count = 0
            for sw in subwords:
                if sw in self.subword_vectors:
                    for d in range(self.dim):
                        vec[d] += self.subword_vectors[sw][d]
                    count += 1
            if count > 0:
                for d in range(self.dim):
                    doc_vec[d] += vec[d] / count
        norm = math.sqrt(sum(v * v for v in doc_vec))
        return [v / norm for v in doc_vec] if norm > 0 else doc_vec

    def _cosine_similarity(self, vec_a, vec_b):
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def _inspect_webpage_url(self, url):
        headers = {"User-Agent": "Mozilla/5.0 (Android; Termux) ArchWise/9.0"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=6) as resp:
                html_raw = resp.read().decode("utf-8", errors="ignore")
                parser = DeepHTMLPageParser()
                parser.feed(html_raw)

                title = parser.title or "Untitled Document"
                h_sample = parser.headings[:3]
                p_sample = parser.paragraphs[:4]

                if not p_sample:
                    return f"**Web Page Ingested**: `{url}`\n\nNo readable text blocks could be extracted."

                out = [f"**Page**: {title}", f"*URL*: `{url}`"]
                if h_sample:
                    out.append("\n**Core Headings:**")
                    for h in h_sample:
                        out.append(f"- {h}")
                
                out.append("\n**Extracted Content Overview:**")
                for p in p_sample:
                    out.append(f"- {p}")

                return "\n".join(out)
        except Exception as e:
            return f"Failed to ingest page `{url}`: {str(e)}"

    def _query_wikipedia_by_content(self, query):
        clean = re.sub(r"^(who is|what is|whos|whats|tell me about|info about|information on|define|search|history of|explain)\s+", "", query.strip(), flags=re.IGNORECASE)
        clean = clean.strip()
        if not clean or len(clean) < 2:
            return None

        headers = {"User-Agent": "ArchWiseAssistant/9.0 (Semantic Search; Termux)"}

        try:
            search_params = {"action": "query", "list": "search", "srsearch": clean, "srlimit": 4, "format": "json"}
            search_url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(search_params)}"
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                search_data = json.loads(resp.read().decode("utf-8"))
                search_results = search_data.get("query", {}).get("search", [])
                if not search_results:
                    return None
                candidate_titles = [r["title"] for r in search_results]

            extract_params = {
                "action": "query", "prop": "extracts", "exintro": 1,
                "explaintext": 1, "redirects": 1, "titles": "|".join(candidate_titles), "format": "json"
            }
            extract_url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(extract_params)}"
            req2 = urllib.request.Request(extract_url, headers=headers)
            with urllib.request.urlopen(req2, timeout=6) as resp2:
                extract_data = json.loads(resp2.read().decode("utf-8"))
                pages = extract_data.get("query", {}).get("pages", {})

            query_vec = self._sentence_embedding(clean)
            query_keywords = set(re.findall(r"\b\w+\b", clean.lower())) - STOPWORDS

            best_article = None
            best_score = -1.0

            for _, page in pages.items():
                title = page.get("title", "")
                raw_extract = page.get("extract", "").strip()
                if not raw_extract:
                    continue

                cleaned_extract = re.sub(r"\[\d+\]", "", raw_extract)
                paragraphs = [p.strip() for p in cleaned_extract.split("\n\n") if p.strip()]
                lead_text = " ".join(paragraphs[:2])

                content_vec = self._sentence_embedding(f"{title} {lead_text[:400]}")
                vector_sim = self._cosine_similarity(query_vec, content_vec)

                lead_words = set(re.findall(r"\b\w+\b", f"{title} {lead_text}".lower()))
                keyword_match_ratio = len(query_keywords.intersection(lead_words)) / max(len(query_keywords), 1)

                composite_score = (0.6 * vector_sim) + (0.4 * keyword_match_ratio)
                if composite_score > best_score:
                    best_score = composite_score
                    page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    summary = "\n\n".join(paragraphs[:2])
                    best_article = f"**{title}**\n\n{summary}\n\n*Source*: {page_url}"

            if best_score >= 0.45:
                return best_article
        except Exception:
            pass

        return None

    def _handle_subjective(self, norm_prompt):
        words = set(norm_prompt.split())
        if "technician" in words or "dribbler" in words or "playmaker" in words:
            return (
                "**The Concept of a 'Technician'**\n\n"
                "In football and tactical sports, 'the greatest technician' refers to players whose game relies on exquisite first touch, ball manipulation under pressure, and micro-geometry:\n\n"
                "* **Lionel Messi**: Unparalleled balance, close-control dribbling, and decision-making density.\n"
                "* **Zinedine Zidane & Ronaldinho**: Masters of aesthetic control, weight of pass, and ball manipulation.\n"
                "* **Dennis Bergkamp & Andrés Iniesta**: Surgical geometric vision and touch execution in congested midfield zones."
            )
        return None

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
                current_patterns = [p.strip().lower() for p in parts[0].split("|") if p.strip()]
                current_reply = [parts[1].strip()]
            elif current_patterns and line_str:
                current_reply.append(line_str)

        if current_patterns and current_reply:
            reply_text = "\n".join(current_reply).strip()
            for pat in current_patterns:
                self.raw_patterns.append(pat)
                self.responses.append(reply_text)

        for pat in self.raw_patterns:
            for w in pat.split():
                for sw in extract_subwords(w):
                    all_subwords[sw] += 1

        random.seed(42)
        for sw, _ in all_subwords.most_common(10000):
            self.subword_vectors[sw] = [round(random.uniform(-0.5, 0.5), 4) for _ in range(self.dim)]

        self.doc_embeddings = [self._sentence_embedding(pat) for pat in self.raw_patterns]

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
        # 1. Inspect URL links directly
        url_match = re.search(r"https?://[^\s]+", prompt)
        if url_match:
            return self._inspect_webpage_url(url_match.group(0))

        norm = self._normalize(prompt)

        # 2. Exact trained response match
        for idx, pat in enumerate(self.raw_patterns):
            if norm == pat or prompt.lower().strip() == pat:
                return self.responses[idx]

        # 3. Subjective / Opinion Analysis Gate
        if any(w in norm.split() for w in OPINION_KEYWORDS):
            subj = self._handle_subjective(norm)
            if subj:
                return subj

        # 4. Local Omnibus exact match (code snippets, HTML/CSS)
        clean_target = re.sub(r"^(what is|who is|tell me about|define|how to)\s+", "", norm).strip()
        if clean_target in self.omnibus:
            return self.omnibus[clean_target]

        for key, val in self.omnibus.items():
            if key in norm and len(key) > 5:
                return val

        # 5. Geography match
        if "capital of" in norm:
            m = re.search(r"capital of\s+([a-z\s]+)", norm)
            if m:
                target = m.group(1).strip()
                if target in self.geography:
                    return f"The capital of **{self.geography[target].get('name', target.title())}** is **{self.geography[target].get('capital')}**."

        # 6. Math calculation evaluation
        try:
            m_expr = re.sub(r"[^0-9\+\-\*\/\(\)\.]", "", norm)
            if len(m_expr) >= 3 and any(op in m_expr for op in "+-*/"):
                tree = ast.parse(m_expr, mode="eval")
                res = safe_eval(tree.body)
                if res is not None:
                    return f"**Result**: `{m_expr}` = **{res}**"
        except Exception:
            pass

        # 7. Semantic Content-Inspected Encyclopedic Match
        pedia_match = self._query_wikipedia_by_content(prompt)
        if pedia_match:
            return pedia_match

        # 8. High-threshold Vector Match (0.75)
        query_vec = self._sentence_embedding(norm)
        best_score = -1.0
        best_idx = -1
        for i, dvec in enumerate(self.doc_embeddings):
            sim = self._cosine_similarity(query_vec, dvec)
            if sim > best_score:
                best_score = sim
                best_idx = i

        if best_score >= 0.75:
            return self.responses[best_idx]

        return f"I analyzed your prompt for **'{prompt}'**, but could not find a relevant code structure, HTML pattern, or verifiable record."

if __name__ == "__main__":
    eng = ArchWiseEngine()
    eng.train()
    eng.save()
    print("ArchWise Engine recompiled with HTML & CSS comprehension.")
