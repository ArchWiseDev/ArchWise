import json
import random
from collections import defaultdict, Counter

class ArchWiseEngine:
    def __init__(self, order=4):
        self.order = order
        self.model = defaultdict(Counter)
        self.vocab = set()

    def train(self, text):
        self.vocab.update(set(text))
        padded = "~" * self.order + text
        for i in range(len(padded) - self.order):
            context = padded[i:i + self.order]
            target = padded[i + self.order]
            self.model[context][target] += 1

    def save(self, filepath="model.json"):
        serializable = {ctx: dict(counts) for ctx, counts in self.model.items()}
        data = {
            "order": self.order,
            "vocab": list(self.vocab),
            "model": serializable
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filepath="model.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.order = data["order"]
        self.vocab = set(data["vocab"])
        self.model = defaultdict(Counter, {k: Counter(v) for k, v in data["model"].items()})

    def generate(self, prompt, max_chars=120, temperature=0.7):
        if not prompt:
            prompt = "hello"
        result = prompt
        context = ("~" * self.order + prompt)[-self.order:]
        
        for _ in range(max_chars):
            counts = self.model.get(context)
            if not counts:
                candidates = list(self.vocab)
                weights = [1.0] * len(candidates)
            else:
                candidates = list(counts.keys())
                raw_weights = [counts[c] for c in candidates]
                weights = [w ** (1.0 / max(temperature, 0.1)) for w in raw_weights]
            
            chosen_char = random.choices(candidates, weights=weights, k=1)[0]
            result += chosen_char
            context = (context + chosen_char)[-self.order:]
            if chosen_char in [".", "!", "?"] and len(result) > len(prompt) + 20:
                break
        return result

if __name__ == "__main__":
    with open("corpus.txt", "r", encoding="utf-8") as f:
        raw_data = f.read().lower()
    engine = ArchWiseEngine(order=4)
    engine.train(raw_data)
    engine.save("model.json")
    print("ArchWise model trained and saved to model.json successfully.")
