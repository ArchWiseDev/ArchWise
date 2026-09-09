import http.server
import socketserver
import json
import os
import secrets
from archwise_engine import ArchWiseEngine

PORT = 8080
KEYS_FILE = "api_keys.json"

engine = ArchWiseEngine()
print("Loading ArchWise engine...")
engine.load("model.json", "lexicon.json", "synsets.json", "math_knowledge.json", "geography.json", "omnibus.json")
print(f"ArchWise is live at http://localhost:{PORT}")

def load_keys():
    if not os.path.exists(KEYS_FILE):
        return {}
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_keys(keys):
    try:
        with open(KEYS_FILE, "w", encoding="utf-8") as f:
            json.dump(keys, f, indent=2)
    except Exception:
        pass

def verify_token(req_handler):
    auth_header = req_handler.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    else:
        token = req_handler.headers.get("X-API-Key", "").strip()

    if not token:
        return False, "Missing API token."

    keys = load_keys()
    if token not in keys:
        return False, "Invalid API key."

    if not keys[token].get("active", False):
        return False, "API key revoked."

    keys[token]["total_requests"] = keys[token].get("total_requests", 0) + 1
    save_keys(keys)
    return True, keys[token].get("client", "User")

class ArchWiseHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
        self.end_headers()

    def do_GET(self):
        # Serve index.html directly from directory
        if self.path in ("/", "/index.html"):
            try:
                with open("index.html", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error loading index.html: {e}".encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if self.path in ("/api/chat", "/chat"):
            prompt = payload.get("prompt", "")
            persona = payload.get("persona", "standard")
            reply = engine.generate(prompt, persona)
            return self._send_json(200, {"response": reply})

        elif self.path == "/api/keys/generate":
            name = payload.get("name", "User").strip()
            new_key = f"ak-{secrets.token_hex(16)}"
            keys = load_keys()
            keys[new_key] = {
                "client": name,
                "active": True,
                "total_requests": 0
            }
            save_keys(keys)
            return self._send_json(200, {"key": new_key, "client": name})

        elif self.path == "/api/v1/chat":
            is_valid, client_or_err = verify_token(self)
            if not is_valid:
                return self._send_json(401, {"error": client_or_err})

            prompt = payload.get("prompt", "")
            persona = payload.get("persona", "standard")
            reply = engine.generate(prompt, persona)
            return self._send_json(200, {"response": reply, "client": client_or_err})

        else:
            return self._send_json(404, {"error": "Not Found"})

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ArchWiseHandler) as httpd:
        httpd.serve_forever()
