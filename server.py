import http.server
import socketserver
import json
import os
import traceback
from archwise_engine import ArchWiseEngine

PORT = 8080

engine = ArchWiseEngine(dim=32)
if os.path.exists("model.json"):
    print("Loading existing ArchWise model with full geography...")
    engine.load("model.json", "lexicon.json", "synsets.json", "math_knowledge.json", "geography.json")
else:
    print("No model.json detected. Training from corpus...")
    engine.train("corpus.txt", "lexicon.json", "synsets.json", "math_knowledge.json", "geography.json")
    engine.save("model.json")

class ArchWiseHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/chat":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len).decode("utf-8")
                
                data = json.loads(post_body)
                user_msg = data.get("message", "").strip()
                
                if not user_msg:
                    reply_text = "Please enter a message."
                else:
                    reply_text = engine.generate(user_msg)

                payload = json.dumps({"reply": reply_text}).encode("utf-8")
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)

            except Exception as e:
                traceback.print_exc()
                err_payload = json.dumps({"reply": f"Engine runtime exception: {str(e)}"}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err_payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(err_payload)
        else:
            self.send_error(404, "Endpoint Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ArchWiseHandler) as httpd:
        print(f"ArchWise engine online at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
