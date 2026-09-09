import http.server
import socketserver
import json
import os
from archwise_engine import ArchWiseEngine

PORT = 8080

# Initialize and load the pre-trained 3-tier engine
engine = ArchWiseEngine(dim=32)
if os.path.exists("model.json"):
    engine.load("model.json", "lexicon.json", "synsets.json")
    print("ArchWise 3-Tier Model loaded into active memory.")
else:
    print("model.json not found. Compiling from scratch...")
    engine.train("corpus.txt", "lexicon.json", "synsets.json")
    engine.save("model.json")

class ArchWiseHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode("utf-8"))
                user_message = payload.get("message", "").strip()
                
                if not user_message:
                    response_text = "Please enter a valid query."
                else:
                    response_text = engine.generate(user_message)
                    
                response_data = {"reply": response_text}
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
                
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ArchWiseHandler) as httpd:
        print(f"ArchWise serving on http://localhost:{PORT} (Press Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer shutting down cleanly.")
