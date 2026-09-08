import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from archwise_engine import ArchWiseEngine

engine = ArchWiseEngine()
engine.load("model.json")

class ArchWiseHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/generate":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode("utf-8"))
            prompt = body.get("prompt", "")
            
            output = engine.generate(prompt=prompt.lower(), max_chars=80, temperature=0.6)
            
            response = json.dumps({"response": output}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        else:
            self.send_error(404)

if __name__ == "__main__":
    server_address = ("", 8080)
    httpd = HTTPServer(server_address, ArchWiseHandler)
    print("ArchWise running on http://localhost:8080 (Press Ctrl+C to stop)")
    httpd.serve_forever()
