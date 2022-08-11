from http.server import HTTPServer, BaseHTTPRequestHandler


class CustomServer(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        self.wfile.write(bytes("<html><body><h1>test</h1></body></html>", "utf-8"))

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        

HOST, PORT = 'localhost', 8080
httpd = HTTPServer((HOST, PORT), CustomServer)
print(f"Running Server. Go to url from browser: {HOST}:{PORT}")
print(f"using curl: curl {HOST}:{PORT}")
httpd.serve_forever()

