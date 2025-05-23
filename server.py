from http.server import SimpleHTTPRequestHandler, HTTPServer
import os

PORT = 8000

class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(__file__), **kwargs)

def run_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"サーバー起動: http://localhost:{PORT}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()