from http.server import SimpleHTTPRequestHandler, HTTPServer
import os

PORT = 9999

class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(__file__), **kwargs)

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        
        ctype = self.guess_type(path)
        if ctype.startswith('text/'):
            ctype += '; charset=utf-8'
        return super().send_head()

def run_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"サーバー起動: http://localhost:{PORT}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()