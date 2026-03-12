#!/usr/bin/env python3
"""
LIBERO 시뮬레이션 실시간 라이브 뷰어
- eval_libero.py 실행 중 현재 프레임을 브라우저에서 실시간으로 확인
- 사용법: python examples/LIBERO/eval_files/live_viewer.py
- 브라우저에서 http://localhost:8888 접속
"""

import http.server
import os
import time

FRAME_PATH = "/tmp/libero_live_frame.jpg"
PORT = 8888

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>LIBERO Live Viewer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f0f1a;
      color: #e0e0ff;
      font-family: 'Courier New', monospace;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 24px;
    }
    h1 {
      font-size: 1.4rem;
      letter-spacing: 0.15em;
      color: #7eb8f7;
      margin-bottom: 16px;
    }
    .frame-box {
      border: 2px solid #334;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 0 30px rgba(100, 160, 255, 0.15);
    }
    img {
      display: block;
      width: 512px;
      height: 512px;
      object-fit: contain;
      background: #111;
    }
    .status {
      margin-top: 12px;
      font-size: 0.8rem;
      color: #556;
    }
    .dot {
      display: inline-block;
      width: 8px; height: 8px;
      border-radius: 50%;
      background: #4f8;
      margin-right: 6px;
      animation: blink 1s infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
  </style>
</head>
<body>
  <h1>&#9654; LIBERO Live Simulation</h1>
  <div class="frame-box">
    <img id="frame" src="/frame.jpg?t=0" alt="Waiting for simulation...">
  </div>
  <p class="status"><span class="dot"></span>Live &mdash; 자동 갱신 중 (0.5초 간격)</p>
  <script>
    const img = document.getElementById('frame');
    function refresh() {
      img.src = '/frame.jpg?t=' + Date.now();
    }
    img.onerror = function() {
      this.alt = '시뮬레이션 시작을 기다리는 중...';
    };
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""


class LiveViewerHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/frame.jpg"):
            if os.path.exists(FRAME_PATH):
                with open(FRAME_PATH, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache, no-store")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # suppress request logs


if __name__ == "__main__":
    print(f"[LiveViewer] http://localhost:{PORT} 에서 라이브 뷰어 실행 중")
    print(f"[LiveViewer] 프레임 파일 감시: {FRAME_PATH}")
    print("[LiveViewer] 브라우저에서 위 주소를 열어주세요. Ctrl+C 로 종료.")
    server = http.server.HTTPServer(("0.0.0.0", PORT), LiveViewerHandler)
    server.serve_forever()
