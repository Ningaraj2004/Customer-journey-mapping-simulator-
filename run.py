"""
EcomML Studio v2 - Launcher
Run with: python run.py
"""

import http.server
import socketserver
import webbrowser
import threading
import os
import sys
import subprocess
import socket
import time
import signal


PORT = 8080
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
DASHBOARD_FILE = "ecomml_dashboard.html"
URL = f"http://localhost:{PORT}/{DASHBOARD_FILE}"


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("localhost", port)) == 0


def kill_port(port):
    try:
        result = subprocess.check_output(["netstat", "-ano"], text=True)
        for line in result.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                pid = parts[4]
                subprocess.call(
                    ["taskkill", "/PID", pid, "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"  [INFO] Released port {port} (PID {pid} terminated)")
                time.sleep(1)
                return
    except Exception as e:
        print(f"  [WARN] {e}")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    def log_error(self, format, *args):
        pass


def open_browser():
    time.sleep(1.8)
    print(f"  >> Opening browser: {URL}\n")
    webbrowser.open(URL)


def _ignore_sigint(signum, frame):
    # Ignore Ctrl+C from parent process; user must close window manually
    pass


def main():
    # Prevent parent process from killing us via SIGINT propagation
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    print("")
    print("=" * 54)
    print("   EcomML Studio v2  --  Dashboard Launcher")
    print("=" * 54)
    print(f"   URL  : {URL}")
    print(f"   Dir  : {DASHBOARD_DIR}")
    print("-" * 54)
    print("   Close this window (or Ctrl+C) to stop")
    print("=" * 54)

    if is_port_in_use(PORT):
        print(f"\n  [INFO] Port {PORT} busy — freeing ...")
        kill_port(PORT)

    os.chdir(DASHBOARD_DIR)
    socketserver.TCPServer.allow_reuse_address = True

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
            print(f"\n  Server live at: {URL}\n")
            # Re-enable Ctrl+C now that we're serving
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped. Goodbye!")
        sys.exit(0)
    except Exception as ex:
        print(f"\n  [ERROR] {type(ex).__name__}: {ex}")
        input("\n  Press Enter to close...")
        sys.exit(1)


if __name__ == "__main__":
    main()
