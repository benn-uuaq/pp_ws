"""Check that the dashboard (29999) client reconnects after a dropped connection.

Previously, when one command failed with an exception, the broken socket was kept, so
every later command returned None forever (`[play] Result: None` on the RCS screen).
A fake dashboard server is started, the connection is dropped on purpose, and the next command
is checked to reconnect and get a proper reply.
"""

import socket
import threading

import pytest

from elite_robot_controller.robot.robot_driver import Robot_29999


class FakeDashboard:
    """Imitates the Elite dashboard. Greets on connect and answers each command with one line.

    After answering `drop_after` commands it closes the connection — same as reloading the task on the
    pendant or restarting the controller.
    """

    def __init__(self, drop_after=None, replies=None):
        self.drop_after = drop_after
        self.replies = replies or {}
        self.received = []
        self.connections = 0
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(4)
        self.port = self._srv.getsockname()[1]
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        self._srv.settimeout(0.2)
        while not self._stop:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                continue
            self.connections += 1
            with conn:
                conn.sendall(b"Connected: Elite Dashboard Server\n")
                answered = 0
                buf = b""
                conn.settimeout(0.2)
                while not self._stop:
                    try:
                        chunk = conn.recv(1024)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        cmd = line.decode().strip()
                        self.received.append(cmd)
                        reply = self.replies.get(cmd, f"OK {cmd}")
                        conn.sendall(f"{reply}\n".encode())
                        answered += 1
                    if self.drop_after is not None and answered >= self.drop_after:
                        break   # drop the connection

    def close(self):
        self._stop = True
        self._srv.close()
        self._thread.join(timeout=1.0)


@pytest.fixture
def server():
    srv = FakeDashboard(drop_after=1)
    yield srv
    srv.close()


def test_command_after_a_dropped_connection_reconnects(server):
    """Even after the connection drops, the next command reconnects and gets a reply."""
    dash = Robot_29999("127.0.0.1", server.port)
    assert dash.connect_29999() is not None

    assert dash.send_command_29999("stop") == "OK stop"
    # the server just closed the connection. Previously everything returned None from here on.
    assert dash.send_command_29999("play") == "OK play"
    assert server.connections >= 2, "did not reconnect"
    dash.disconnect_29999()


def test_unreachable_dashboard_reports_why():
    """If it cannot connect, it returns None and keeps the reason — 'Result: None' alone tells nothing."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()           # nobody is on this port

    dash = Robot_29999("127.0.0.1", port)
    assert dash.send_command_29999("play") is None
    assert "connection failed" in dash.last_error


def test_stale_reply_is_not_taken_as_the_next_answer():
    """A late reply to an earlier command is not read as the reply to the next command."""
    srv = FakeDashboard()
    try:
        dash = Robot_29999("127.0.0.1", srv.port)
        dash.connect_29999()
        # imitate the server sending one line without a request.
        assert dash.send_command_29999("robotMode") == "OK robotMode"
        dash.sock.sendall(b"")  # only checks that the connection is alive
        assert dash.send_command_29999("play") == "OK play"
        dash.disconnect_29999()
    finally:
        srv.close()
