import json
import subprocess
import sys
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import IO

from harness_win.rules import STDOUT_CAP, WATCHDOG_GRACE_MS

RUNNER = Path(__file__).resolve().parent / "runner.py"


class AgentFailure(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def local(directory: Path) -> "Agent":
    """Run an agent as a process on this machine, through the platform's runner."""
    return Agent([sys.executable, str(RUNNER), str(directory.resolve())])


class Agent:
    """One agent process, spoken to exactly as the platform speaks to a container."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.stderr_tail = ""
        self._process: subprocess.Popen[bytes] | None = None
        self._buffer = b""
        self._tail = b""
        self._queue: Queue[tuple[str, bytes]] = Queue()
        self._stop_event = threading.Event()

    def start(self, init_budget_s: float) -> None:
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._process = process
        self._stop_event.clear()

        # Start reader threads
        threading.Thread(target=self._read_stream, args=(process.stdout, "stdout"), daemon=True).start()
        threading.Thread(target=self._read_stream, args=(process.stderr, "stderr"), daemon=True).start()

        ready = self._await_line(time.monotonic() + init_budget_s)
        if ready is None:
            raise AgentFailure("init" if process.poll() is None else "crash")
        if not _is_ready(ready):
            raise AgentFailure("init")

    def move(self, fen: str, time_left_ms: int) -> str:
        if self._process is None:
            raise RuntimeError("agent moved before start")
        request = json.dumps({"fen": fen, "time_left_ms": time_left_ms}).encode()
        try:
            stdin = self._process.stdin
            if stdin is None:
                raise AgentFailure("crash")
            stdin.write(request + b"\n")
            stdin.flush()
        except (BrokenPipeError, AttributeError):
            raise AgentFailure("crash") from None

        line = self._await_line(time.monotonic() + (time_left_ms + WATCHDOG_GRACE_MS) / 1000.0)
        if line is None:
            raise AgentFailure("flag")
        return _parse_move(line)

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.kill()
        self._stop_event.set()
        self._drain()
        self.stderr_tail = self._tail.decode("utf-8", "replace")
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()
        self._process.wait()
        self._process = None

    def _read_stream(self, stream: IO[bytes], name: str) -> None:
        while not self._stop_event.is_set():
            try:
                # Read in small chunks to avoid blocking forever on a long line
                chunk = stream.read(STDOUT_CAP)
                if not chunk:
                    if name == "stdout":
                        # Signify stdout closed (crash)
                        self._queue.put(("crash", b""))
                    break
                self._queue.put((name, chunk))
            except Exception:
                break

    def _await_line(self, deadline: float) -> bytes | None:
        while b"\n" not in self._buffer:
            if len(self._buffer) >= STDOUT_CAP:
                raise AgentFailure("illegal")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                name, chunk = self._queue.get(timeout=max(0, remaining))
                if name == "crash":
                    raise AgentFailure("crash")
                if name == "stderr":
                    self._tail += chunk
                else:
                    self._buffer += chunk
            except Empty:
                return None

        line, _, self._buffer = self._buffer.partition(b"\n")
        return line

    def _drain(self) -> None:
        # Read remaining items from queue
        while not self._queue.empty():
            try:
                name, chunk = self._queue.get_nowait()
                if name == "stderr":
                    self._tail += chunk
            except Empty:
                break



def _is_ready(line: bytes) -> bool:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("ready") is True


def _parse_move(line: bytes) -> str:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AgentFailure("illegal") from None
    move = payload.get("move") if isinstance(payload, dict) else None
    if not isinstance(move, str):
        raise AgentFailure("illegal")
    return move
