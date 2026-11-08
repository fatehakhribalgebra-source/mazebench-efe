"""Persistent connection to the real MazeBench engine.

``scripts/maze-bridge.js`` in MazeBenchEngine is a line-oriented JSON server on
stdin/stdout: write one request object per line, read one response object per
line. It holds the authoritative game session in-process, so we keep one
subprocess alive for a whole rollout instead of re-launching per step.

Launch flags (from ``scripts/maze-bridge.js``'s own ``parseArgs``):
    --level level_HxI --view top-diagonal --yaw 0
    --observation-mode text|json  --omniscient  --hide-names  --hide-names-seed S

Commands (from its ``handleCommand``):
    observe | move{direction} | rotate_camera{direction} | undo | reset_level
    goto_level{x,y} | reset_run | scorecard | quit | no_move | close
    replay_fast{enabled} | restore_checkpoint{checkpoint}
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

# Node ships with several desktop tools; this one is a modern runtime that is
# already present on many Windows installs. Checked before use.
_BUNDLED_NODE_HINTS = [
    Path(os.path.expandvars(r"%LOCALAPPDATA%\OpenAI\Codex\runtimes\cua_node")),
]


def discover_node(explicit: str | None = None) -> str:
    """Find a Node >= 18 executable."""
    if explicit:
        return explicit
    env_node = os.environ.get("MAZEBENCH_NODE")
    if env_node:
        return env_node
    found = shutil.which("node")
    if found:
        return found
    for root in _BUNDLED_NODE_HINTS:
        if root.is_dir():
            for candidate in sorted(root.glob("*/bin/node.exe")):
                return str(candidate)
    raise RuntimeError(
        "No Node runtime found. Install Node 18+ (or set MAZEBENCH_NODE to a "
        "node executable). The MazeBench engine is JavaScript; it cannot run "
        "without one."
    )


def discover_repo(explicit: str | None = None) -> Path:
    """Locate a MazeBenchEngine checkout containing scripts/maze-bridge.js."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_repo = os.environ.get("MAZEBENCH_REPO")
    if env_repo:
        candidates.append(Path(env_repo))
    candidates += [
        Path.home() / "Documents" / "mazebench_src" / "repo",
        Path.cwd() / "MazeBenchEngine",
    ]
    for candidate in candidates:
        if (candidate / "scripts" / "maze-bridge.js").is_file():
            return candidate
    raise RuntimeError(
        "Could not find scripts/maze-bridge.js. Clone "
        "https://github.com/mazebench/MazeBenchEngine and set MAZEBENCH_REPO "
        "to the checkout path."
    )


class BridgeError(RuntimeError):
    """The engine rejected a command, or the subprocess died."""


@dataclass
class BridgeConfig:
    level_id: str = "level_HxI"
    view: str = "top-diagonal"
    yaw: int = 0
    observation_mode: str = "text"      # "text" (ASCII) or "json"
    omniscient: bool = False
    hide_names: bool = False
    hide_names_seed: str = "1"
    timeout_seconds: float = 30.0
    node_bin: str | None = None
    repo_root: str | None = None


class MazeBridge:
    """One live MazeBench game session."""

    def __init__(self, config: BridgeConfig | None = None):
        self.config = config or BridgeConfig()
        self.node = discover_node(self.config.node_bin)
        self.repo = discover_repo(self.config.repo_root)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._start()

    # -- lifecycle ----------------------------------------------------------
    def _start(self) -> None:
        script = self.repo / "scripts" / "maze-bridge.js"
        argv = [
            self.node, str(script),
            "--level", self.config.level_id,
            "--view", self.config.view,
            "--yaw", str(int(self.config.yaw)),
            "--observation-mode", self.config.observation_mode,
        ]
        if self.config.omniscient:
            argv.append("--omniscient")
        if self.config.hide_names:
            argv += ["--hide-names", "--hide-names-seed", self.config.hide_names_seed]

        self._proc = subprocess.Popen(
            argv,
            cwd=str(self.repo),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf8",
            errors="replace",
            bufsize=1,
        )

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        if not self.alive:
            self._proc = None
            return
        try:
            self._send_raw({"command": "close"})
        except Exception:
            pass
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def __enter__(self) -> "MazeBridge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- transport ----------------------------------------------------------
    def _send_raw(self, message: dict) -> dict:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise BridgeError("maze-bridge.js is not running")
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            stderr = ""
            try:
                stderr = (proc.stderr.read() or "")[:2000]
            except Exception:
                pass
            raise BridgeError(f"maze-bridge.js closed its output. stderr: {stderr}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise BridgeError(f"invalid JSON from bridge: {line[:400]}") from error

    def call(self, command: str, **kwargs) -> dict:
        """Send one command and return its response, raising on engine errors."""
        with self._lock:
            response = self._send_raw({"command": command, **kwargs})
        if not response.get("ok", False):
            raise BridgeError(f"{command} failed: {response.get('error')}")
        return response

    # -- game commands ------------------------------------------------------
    def observe(self) -> dict:
        return self.call("observe")

    def move(self, direction: str) -> dict:
        return self.call("move", direction=direction)

    def rotate_camera(self, direction: str) -> dict:
        return self.call("rotate_camera", direction=direction)

    def undo(self) -> dict:
        return self.call("undo")

    def reset_level(self) -> dict:
        return self.call("reset_level")

    def goto_level(self, x: str, y: str) -> dict:
        return self.call("goto_level", x=x, y=y)

    def no_move(self) -> dict:
        return self.call("no_move")

    def reset_run(self) -> dict:
        return self.call("reset_run")

    def scorecard(self) -> dict:
        return self.call("scorecard")

    def quit_game(self) -> dict:
        return self.call("quit")

    def set_replay_fast(self, enabled: bool) -> dict:
        return self.call("replay_fast", enabled=bool(enabled))

    def restore_checkpoint(self, checkpoint: dict) -> dict:
        """Restore a previously captured ``_render_state``.

        Lets you snapshot and rewind the authoritative engine. Not used by the
        planner (that is the world model's job) but useful for debugging a
        divergence between imagined and real outcomes.
        """
        return self.call("restore_checkpoint", checkpoint=checkpoint)
