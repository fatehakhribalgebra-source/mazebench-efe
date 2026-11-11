"""Persistent connection to the real engine's headless renderer.

``scripts/maze-render-frame.js --serve`` is a line-oriented JSON server exactly
like ``maze-bridge.js``: one request object per line in, one response object per
line out. It drives a headless Chromium through playwright-core and returns a
PNG per request.

**The snapshot route, not the action route.** Every bridge observation carries
``_render_state``; the renderer accepts it as ``snapshot``. We step the bridge,
take its ``_render_state``, and push that to the renderer. Driving both with
actions instead would silently desynchronise them -- the play page auto-undoes
falls while agent sessions keep deaths terminal, and that disagreement is
commented in the renderer itself.

Two Windows-specific traps, both of which cost time if you meet them blind:

1. ``DEFAULT_BROWSER_PATHS`` in the script lists only macOS and Linux paths, so
   browser discovery fails on Windows and the error surfaces somewhere
   unrelated. ``browser`` is therefore **required** here, not optional.
2. ``applyRenderStateSnapshot`` reads ``yaw`` and ``pitch`` *from the snapshot*
   and overwrites the session camera with them. The camera is thus controlled by
   the bridge's ``--yaw``/``--pitch``, not by what we pass to ``init``. We pass
   both anyway so the first frame (before any snapshot) matches.

Firewall note: ``_render_state`` is evaluator-only. Handing it to the renderer
is legitimate -- the renderer is the thing that draws the picture the agent is
allowed to see. It must never reach the model, the objective or the planner.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from .bridge import discover_node, discover_repo

#: Chromium-family browsers, in the order we would rather have them. Chrome and
#: Edge ship on essentially every Windows box, so this almost always resolves.
WINDOWS_BROWSER_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files\Chromium\Application\chrome.exe",
)


def discover_browser(explicit: str | None = None) -> str:
    """Find a Chromium-family browser executable.

    The renderer cannot do this for us on Windows; see the module docstring.
    """
    if explicit:
        if not Path(explicit).is_file():
            raise RuntimeError(f"browser does not exist: {explicit}")
        return explicit
    env_browser = os.environ.get("MAZEBENCH_BROWSER")
    if env_browser:
        if not Path(env_browser).is_file():
            raise RuntimeError(f"MAZEBENCH_BROWSER does not exist: {env_browser}")
        return env_browser
    for candidate in WINDOWS_BROWSER_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "No Chromium-family browser found. The renderer needs one and its own "
        "default paths are macOS/Linux only. Install Chrome/Edge/Brave or set "
        "MAZEBENCH_BROWSER to a browser executable."
    )


class RendererError(RuntimeError):
    """The renderer rejected a request, or the subprocess died."""


@dataclass
class RendererConfig:
    level_id: str = "level_HxI"
    game_id: str = "maze"
    width: int = 512
    height: int = 512
    #: Rings of neighbouring rooms drawn around the player. 1 is the default the
    #: play page uses -- the classic 3x3 neighbourhood -- and is the benchmark's
    #: own vision observation. Not a single room.
    view: int | str = 1
    yaw: int = 0
    pitch: int = 1
    edges: bool = True
    camera_tilt_degrees: float = 58.0
    camera_step_degrees: float = 18.0
    camera_zoom: float = 1.0
    timeout_seconds: float = 180.0
    node_bin: str | None = None
    repo_root: str | None = None
    browser: str | None = None


class FrameRenderer:
    """One live headless-Chromium render session."""

    def __init__(self, config: RendererConfig | None = None):
        self.config = config or RendererConfig()
        self.node = discover_node(self.config.node_bin)
        self.repo = discover_repo(self.config.repo_root)
        self.browser = discover_browser(self.config.browser)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_tail: list[str] = []
        self._start()

    # -- lifecycle ----------------------------------------------------------
    def _start(self) -> None:
        script = self.repo / "scripts" / "maze-render-frame.js"
        if not script.is_file():
            raise RendererError(f"missing {script}")
        self._proc = subprocess.Popen(
            [self.node, str(script), "--serve"],
            cwd=str(self.repo),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf8",
            errors="replace",
            bufsize=1,
        )
        # The renderer is chatty on stderr (Chromium warnings). Drain it on a
        # thread or a full pipe buffer deadlocks the whole collection run.
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip())
            del self._stderr_tail[:-40]

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        if not self.alive:
            self._proc = None
            return
        try:
            self._request({"command": "close"})
        except Exception:
            pass
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        for closer in (lambda: proc.stdin.close(), lambda: proc.wait(timeout=10)):
            try:
                closer()
            except Exception:
                pass
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:
                pass

    def __enter__(self) -> "FrameRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- transport ----------------------------------------------------------
    def _request(self, message: dict) -> dict:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise RendererError(
                "maze-render-frame.js is not running. stderr tail:\n"
                + "\n".join(self._stderr_tail[-20:])
            )
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RendererError(
                "maze-render-frame.js closed its output. stderr tail:\n"
                + "\n".join(self._stderr_tail[-20:])
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise RendererError(f"invalid JSON from renderer: {line[:400]}") from error

    def _payload(self, **extra) -> dict:
        config = self.config
        payload = {
            "gameId": config.game_id,
            "levelId": config.level_id,
            "width": config.width,
            "height": config.height,
            "view": config.view,
            "yaw": config.yaw,
            "edges": config.edges,
            "cameraTiltDegrees": config.camera_tilt_degrees,
            "cameraStepDegrees": config.camera_step_degrees,
            "cameraZoom": config.camera_zoom,
            "browser": self.browser,
        }
        payload.update(extra)
        return payload

    @staticmethod
    def _decode(response: dict) -> bytes:
        frame = response.get("frame") or ""
        if not frame:
            raise RendererError("response carried no frame")
        _, _, b64 = frame.partition(",")
        return base64.b64decode(b64 or frame)

    def _call(self, message: dict) -> dict:
        with self._lock:
            response = self._request(message)
        if not response.get("ok", False):
            raise RendererError(
                f"{message.get('command')} failed: {response.get('error')}"
            )
        return response

    # -- commands -----------------------------------------------------------
    def init(self) -> bytes:
        """Launch the browser and return the first frame."""
        return self._decode(self._call(self._payload(command="init")))

    def render(self, snapshot: dict) -> bytes:
        """Draw the authoritative ``_render_state`` and return the PNG.

        The session is reused whenever the render options are unchanged, so this
        is one browser round trip, not a browser launch.
        """
        if not snapshot or not snapshot.get("level_id") or not isinstance(
            snapshot.get("actors"), list
        ):
            raise RendererError(
                "snapshot must be a bridge _render_state with level_id and actors"
            )
        return self._decode(
            self._call(self._payload(command="render", snapshot=snapshot))
        )

    def frame(self) -> bytes:
        """Re-capture the current session without changing it."""
        return self._decode(self._call({"command": "frame"}))
