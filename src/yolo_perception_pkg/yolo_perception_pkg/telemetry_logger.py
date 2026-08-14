"""
TH OWL — Telemetry Logger & HUD Rendering Module
=================================================
Provides two independent utilities:

  1. TelemetryLogger — Async background-thread CSV writer.
     Logs per-frame metrics to results/telemetry_log.csv without
     blocking the ROS2 image callback thread.

  2. compute_weather_entropy() — Normalized Shannon entropy of the
     classifier softmax probability vector (uncertainty metric).

  3. render_hud_panel() — Renders a dark side-panel HUD alongside a
     BGR frame displaying live pipeline telemetry. Shared between
     perception_carla_node and perception_offline_node.

Usage:
    from yolo_perception_pkg.telemetry_logger import (
        TelemetryLogger, compute_weather_entropy, render_hud_panel
    )
"""

import os
import csv
import queue
import threading
import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

# -------------------------------------------------------------------------------------
#  CSV Output Configuration
# -------------------------------------------------------------------------------------

_DEFAULT_LOG_DIR  = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'results')
_DEFAULT_LOG_FILE = os.path.join(_DEFAULT_LOG_DIR, 'telemetry_log.csv')

CSV_FIELDNAMES = [
    'timestamp_iso',
    'frame_id',
    'weather_label',
    'weather_confidence',
    'weather_entropy',
    'gating_reason',
    'route',
    'cwqi',
    'severity_level',
    'classifier_latency_ms',
    'restorer_latency_ms',
    'total_latency_ms',
]

# -------------------------------------------------------------------------------------
#  HUD Color Palette (BGR format for OpenCV)
# -------------------------------------------------------------------------------------

_COL_BG        = (30,  30,  30)    # Dark panel background
_COL_TITLE     = (0,  200, 255)    # Amber/gold — title bar
_COL_SECTION   = (0,  180, 255)    # Amber/gold — section headers
_COL_TEXT      = (220, 220, 220)   # Light grey — normal text
_COL_SEPARATOR = (80,  80,  80)    # Dim separator lines
_COL_RESTORED  = (0,  200, 100)    # Green — RESTORED route
_COL_BYPASS    = (0,  200, 255)    # Cyan — BYPASS route
_COL_SEV_LOW   = (0,  200,   0)    # Green  — severity 1–2
_COL_SEV_MED   = (0,  200, 255)    # Amber  — severity 3
_COL_SEV_HIGH  = (0,   80, 220)    # Red    — severity 4–5


# -------------------------------------------------------------------------------------
#  TelemetryLogger
# -------------------------------------------------------------------------------------

class TelemetryLogger:
    """
    Non-blocking per-frame telemetry CSV logger with async background writer.

    A daemon thread drains a bounded queue and appends rows to the CSV file.
    The calling thread (ROS2 callback) only enqueues a dict — it never blocks
    on disk I/O. If the queue fills up (>500 pending records) the new record
    is silently dropped to prevent backpressure on the callback.

    Lifecycle:
        logger = TelemetryLogger()          # starts background thread
        logger.log({...})                   # called per frame (non-blocking)
        logger.shutdown()                   # call in node.destroy_node()
    """

    def __init__(self, log_path: str = _DEFAULT_LOG_FILE) -> None:
        self._log_path    = log_path
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._stop_event  = threading.Event()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self._log_path)), exist_ok=True)

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name='TelemetryCSVWriter',
            daemon=True,
        )
        self._writer_thread.start()

    # ------------------------------------------------------------------

    def log(self, record: Dict) -> None:
        """
        Submit a telemetry record for async CSV writing.

        Always non-blocking. Inserts a UTC ISO-8601 timestamp if not present.
        Drops the record silently if the internal queue is full.

        Args:
            record: Dict matching CSV_FIELDNAMES (extra keys are ignored).
        """
        record.setdefault(
            'timestamp_iso',
            datetime.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
        )
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass  # Intentional drop — never block the ROS2 callback

    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Signal the writer to stop and flush remaining queued records.
        Blocks at most 3 seconds waiting for the thread to finish.
        Call from node.destroy_node().
        """
        self._stop_event.set()
        self._writer_thread.join(timeout=3.0)

    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """Background thread body: write queued records to CSV."""
        file_exists = os.path.isfile(self._log_path)
        with open(self._log_path, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(
                fh, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
                fh.flush()

            while not self._stop_event.is_set():
                try:
                    record = self._queue.get(timeout=0.1)
                    writer.writerow(record)
                    fh.flush()
                except queue.Empty:
                    continue

            # Drain any remaining records on graceful shutdown
            while True:
                try:
                    record = self._queue.get_nowait()
                    writer.writerow(record)
                except queue.Empty:
                    break
            fh.flush()


# -------------------------------------------------------------------------------------
#  Utility: Classifier Uncertainty (Normalized Entropy)
# -------------------------------------------------------------------------------------

def compute_weather_entropy(probabilities: List[float]) -> float:
    """
    Compute the normalized Shannon entropy of a classifier softmax output.

    A low entropy (≈ 0) means the model is highly confident (one class
    dominates). A high entropy (≈ 1) means the model is uncertain across
    all classes — useful as a calibration / OOD signal.

    Args:
        probabilities: Full softmax probability vector (list of floats, sums to 1).

    Returns:
        Normalized entropy in [0.0, 1.0].
    """
    probs = np.array(probabilities, dtype=np.float64)
    probs = probs[probs > 0.0]
    raw_entropy = float(-np.sum(probs * np.log2(probs)))
    n_classes   = len(probabilities)
    max_entropy = float(np.log2(n_classes)) if n_classes > 1 else 1.0
    return min(raw_entropy / max_entropy, 1.0) if max_entropy > 0.0 else 0.0


# -------------------------------------------------------------------------------------
#  HUD Rendering
# -------------------------------------------------------------------------------------

def render_hud_panel(
    frame:       np.ndarray,
    panel_width: int,
    hud_data:    Dict,
) -> np.ndarray:
    """
    Render a dark telemetry HUD side-panel and concatenate it to the right
    edge of the given frame.

    Args:
        frame:       BGR uint8 numpy array (H, W, 3). Not modified in-place.
        panel_width: Width in pixels of the HUD panel column.
        hud_data:    Dictionary of display values. Expected keys:

            frame_num            (int|str)  — Frame counter / frame_id
            weather_class        (str)      — Predicted weather label
            confidence           (float)   — Classifier confidence [0–100]
            weather_entropy      (float)   — Normalized entropy [0–1]
            gating_reason        (str)     — 'bypass'|'low_light'|'glare'|'blur'
            route                (str)     — 'BYPASS' or 'RESTORED'
            cwqi                 (float)   — CWQI degradation index [0–1]
            severity_level       (int)     — Severity level [1–5]
            severity_label       (str)     — Severity label string
            classifier_latency_ms(float)  — Classifier inference latency (ms)
            restorer_latency_ms  (float)  — Restorer inference latency (ms)
            total_latency_ms     (float)  — Full callback latency (ms)
            fps                  (float)  — Frames per second
            device               (str)    — 'cuda:0' or 'cpu'

    Returns:
        canvas: numpy array (H, W + panel_width, 3) with frame + HUD.
    """
    h, w = frame.shape[:2]
    panel = np.full((h, panel_width, 3), _COL_BG, dtype=np.uint8)

    ml = 14                   # left margin
    mr = panel_width - 14     # right margin

    def _text(txt: str, y: int, color=_COL_TEXT,
               scale: float = 0.46, bold: bool = False) -> None:
        cv2.putText(panel, txt, (ml, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    1, cv2.LINE_AA)

    def _bar(y: int, fraction: float, color: tuple) -> None:
        bw = mr - ml
        cv2.rectangle(panel, (ml, y), (mr, y + 9), (55, 55, 55), -1)
        fill = int(bw * max(0.0, min(1.0, fraction)))
        if fill > 0:
            cv2.rectangle(panel, (ml, y), (ml + fill, y + 9), color, -1)

    def _sep(y: int) -> None:
        cv2.line(panel, (ml, y), (mr, y), _COL_SEPARATOR, 1)

    y = 0

    # ── Title Bar ─────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (panel_width, 36), (18, 18, 18), -1)
    cv2.putText(panel, 'WEATHER PERCEPTION PIPELINE', (ml, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, _COL_TITLE, 1, cv2.LINE_AA)
    y = 46

    _text(f"Frame: {hud_data.get('frame_num', '--')}", y);  y += 20
    _sep(y);                                                  y += 16

    # ── WEATHER PROFILE ───────────────────────────────────────────
    _text('WEATHER PROFILE', y, _COL_SECTION, 0.46, bold=True); y += 20
    _text(f"  Class:    {hud_data.get('weather_class', '---')}", y);  y += 18
    conf = hud_data.get('confidence', 0.0)
    _text(f"  Conf:     {conf:.1f}%", y);          y += 16
    _bar(y, conf / 100.0, _COL_SECTION);           y += 22
    entr = hud_data.get('weather_entropy', 0.0)
    _text(f"  Entropy:  {entr:.3f}", y);            y += 20
    _sep(y);                                        y += 16

    # ── CWQI SEVERITY ─────────────────────────────────────────────
    _text('CWQI SEVERITY', y, _COL_SECTION, 0.46, bold=True); y += 20
    cwqi = hud_data.get('cwqi', 0.0)
    sev  = hud_data.get('severity_level', 1)
    slbl = hud_data.get('severity_label', 'Excellent')
    sev_col = (_COL_SEV_LOW  if sev <= 2 else
               _COL_SEV_MED  if sev == 3 else _COL_SEV_HIGH)
    _text(f"  CWQI:     {cwqi:.3f}", y);             y += 16
    _bar(y, cwqi, sev_col);                          y += 22
    _text(f"  Level {sev}: {slbl}", y, sev_col, bold=True); y += 20
    _sep(y);                                          y += 16

    # ── PIPELINE ROUTING ──────────────────────────────────────────
    _text('PIPELINE ROUTING', y, _COL_SECTION, 0.46, bold=True); y += 20
    route   = hud_data.get('route', 'BYPASS')
    reason  = hud_data.get('gating_reason', '')
    r_col   = _COL_RESTORED if route == 'RESTORED' else _COL_BYPASS
    _text(f"  Track:    {route}", y, r_col, bold=True);   y += 18
    _text(f"  Reason:   {reason}", y);                     y += 20
    _sep(y);                                               y += 16

    # ── PERFORMANCE ───────────────────────────────────────────────
    _text('PERFORMANCE', y, _COL_SECTION, 0.46, bold=True); y += 20
    _text(f"  Cls:      {hud_data.get('classifier_latency_ms', 0.0):.1f} ms", y); y += 18
    _text(f"  Rest:     {hud_data.get('restorer_latency_ms', 0.0):.1f} ms",   y); y += 18
    _text(f"  Total:    {hud_data.get('total_latency_ms', 0.0):.1f} ms",      y); y += 18
    _text(f"  FPS:      {hud_data.get('fps', 0.0):.1f}",                      y); y += 20
    _sep(y);                                                                       y += 16

    # ── DEVICE ────────────────────────────────────────────────────
    _text('DEVICE', y, _COL_SECTION, 0.46, bold=True); y += 20
    _text(f"  {hud_data.get('device', 'cpu')}", y);    y += 20

    return np.hstack([frame, panel])
