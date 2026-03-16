"""
BCH Payment Gate — NeuroSymbolic V17
=====================================

Requires a one-time BCH payment to a fixed address before the engine
can be used.  Payment is verified via the Blockchair public API (no API
key required for moderate usage).

Once verified, a local access token is written to disk so the check
does not require an internet hit on every subsequent run.

Architecture
───────────
  1. On startup, look for a valid local token file (.v17_access_token).
  2. If the token is valid (correct hash + not expired), proceed.
  3. If no token, check the Blockchair API for any incoming transaction
     to BCH_PAYMENT_ADDRESS with value >= REQUIRED_BCH.
  4. If paid, write the token file and proceed.
  5. If not paid, print the payment details and block execution.

Constants to customise
──────────────────────
  BCH_PAYMENT_ADDRESS  — your BCH address
  REQUIRED_BCH         — minimum BCH amount (float, e.g. 0.01)
  LICENSE_DURATION_DAYS — how many days a paid token stays valid
                          (set to 0 for lifetime / never-expiring)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Configurable constants ────────────────────────────────────────────────

BCH_PAYMENT_ADDRESS   : str   = "bitcoincash:qp5nsh9czn6jadwdzwldgnn86y9358l4rczv0tu37e"
# 10 cents USD at ~$464/BCH = 0.000216 BCH  (update periodically as price changes)
REQUIRED_BCH          : float = 0.001      # ≈ $0.10 USD per session
LICENSE_DURATION_DAYS : int   = 1            # 0 = lifetime; >0 = expires after N days
TOKEN_FILE            : Path  = Path(".v17_access_token")
BLOCKCHAIR_API_URL    : str   = (
    "https://api.blockchair.com/bitcoin-cash/dashboards/address/{address}"
    "?transaction_details=false"
)

# ── Session timeout ───────────────────────────────────────────────────────
# After SESSION_TIMEOUT_SECONDS of activity, the Generate button fades out
# and is disabled until the page is refreshed (GUI) or the script re-run (CLI).
SESSION_TIMEOUT_SECONDS : int = 600   # 10 minutes

# ── Internal helpers ──────────────────────────────────────────────────────

_SALT = "v17-neurosymbolic-bch-gate-salt-2025"


def _token_hash(address: str, timestamp: float) -> str:
    payload = f"{address}|{timestamp:.0f}|{_SALT}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_token(address: str) -> None:
    ts = time.time()
    data = {
        "address"  : address,
        "timestamp": ts,
        "expires"  : ts + LICENSE_DURATION_DAYS * 86400 if LICENSE_DURATION_DAYS > 0 else 0,
        "hash"     : _token_hash(address, ts),
    }
    TOKEN_FILE.write_text(json.dumps(data, indent=2))
    print(f"[BCH] ✅  Access token written to {TOKEN_FILE}")
    # Start the session clock the moment payment is confirmed/written
    _session_timer.start()
    print(f"[Session] ⏱  {SESSION_TIMEOUT_SECONDS // 60}-minute session timer started.")


def _read_token() -> Optional[dict]:
    if not TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(TOKEN_FILE.read_text())
        return data
    except Exception:
        return None


def _token_valid(data: dict) -> bool:
    expected_hash = _token_hash(data["address"], data["timestamp"])
    if data.get("hash") != expected_hash:
        return False
    expires = data.get("expires", 0)
    if expires > 0 and time.time() > expires:
        return False
    return True


def _query_blockchair(address: str) -> Optional[dict]:
    """
    Returns the Blockchair address summary dict, or None on error.
    The bare address (without 'bitcoincash:' prefix) is used in the URL.
    """
    bare = address.replace("bitcoincash:", "").strip()
    url  = BLOCKCHAIR_API_URL.format(address=bare)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NeuroSymbolicV17/1.0 (payment-gate)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw  = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        print(f"[BCH] ⚠  Blockchair HTTP error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"[BCH] ⚠  Network error querying Blockchair: {e.reason}")
    except Exception as e:
        print(f"[BCH] ⚠  Unexpected error: {e}")
    return None


def _get_received_bch(address: str) -> float:
    """
    Returns total BCH received by *address* (confirmed, from Blockchair).
    Blockchair reports satoshis in .data[address].address.received.
    """
    data = _query_blockchair(address)
    if data is None:
        return 0.0
    try:
        bare    = address.replace("bitcoincash:", "").strip()
        summary = data["data"][bare]["address"]
        # Blockchair uses satoshis
        satoshis_received = summary.get("received", 0) or 0
        return satoshis_received / 1e8
    except (KeyError, TypeError):
        return 0.0


# ── Public API ────────────────────────────────────────────────────────────

def check_payment_cli(silent: bool = False) -> bool:
    """
    Gate the process: return True if access is granted, False otherwise.

    The session timer is started HERE, on confirmed payment — not before.
    A returning user with a cached token gets a fresh 10-minute window
    each time they launch the script.
    """
    # 1. Valid local token → fast path, start timer for this run
    token = _read_token()
    if token and _token_valid(token):
        if not silent:
            addr  = token["address"]
            ts    = datetime.fromtimestamp(token["timestamp"], tz=timezone.utc)
            exp   = token.get("expires", 0)
            exp_s = "lifetime" if exp == 0 else datetime.fromtimestamp(exp, tz=timezone.utc).strftime("%Y-%m-%d")
            print(
                f"\n[BCH] ✅  Licensed copy — address {addr}\n"
                f"         Activated : {ts.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"         Expires   : {exp_s}\n"
            )
        # New process = new session window
        _session_timer.start()
        print(f"[Session] ⏱  {SESSION_TIMEOUT_SECONDS // 60}-minute session started.")
        return True

    # 2. No valid token — query the blockchain
    if not silent:
        _print_payment_banner()

    print("[BCH] 🔍  Checking blockchain for payment …")
    received = _get_received_bch(BCH_PAYMENT_ADDRESS)
    print(f"[BCH]     Total received by address: {received:.8f} BCH")

    if received >= REQUIRED_BCH:
        print(f"[BCH] ✅  Payment confirmed ({received:.8f} BCH ≥ {REQUIRED_BCH} BCH required).")
        _write_token(BCH_PAYMENT_ADDRESS)  # _write_token calls _session_timer.start()
        return True

    print(
        f"\n[BCH] ❌  Payment not yet confirmed.\n"
        f"         Required : {REQUIRED_BCH} BCH\n"
        f"         Received : {received:.8f} BCH\n"
        f"\n         Send at least {REQUIRED_BCH} BCH to:\n"
        f"         {BCH_PAYMENT_ADDRESS}\n"
        f"\n         Then re-run the script.\n"
    )
    return False


def _print_payment_banner() -> None:
    width = 68
    top   = "╔" + "═" * width + "╗"
    bot   = "╚" + "═" * width + "╝"
    mid   = lambda s: "║  " + s.ljust(width - 2) + "║"
    sep   = "╠" + "═" * width + "╣"

    lines = [
        top,
        mid("NeuroSymbolic V17 — Licensed Access"),
        sep,
        mid("This software requires a one-time BCH payment."),
        mid(""),
        mid(f"  Send ≥ {REQUIRED_BCH} BCH to:"),
        mid(""),
        mid(f"  {BCH_PAYMENT_ADDRESS}"),
        mid(""),
        mid("  Payment is verified automatically via the Blockchair"),
        mid("  public blockchain API.  No account required."),
        mid(""),
        mid("  After sending, re-run the script.  Access is granted"),
        mid("  as soon as the transaction appears on-chain."),
        bot,
    ]
    print("\n" + "\n".join(lines) + "\n")


# ── Session tracker (CLI + shared state for GUI) ─────────────────────────

class SessionTimer:
    """
    Tracks one 10-minute session window.

    Usage (CLI):
        timer = SessionTimer()
        timer.start()
        ...
        if timer.expired():
            print("Session expired.")
            sys.exit(0)

    Usage (GUI):
        The GUI reads timer.seconds_remaining() and timer.expired() to
        fade the Generate button and show a countdown.
    """

    def __init__(self, duration: int = SESSION_TIMEOUT_SECONDS):
        self.duration  = duration
        self._start_ts : Optional[float] = None

    def start(self) -> None:
        self._start_ts = time.time()

    def elapsed(self) -> float:
        if self._start_ts is None:
            return 0.0
        return time.time() - self._start_ts

    def seconds_remaining(self) -> float:
        return max(0.0, self.duration - self.elapsed())

    def expired(self) -> bool:
        if self._start_ts is None:
            return False
        return self.elapsed() >= self.duration

    def progress(self) -> float:
        """0.0 = just started, 1.0 = fully expired."""
        if self._start_ts is None:
            return 0.0
        return min(1.0, self.elapsed() / self.duration)

    def status_line(self) -> str:
        if self._start_ts is None:
            return "Session not started."
        if self.expired():
            return "⏱  Session EXPIRED — please restart the script."
        rem = int(self.seconds_remaining())
        m, s = divmod(rem, 60)
        pct  = int(self.progress() * 100)
        bar_len = 20
        filled  = int(bar_len * self.progress())
        bar     = "█" * filled + "░" * (bar_len - filled)
        return f"⏱  {m:02d}:{s:02d} remaining  [{bar}] {pct}%"


# Module-level shared timer (GUI reads this from callbacks)
_session_timer = SessionTimer()


def get_session_timer() -> SessionTimer:
    """Return the module-level session timer (shared across GUI callbacks)."""
    return _session_timer


# ── Gradio tab builder ────────────────────────────────────────────────────

def build_gradio_tab(gui_instance) -> None:  # type: ignore[override]
    """
    Call this inside a gr.Blocks() context to append a "Payment / License"
    tab.  *gui_instance* is the V17GUI object (for access to engine state).

    Usage (inside launch_gui):
        with gr.Blocks(...) as app:
            ...existing tabs...
            with gr.Tab("License"):
                build_gradio_tab(gui)
    """
    import gradio as gr  # local import so the module works without gradio

    gr.Markdown(
        "## BCH License & Payment\n"
        "This tab lets you verify your payment status and check the "
        "blockchain in real time."
    )

    addr_display = gr.Textbox(
        label    = "Payment Address",
        value    = BCH_PAYMENT_ADDRESS,
        interactive = False,
        lines    = 1,
    )
    required_display = gr.Textbox(
        label    = "Required Amount",
        value    = f"{REQUIRED_BCH} BCH",
        interactive = False,
        lines    = 1,
    )

    with gr.Row():
        check_btn   = gr.Button("🔍  Check Payment Now", variant="primary")
        refresh_btn = gr.Button("🔄  Re-read Local Token")

    status_out = gr.Textbox(label="Payment Status", lines=12, interactive=False)

    def _check_payment_gui() -> str:
        lines = []

        # Local token check
        token = _read_token()
        if token and _token_valid(token):
            ts  = datetime.fromtimestamp(token["timestamp"], tz=timezone.utc)
            exp = token.get("expires", 0)
            exp_s = "Lifetime (never expires)" if exp == 0 else \
                    datetime.fromtimestamp(exp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            # Token already valid — start/reset timer for this session if not already running
            if not _session_timer.expired() and _session_timer._start_ts is None:
                _session_timer.start()
            lines += [
                "✅  LICENSED COPY",
                f"   Address   : {token['address']}",
                f"   Activated : {ts.strftime('%Y-%m-%d %H:%M UTC')}",
                f"   Expires   : {exp_s}",
                "",
                "No further payment required.",
                "",
                f"⏱  Session timer: {_session_timer.status_line()}",
            ]
            return "\n".join(lines)

        # Live blockchain check
        lines.append("📡  Querying Blockchair API…")
        received = _get_received_bch(BCH_PAYMENT_ADDRESS)
        lines.append(f"   Address  : {BCH_PAYMENT_ADDRESS}")
        lines.append(f"   Received : {received:.8f} BCH")
        lines.append(f"   Required : {REQUIRED_BCH} BCH")
        lines.append("")

        if received >= REQUIRED_BCH:
            lines.append("✅  PAYMENT CONFIRMED — writing access token…")
            _write_token(BCH_PAYMENT_ADDRESS)  # starts the timer
            lines.append("   Token saved. Your 10-minute session has started!")
            lines.append("")
            lines.append(f"⏱  {_session_timer.status_line()}")
            lines.append("")
            lines.append("▶  Switch to the Generate tab to begin.")
        else:
            remaining_bch = max(0.0, REQUIRED_BCH - received)
            lines += [
                "❌  PAYMENT PENDING  (timer not started)",
                f"   Still needed : {remaining_bch:.8f} BCH",
                "",
                "   Send BCH to the address above, then click",
                "   'Check Payment Now' again after the transaction",
                "   has been broadcast to the network.",
            ]
        return "\n".join(lines)

    def _refresh_token_gui() -> str:
        token = _read_token()
        if token is None:
            return "No local access token found."
        if not _token_valid(token):
            return "Local token found but it is INVALID or EXPIRED."
        ts  = datetime.fromtimestamp(token["timestamp"], tz=timezone.utc)
        exp = token.get("expires", 0)
        exp_s = "Lifetime" if exp == 0 else \
                datetime.fromtimestamp(exp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"✅  Valid local token\n"
            f"   Address   : {token['address']}\n"
            f"   Activated : {ts.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"   Expires   : {exp_s}\n"
            f"   Hash      : {token['hash'][:24]}…"
        )

    check_btn.click  (_check_payment_gui,  outputs=status_out)
    refresh_btn.click(_refresh_token_gui,  outputs=status_out)
