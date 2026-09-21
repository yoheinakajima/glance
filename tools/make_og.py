"""Render the link-preview card `site/og_card.html` (written by tools/make_site.py) to `site/og.png`, 1200 x 630, with a local Chrome.
Fetches only the page's own web fonts. Run after make_site.py whenever the card's numbers change.

uv run python tools/make_og.py
"""
import pathlib
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
out = ROOT / "site/og.png"
out.unlink(missing_ok=True)
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:  # Chrome may still be writing its profile while it shuts down
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}", "--hide-scrollbars", "--window-size=1200,630",
                             "--virtual-time-budget=4000", f"--screenshot={out}", (ROOT / "site/og_card.html").as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        if out.exists() and out.stat().st_size > 0:
            break
        time.sleep(1)
    time.sleep(1)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
print("wrote", out, out.stat().st_size // 1024, "KB")
