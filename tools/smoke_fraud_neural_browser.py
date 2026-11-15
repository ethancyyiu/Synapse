"""One focused Fraud UI and original Science page browser smoke check."""
from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8771"
SCREENSHOT = Path("data/fraud_neural/browser-check.png")

with sync_playwright() as playwright:
    try:
        browser = playwright.chromium.launch(headless=True)
    except Exception:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.goto(f"{BASE}/fraud.html", wait_until="domcontentloaded")
        assert response and response.status == 200
        page.wait_for_function("document.querySelector('#workspace-status')?.textContent.includes('Analyzed')", timeout=30000)
        page.locator("#neural-next").wait_for(state="visible")
        page.wait_for_function("!document.querySelector('#neural-next').disabled", timeout=30000)
        source = page.locator("#source-mode").inner_text()
        count = page.locator("#tx-list .tx").count()
        assert count > 0, "No purchase cards loaded"
        cue = page.locator("#neural-cue").inner_text()
        assert cue.startswith("Cue "), cue
        page.locator("#neural-next").click()
        trace = page.locator("#neural-trace").inner_text()
        assert "DNge175 spike counts" in trace, trace
        summary = page.locator("#neural-summary").inner_text()
        assert "Predeclared synthetic teaching gate:" in summary, summary
        page.locator("#neural-probe").click()
        page.wait_for_function(
            "document.querySelector('#neural-probe-result')?.textContent.includes('Frozen Fly B chose') || document.querySelector('#neural-probe-result')?.textContent.includes('Probe failed')",
            timeout=120000,
        )
        probe = page.locator("#neural-probe-result").inner_text()
        assert probe.startswith("Frozen Fly B chose"), probe
        page.locator("#neural-teaching").scroll_into_view_if_needed()
        page.screenshot(path=str(SCREENSHOT))
        science = page.goto(f"{BASE}/index.html", wait_until="domcontentloaded")
        assert science and science.status == 200
        assert "Synapse" in page.title()
        assert page.locator("#hero-play-fast").count() == 1
        assert not errors, errors
        print(json.dumps({"fraud": "passed", "source": source, "purchases": count,
                          "cue": cue, "trace": "measured DN spikes", "probe": probe,
                          "science": "passed", "page_errors": errors}))
    finally:
        browser.close()
