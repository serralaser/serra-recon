#!/usr/bin/env python3
"""
Serra automated reconstruction service.

Polls Cloudflare Worker for queued scan jobs, reads drawings via xAI Grok
vision API, builds laser-cut-ready DXF, marks jobs done.

Env:
  RECON_API_BASE   default https://serra-recon-jobs.lively-shadow-9fe4.workers.dev
  AGENT_KEY        Worker agent secret (required)
  XAI_API_KEY      xAI API key for vision (required for new drawings)
  POLL_SECONDS     default 20
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from dxf_writer import DXFWriter

API = os.environ.get(
    "RECON_API_BASE", "https://serra-recon-jobs.lively-shadow-9fe4.workers.dev"
).rstrip("/")
AGENT_KEY = os.environ.get("AGENT_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
POLL = float(os.environ.get("POLL_SECONDS", "20"))
XAI_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-2-vision-1212")


def log(msg: str) -> None:
    print(msg, flush=True)


def http_json(
    method: str,
    url: str,
    body: Optional[dict] = None,
    headers: Optional[dict] = None,
    raw: Optional[bytes] = None,
    timeout: int = 120,
) -> Any:
    h = dict(headers or {})
    data = None
    if raw is not None:
        data = raw
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_resp = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype or raw_resp[:1] in (b"{", b"["):
                return json.loads(raw_resp.decode("utf-8"))
            return raw_resp
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {err_body}") from e


def list_queued() -> List[dict]:
    data = http_json(
        "GET",
        f"{API}/jobs?status=queued",
        headers={"Authorization": f"Bearer {AGENT_KEY}"},
    )
    return data.get("jobs") or []


def download_source(job_id: str) -> tuple[bytes, str]:
    req = urllib.request.Request(
        f"{API}/jobs/{job_id}/source",
        headers={"Authorization": f"Bearer {AGENT_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        ctype = resp.headers.get("Content-Type", "application/octet-stream")
        return data, ctype


def patch_done(job_id: str, dxf_bytes: bytes, summary: dict) -> dict:
    payload = {
        "status": "done",
        "dxfBase64": base64.b64encode(dxf_bytes).decode("ascii"),
        "summary": summary,
    }
    return http_json(
        "PATCH",
        f"{API}/jobs/{job_id}",
        body=payload,
        headers={"Authorization": f"Bearer {AGENT_KEY}"},
    )


def patch_failed(job_id: str, error: str) -> dict:
    return http_json(
        "PATCH",
        f"{API}/jobs/{job_id}",
        body={"status": "failed", "error": error},
        headers={"Authorization": f"Bearer {AGENT_KEY}"},
    )


VISION_PROMPT = """You are extracting geometry for laser/waterjet cutting from an engineering drawing image.

Return ONLY valid JSON (no markdown) with this schema:
{
  "units": "in" or "mm",
  "overall": [width, height],
  "outer": "rectangle" | "circle" | "polygon",
  "outer_size": [width, height] for rectangle, or [diameter] for circle,
  "holes": [
    {"x": center_x, "y": center_y, "d": diameter}
  ],
  "assumptions": ["list of assumptions"],
  "notes": "brief"
}

Rules:
- Origin at bottom-left of the outer profile.
- Prefer labeled dimensions over scaling pixels.
- If a hole diameter is labeled once and holes look identical, apply it to all.
- If vertical hole pairs have no edge offsets, center them in the height.
- If horizontal span between hole columns is labeled, place columns symmetrically unless edge distances are given.
- overall and outer_size should match the outer cut boundary.
- holes coordinates are centers in the same coordinate system as overall.
"""


def vision_extract(image_bytes: bytes, content_type: str) -> dict:
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY not set")

    # normalize mime
    mime = "image/png"
    if "jpeg" in content_type or "jpg" in content_type:
        mime = "image/jpeg"
    elif "webp" in content_type:
        mime = "image/webp"
    elif "pdf" in content_type:
        raise RuntimeError("PDF input: convert first page to PNG before vision")

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    body = {
        "model": XAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
        "temperature": 0.1,
    }
    result = http_json(
        "POST",
        XAI_URL,
        body=body,
        headers={
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=180,
    )
    text = result["choices"][0]["message"]["content"]
    # strip markdown fences if any
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def build_dxf_from_extract(data: dict) -> tuple[bytes, dict]:
    units = str(data.get("units") or "in").lower()
    if units.startswith("mm"):
        units = "mm"
    else:
        units = "inches"

    overall = data.get("overall") or data.get("outer_size")
    if not overall or len(overall) < 2:
        raise RuntimeError(f"missing overall size in extract: {data}")

    w, h = float(overall[0]), float(overall[1])
    doc = DXFWriter(units=units)
    outer = str(data.get("outer") or "rectangle").lower()

    if outer == "circle":
        d = float((data.get("outer_size") or [w])[0])
        doc.circle(d / 2.0, d / 2.0, d / 2.0)
        w = h = d
    else:
        doc.rect(w, h)

    holes = data.get("holes") or []
    for hole in holes:
        x = float(hole["x"])
        y = float(hole["y"])
        d = float(hole["d"])
        if d <= 0:
            continue
        doc.circle(x, y, d / 2.0)

    summary = {
        "overall": [w, h],
        "units": "in" if units.startswith("in") else "mm",
        "holes": len(holes),
        "assumptions": data.get("assumptions") or [],
        "notes": data.get("notes") or "",
    }
    return doc.to_bytes(), summary


def process_job(job: dict) -> None:
    jid = job["id"]
    fname = (job.get("source") or {}).get("filename") or ""
    log(f"Processing {jid} ({fname})")

    # mark running (best-effort)
    try:
        http_json(
            "PATCH",
            f"{API}/jobs/{jid}",
            body={"status": "running"},
            headers={"Authorization": f"Bearer {AGENT_KEY}"},
        )
    except Exception:
        pass

    raw, ctype = download_source(jid)
    log(f"  source {len(raw)} bytes, {ctype}")

    if raw[:4] == b"%PDF":
        patch_failed(
            jid,
            "PDF not supported in v1 recon service — export page as PNG or enable PDF rasterize",
        )
        return

    extract = vision_extract(raw, ctype)
    log(f"  extract: {json.dumps(extract)[:300]}")
    dxf_bytes, summary = build_dxf_from_extract(extract)
    result = patch_done(jid, dxf_bytes, summary)
    log(f"  done: {result}")


def main() -> None:
    if not AGENT_KEY:
        log("ERROR: AGENT_KEY env required")
        sys.exit(1)
    if not XAI_API_KEY:
        log("WARNING: XAI_API_KEY not set — vision extraction will fail")

    log(f"Serra recon service")
    log(f"  API={API}")
    log(f"  poll={POLL}s")
    log(f"  vision={'yes' if XAI_API_KEY else 'NO'}")

    while True:
        try:
            jobs = list_queued()
            if not jobs:
                log(f"No queued jobs")
            for job in jobs:
                try:
                    process_job(job)
                except Exception as e:
                    log(f"Job {job.get('id')} failed: {e}")
                    traceback.print_exc()
                    try:
                        patch_failed(job["id"], str(e)[:500])
                    except Exception:
                        pass
        except Exception as e:
            log(f"Poll error: {e}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
