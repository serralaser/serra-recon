#!/usr/bin/env python3
"""
Serra automated reconstruction service (R2 direct).

Polls the serra-recon R2 bucket for queued jobs (bypasses Worker HTTP / CF 1010),
reads drawings via xAI vision, builds DXF, writes result back to R2.

Env:
  R2_ACCOUNT_ID
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET          default serra-recon
  XAI_API_KEY
  XAI_MODEL          default grok-4.6
  POLL_SECONDS       default 5
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
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config

from dxf_writer import DXFWriter

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET = os.environ.get("R2_BUCKET", "serra-recon").strip()
XAI_API_KEY = os.environ.get("XAI_API_KEY", "").strip()
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4.6").strip()
POLL = float(os.environ.get("POLL_SECONDS", "5"))
XAI_URL = "https://api.x.ai/v1/chat/completions"


def log(msg: str) -> None:
    print(msg, flush=True)


def r2_client():
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        raise RuntimeError("R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY required")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def list_queued(s3) -> List[dict]:
    jobs = []
    token = None
    while True:
        kwargs = {"Bucket": R2_BUCKET, "MaxKeys": 500}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            key = obj["Key"]
            if not key.endswith("/job.json"):
                continue
            try:
                body = s3.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
                job = json.loads(body)
                if job.get("status") == "queued":
                    jobs.append(job)
            except Exception as e:
                log(f"  skip {key}: {e}")
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    jobs.sort(key=lambda j: str(j.get("createdAt") or ""))
    return jobs


def get_bytes(s3, key: str) -> bytes:
    return s3.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()


def put_json(s3, key: str, data: dict) -> None:
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=json.dumps(data).encode("utf-8"),
        ContentType="application/json",
    )


def put_bytes(s3, key: str, data: bytes, content_type: str) -> None:
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def http_json(method: str, url: str, body: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 180) -> Any:
    h = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {err}") from e


VISION_PROMPT = """You extract laser/waterjet cut geometry from an engineering drawing.

Return ONLY valid JSON (no markdown). Use this schema (include only fields that apply):

{
  "units": "in" or "mm",
  "part_type": "plate" | "flange" | "ring" | "bracket",
  "thickness": number or null,
  "material": "string or null",
  "origin": "center" | "bottom_left",
  "outer": "rectangle" | "circle" | "circle_with_tabs",
  "body_od": number,
  "body_id": number,
  "outer_size": [width, height] for rectangle, or [diameter] for simple circle,
  "overall": [width, height],
  "bolt_circles": [
    {"d": bolt_circle_diameter, "count": N, "hole_d": diameter, "start_angle_deg": 0, "angles_deg": null}
  ],
  "holes": [
    {"x": cx, "y": cy, "d": diameter}
  ],
  "tabs": {
    "count": 4,
    "angles_deg": [0, 90, 180, 270],
    "width": number,
    "outer_r": number,
    "fillet_r": number,
    "hole_bhc": number,
    "hole_d": number
  },
  "assumptions": ["..."],
  "notes": "brief"
}

Rules:
- Prefer labeled dimensions over measuring pixels.
- origin "center" for round/flange parts; coordinates are then relative to center.
- origin "bottom_left" for rectangular plates; hole x,y from bottom-left of outer.
- body_od = outer diameter of main circular body (not including tab tips if tabs protrude).
- body_id = center hole diameter when present.
- For flanges with tabs: outer="circle_with_tabs", fill tabs{}, body_od, body_id.
- tabs.outer_r = radius from center to tab tip (use a labeled radial/overall dimension when present).
- tabs.fillet_r = tip/corner radius (R0.50 TYP etc).
- bolt_circles: use for equally spaced holes on a diameter. start_angle_deg defaults 0.
  If tabs are on 0/90/180/270 and a BHC should sit between tabs, set start_angle_deg to 15.
- holes[]: individual holes not on a bolt circle (or extra holes).
- If a diameter is labeled once for identical holes, apply to all.
- overall should be the bounding box of the cut outer profile.
- Never invent features that are not on the drawing; list uncertainties in assumptions.
"""


def pdf_to_png(pdf_bytes: bytes, max_side: int = 2400) -> bytes:
    """Rasterize first page of a PDF to PNG bytes for vision."""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count < 1:
        raise RuntimeError("PDF has no pages")
    page = doc.load_page(0)
    rect = page.rect
    scale = max_side / max(rect.width, rect.height)
    scale = max(scale, 2.0)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png = pix.tobytes("png")
    doc.close()
    return png


def vision_extract(image_bytes: bytes, content_type: str) -> dict:
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY not set")
    mime = "image/png"
    if "jpeg" in content_type or "jpg" in content_type:
        mime = "image/jpeg"
    elif "webp" in content_type:
        mime = "image/webp"

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    body = {
        "model": XAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
        "temperature": 0.05,
    }
    result = http_json(
        "POST",
        XAI_URL,
        body=body,
        headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
        timeout=300,
    )
    text = result["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _build_flange(doc: DXFWriter, data: dict) -> Tuple[float, float, int]:
    """Center-origin flange / ring with optional tabs. Returns (w, h, hole_count)."""
    import math

    body_od = float(data.get("body_od") or 0)
    body_id = float(data.get("body_id") or 0)
    if body_od <= 0:
        # fall back to outer_size diameter
        osz = data.get("outer_size") or data.get("overall") or [0]
        body_od = float(osz[0]) if osz else 0
    if body_od <= 0:
        raise RuntimeError(f"flange missing body_od: {data}")

    body_r = body_od / 2.0
    hole_count = 0

    if body_id > 0:
        doc.circle(0, 0, body_id / 2.0)
        hole_count += 1

    # bolt circles
    for bc in data.get("bolt_circles") or []:
        bhc = float(bc["d"])
        n = int(bc["count"])
        hd = float(bc["hole_d"])
        if n <= 0 or hd <= 0 or bhc <= 0:
            continue
        start = float(bc.get("start_angle_deg") or 0)
        angles = bc.get("angles_deg")
        if angles:
            ang_list = [float(a) for a in angles]
        else:
            step = 360.0 / n
            ang_list = [start + i * step for i in range(n)]
        for ang in ang_list:
            a = math.radians(ang)
            doc.circle((bhc / 2) * math.cos(a), (bhc / 2) * math.sin(a), hd / 2)
            hole_count += 1

    # explicit holes
    for hole in data.get("holes") or []:
        d = float(hole["d"])
        if d <= 0:
            continue
        doc.circle(float(hole["x"]), float(hole["y"]), d / 2)
        hole_count += 1

    tabs = data.get("tabs") or {}
    outer = str(data.get("outer") or "").lower()
    if outer == "circle_with_tabs" or (tabs and int(tabs.get("count") or 0) > 0):
        count = int(tabs.get("count") or 4)
        angles = tabs.get("angles_deg") or [i * (360.0 / count) for i in range(count)]
        angles = [float(a) for a in angles]
        width = float(tabs.get("width") or body_od * 0.15)
        outer_r = float(tabs.get("outer_r") or (body_r + width * 0.5))
        fillet_r = float(tabs.get("fillet_r") or 0.25)
        # tab holes on optional BHC
        tab_bhc = float(tabs.get("hole_bhc") or 0)
        tab_hd = float(tabs.get("hole_d") or 0)
        if tab_bhc > 0 and tab_hd > 0:
            for a0 in angles:
                a = math.radians(a0)
                doc.circle((tab_bhc / 2) * math.cos(a), (tab_bhc / 2) * math.sin(a), tab_hd / 2)
                hole_count += 1

        half_ang = math.degrees(math.asin(min(0.999, (width / 2) / body_r))) if body_r > 0 else 10.0
        tip_center_r = max(outer_r - fillet_r, body_r * 0.5)

        for i, a0 in enumerate(angles):
            prev = angles[i - 1]
            a_start = prev + half_ang
            a_end = a0 - half_ang
            if a_end <= a_start:
                a_end += 360.0
            doc.arc(0, 0, body_r, a_start, a_end)

            tc = (tip_center_r * math.cos(math.radians(a0)), tip_center_r * math.sin(math.radians(a0)))
            sa, ea = a0 - 90.0, a0 + 90.0
            pL0 = (body_r * math.cos(math.radians(a0 - half_ang)), body_r * math.sin(math.radians(a0 - half_ang)))
            pR0 = (body_r * math.cos(math.radians(a0 + half_ang)), body_r * math.sin(math.radians(a0 + half_ang)))
            pL1 = (tc[0] + fillet_r * math.cos(math.radians(sa)), tc[1] + fillet_r * math.sin(math.radians(sa)))
            pR1 = (tc[0] + fillet_r * math.cos(math.radians(ea)), tc[1] + fillet_r * math.sin(math.radians(ea)))
            doc.line(pL0, pL1)
            doc.arc(tc[0], tc[1], fillet_r, sa, ea)
            doc.line(pR1, pR0)

        span = 2.0 * outer_r
        return span, span, hole_count

    # plain circle outer
    doc.circle(0, 0, body_r)
    return body_od, body_od, hole_count


def build_dxf_from_extract(data: dict) -> Tuple[bytes, dict]:
    import math
    units = str(data.get("units") or "in").lower()
    units = "mm" if units.startswith("mm") else "inches"
    doc = DXFWriter(units=units)
    origin = str(data.get("origin") or "").lower()
    outer = str(data.get("outer") or "rectangle").lower()
    part_type = str(data.get("part_type") or "").lower()

    hole_count = 0
    w = h = 0.0

    is_round = (
        origin == "center"
        or outer in ("circle", "circle_with_tabs")
        or part_type in ("flange", "ring")
        or data.get("body_od")
        or data.get("tabs")
        or data.get("bolt_circles")
    )

    if is_round and outer != "rectangle":
        w, h, hole_count = _build_flange(doc, data)
    else:
        overall = data.get("overall") or data.get("outer_size")
        if not overall or len(overall) < 1:
            raise RuntimeError(f"missing overall size: {data}")
        if outer == "circle":
            d = float(overall[0] if len(overall) == 1 else overall[0])
            doc.circle(d / 2.0, d / 2.0, d / 2.0)
            w = h = d
        else:
            w, h = float(overall[0]), float(overall[1] if len(overall) > 1 else overall[0])
            doc.rect(w, h)
        for hole in data.get("holes") or []:
            d = float(hole["d"])
            if d <= 0:
                continue
            doc.circle(float(hole["x"]), float(hole["y"]), d / 2.0)
            hole_count += 1
        # bolt circles on plate coords (rare)
        for bc in data.get("bolt_circles") or []:
            bhc = float(bc["d"])
            n = int(bc["count"])
            hd = float(bc["hole_d"])
            start = float(bc.get("start_angle_deg") or 0)
            cx = float(bc.get("cx") or w / 2)
            cy = float(bc.get("cy") or h / 2)
            for i in range(n):
                a = math.radians(start + i * (360.0 / n))
                doc.circle(cx + (bhc / 2) * math.cos(a), cy + (bhc / 2) * math.sin(a), hd / 2)
                hole_count += 1

    summary = {
        "overall": [w, h],
        "units": "in" if units.startswith("in") else "mm",
        "holes": hole_count,
        "part_type": part_type or outer,
        "thickness": data.get("thickness"),
        "material": data.get("material"),
        "assumptions": data.get("assumptions") or [],
        "notes": data.get("notes") or "",
    }
    return doc.to_bytes(), summary



def process_job(s3, job: dict) -> None:
    jid = job["id"]
    fname = (job.get("source") or {}).get("filename") or ""
    log(f"Processing {jid} ({fname})")

    job["status"] = "running"
    job["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    put_json(s3, f"{jid}/job.json", job)

    source_key = (job.get("source") or {}).get("key") or f"{jid}/source"
    raw = get_bytes(s3, source_key)
    ctype = (job.get("source") or {}).get("contentType") or "image/png"
    log(f"  source {len(raw)} bytes, {ctype}")

    if raw[:4] == b"%PDF" or "pdf" in (ctype or "").lower() or fname.lower().endswith(".pdf"):
        log("  rasterizing PDF page 1 → PNG")
        raw = pdf_to_png(raw)
        ctype = "image/png"
        log(f"  rasterized {len(raw)} bytes PNG")

    extract = vision_extract(raw, ctype)
    log(f"  extract: {json.dumps(extract)[:300]}")
    dxf_bytes, summary = build_dxf_from_extract(extract)

    dxf_key = f"{jid}/result.dxf"
    put_bytes(s3, dxf_key, dxf_bytes, "application/dxf")

    job["status"] = "done"
    job["error"] = None
    job["result"] = {"dxfKey": dxf_key, "summary": summary}
    job["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    put_json(s3, f"{jid}/job.json", job)
    log(f"  done {jid}")


def main() -> None:
    log("Serra recon service (R2 direct)")
    log(f"  bucket={R2_BUCKET}")
    log(f"  poll={POLL}s")
    log(f"  vision={'yes' if XAI_API_KEY else 'NO'}")

    if not XAI_API_KEY:
        log("ERROR: XAI_API_KEY required")
        sys.exit(1)

    try:
        s3 = r2_client()
        # smoke test
        s3.list_objects_v2(Bucket=R2_BUCKET, MaxKeys=1)
        log("  R2 connection: ok")
    except Exception as e:
        log(f"ERROR: R2 connection failed: {e}")
        sys.exit(1)

    while True:
        try:
            jobs = list_queued(s3)
            if not jobs:
                log("No queued jobs")
            for job in jobs:
                try:
                    process_job(s3, job)
                except Exception as e:
                    log(f"Job {job.get('id')} failed: {e}")
                    traceback.print_exc()
                    try:
                        job["status"] = "failed"
                        job["error"] = str(e)[:500]
                        put_json(s3, f"{job['id']}/job.json", job)
                    except Exception:
                        pass
        except Exception as e:
            log(f"Poll error: {e}")
            traceback.print_exc()
        time.sleep(POLL)


if __name__ == "__main__":
    main()
