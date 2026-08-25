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
  XAI_MODEL          grok 4.6
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
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-2-vision-1212").strip()
POLL = float(os.environ.get("POLL_SECONDS", "20"))
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


def pdf_to_png(pdf_bytes: bytes, max_side: int = 2000) -> bytes:
    """Rasterize first page of a PDF to PNG bytes for vision."""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count < 1:
        raise RuntimeError("PDF has no pages")
    page = doc.load_page(0)
    # Scale so longest side ~ max_side for readable dimensions
    rect = page.rect
    scale = max_side / max(rect.width, rect.height)
    scale = max(scale, 2.0)  # at least 2x for clear text/dims
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
        headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
    )
    text = result["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def build_dxf_from_extract(data: dict) -> Tuple[bytes, dict]:
    units = str(data.get("units") or "in").lower()
    units = "mm" if units.startswith("mm") else "inches"
    overall = data.get("overall") or data.get("outer_size")
    if not overall or len(overall) < 2:
        raise RuntimeError(f"missing overall size: {data}")
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
        d = float(hole["d"])
        if d <= 0:
            continue
        doc.circle(float(hole["x"]), float(hole["y"]), d / 2.0)
    summary = {
        "overall": [w, h],
        "units": "in" if units.startswith("in") else "mm",
        "holes": len(holes),
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
