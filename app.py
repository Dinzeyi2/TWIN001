"""
Codeastra Decision Fidelity Benchmark
Real-time streaming + twin data download
"""
import asyncio, base64, io, json, logging, math, os, time, traceback, uuid, warnings
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("codeastra.benchmark")

import numpy  as np
import pandas as pd
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import roc_auc_score
from sklearn.pipeline        import Pipeline

import requests as _requests
from fastapi           import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

app = FastAPI(title="Codeastra Decision Fidelity Benchmark", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

CODEASTRA_BASE = "https://app.codeastra.dev"
BATCH_SIZE     = 100
RANDOM_STATE   = 42

# Job queues — each running benchmark gets its own asyncio.Queue
_jobs: dict[str, asyncio.Queue] = {}

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Codeastra · Decision Fidelity Benchmark</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:#06060e; --card:#0c0c18; --border:#16163a;
  --blue:#4f7fff; --green:#00d68f; --gold:#f5c842; --red:#ff4d6d;
  --text:#eeeef8; --muted:#52527a;
  --mono:'DM Mono',monospace; --serif:'Instrument Serif',serif; --sans:'DM Sans',sans-serif;
}
body { background:var(--bg); color:var(--text); font-family:var(--sans);
       min-height:100vh; display:flex; flex-direction:column; align-items:center; }

header { width:100%; max-width:620px; padding:72px 24px 0; text-align:center; }
.eyebrow { font-family:var(--mono); font-size:11px; letter-spacing:.22em;
           color:var(--blue); text-transform:uppercase; margin-bottom:24px; }
h1 { font-family:var(--serif); font-size:clamp(36px,6vw,52px); font-weight:400;
     line-height:1.1; margin-bottom:20px; }
h1 em { font-style:italic; color:var(--green); }
.sub { font-size:15px; color:var(--muted); line-height:1.75; margin-bottom:60px;
       max-width:460px; margin-left:auto; margin-right:auto; }

.card { width:100%; max-width:580px; padding:0 24px; margin-bottom:20px; }
.card-inner { background:var(--card); border:1px solid var(--border);
              border-radius:16px; padding:36px; }

/* Drop zone */
.drop { border:1.5px dashed var(--border); border-radius:10px; padding:44px 24px;
        text-align:center; cursor:pointer; position:relative;
        transition:border-color .2s,background .2s; margin-bottom:20px; }
.drop:hover,.drop.over { border-color:var(--blue); background:rgba(79,127,255,.04); }
.drop input { position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%; }
.drop-icon { font-size:28px; margin-bottom:10px; }
.drop-main { font-size:14px; font-weight:500; margin-bottom:4px; }
.drop-hint { font-size:11px; font-family:var(--mono); color:var(--muted); }
.drop-chosen { margin-top:10px; font-size:12px; font-family:var(--mono); color:var(--green); }

.key-label { display:block; font-size:11px; font-family:var(--mono);
             color:var(--muted); letter-spacing:.06em; margin-bottom:8px; }
.key-input { width:100%; background:var(--bg); border:1px solid var(--border);
             border-radius:8px; padding:13px 14px; color:var(--text);
             font-family:var(--mono); font-size:13px; outline:none;
             transition:border-color .2s; margin-bottom:24px; }
.key-input:focus { border-color:var(--blue); }
.key-input::placeholder { color:var(--muted); }

.btn { width:100%; padding:15px; background:var(--blue); color:#fff;
       font-family:var(--sans); font-size:15px; font-weight:600; border:none;
       border-radius:8px; cursor:pointer; transition:opacity .18s; }
.btn:hover:not(:disabled) { opacity:.85; }
.btn:disabled { opacity:.35; cursor:not-allowed; }

/* Progress stream */
#streamBox { display:none; margin-top:28px; }
.stream-label { font-family:var(--mono); font-size:10px; letter-spacing:.18em;
                color:var(--blue); text-transform:uppercase; margin-bottom:14px;
                display:flex; align-items:center; gap:10px; }
.stream-label::after { content:''; flex:1; height:1px; background:var(--border); }
.progress-bar-wrap { background:var(--bg); border:1px solid var(--border);
                     border-radius:6px; height:6px; margin-bottom:16px; overflow:hidden; }
.progress-bar { height:100%; background:var(--green); width:0%;
                transition:width .4s ease; border-radius:6px; }
.batch-log { max-height:160px; overflow-y:auto; font-family:var(--mono);
             font-size:11px; color:var(--muted); line-height:1.8; }
.batch-log .done  { color:var(--green); }
.batch-log .active { color:var(--text); }

/* Error */
.err { background:rgba(255,77,109,.07); border:1px solid rgba(255,77,109,.25);
       border-radius:10px; padding:16px 20px; font-family:var(--mono);
       font-size:12px; color:var(--red); line-height:1.6;
       margin-top:16px; display:none; }

/* Results */
#resultsBox { display:none; }
.hero { text-align:center; padding:32px 0 36px;
        border-bottom:1px solid var(--border); margin-bottom:32px; }
.hero-num { font-family:var(--serif); font-size:88px; line-height:1;
            color:var(--green); margin-bottom:8px; letter-spacing:-.04em; }
.hero-label { font-family:var(--mono); font-size:11px; letter-spacing:.18em;
              color:var(--muted); text-transform:uppercase; }
.hero-file { margin-top:8px; font-size:12px; font-family:var(--mono); color:var(--muted); }

.auc-row { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:28px; }
.auc-box { background:var(--bg); border:1px solid var(--border);
           border-radius:10px; padding:20px; text-align:center; }
.auc-val { font-family:var(--mono); font-size:26px; font-weight:500; margin-bottom:6px; }
.auc-lbl { font-family:var(--mono); font-size:10px; color:var(--muted);
           letter-spacing:.1em; text-transform:uppercase; }

.verdict { font-size:14px; line-height:1.8; color:var(--muted); text-align:center;
           padding:24px; background:var(--bg); border:1px solid var(--border);
           border-radius:10px; margin-bottom:20px; }
.verdict strong { color:var(--gold); font-weight:500; }

/* Download button */
.btn-download { width:100%; padding:15px; background:transparent;
                border:1.5px solid var(--green); color:var(--green);
                font-family:var(--sans); font-size:15px; font-weight:600;
                border-radius:8px; cursor:pointer; transition:background .18s,color .18s;
                margin-bottom:12px; display:none; }
.btn-download:hover { background:var(--green); color:var(--bg); }
.btn-again { width:100%; padding:13px; background:transparent;
             border:1px solid var(--border); border-radius:8px; color:var(--muted);
             font-family:var(--sans); font-size:13px; cursor:pointer;
             transition:border-color .2s,color .2s; }
.btn-again:hover { border-color:var(--blue); color:var(--text); }
footer { padding:0 0 40px; font-family:var(--mono); font-size:10px;
         color:var(--muted); letter-spacing:.08em; }
</style>
</head>
<body>

<header>
  <div class="eyebrow">Codeastra</div>
  <h1>Does your twin data make<br><em>the same decisions?</em></h1>
  <p class="sub">Upload any dataset. We twin it via the Codeastra API,
  benchmark real vs twin models, and let you download your twinned data.</p>
</header>

<div class="card">
  <div class="card-inner">

    <div class="drop" id="drop">
      <input type="file" id="fileInput" accept=".csv,.xls,.xlsx">
      <div class="drop-icon">↑</div>
      <div class="drop-main">Drop your file here</div>
      <div class="drop-hint">CSV · XLS · XLSX</div>
      <div class="drop-chosen" id="dropChosen"></div>
    </div>

    <label class="key-label" for="apiKey">Codeastra API Key</label>
    <input class="key-input" type="password" id="apiKey" placeholder="sk-guard-...">

    <button class="btn" id="runBtn" onclick="run()" disabled>Run Benchmark</button>

    <!-- Live stream -->
    <div id="streamBox">
      <div class="stream-label">Twinning in progress</div>
      <div class="progress-bar-wrap"><div class="progress-bar" id="progressBar"></div></div>
      <div class="batch-log" id="batchLog"></div>
    </div>

    <div class="err" id="errBox"></div>
  </div>
</div>

<!-- Results -->
<div class="card" id="resultsBox">
  <div class="card-inner">
    <div class="hero">
      <div class="hero-num" id="rAgreement">—</div>
      <div class="hero-label">Prediction Agreement</div>
      <div class="hero-file" id="rFile"></div>
    </div>
    <div class="auc-row">
      <div class="auc-box">
        <div class="auc-val" style="color:#4f7fff" id="rRealAuc">—</div>
        <div class="auc-lbl">Real Data AUC</div>
      </div>
      <div class="auc-box">
        <div class="auc-val" style="color:#00d68f" id="rTwinAuc">—</div>
        <div class="auc-lbl">Twin Data AUC</div>
      </div>
    </div>
    <div class="verdict" id="rVerdict"></div>
    <button class="btn-download" id="btnDownload" onclick="downloadTwins()">
      ↓ Download Twin Data (CSV)
    </button>
    <button class="btn-again" onclick="reset()">Run another dataset</button>
  </div>
</div>

<footer>codeastra.dev</footer>

<script>
let currentFile = null;
let twinCsvB64  = null;
let twinFilename = null;

// Drop zone
const drop = document.getElementById('drop');
drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', ()  => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
document.getElementById('fileInput').addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});
document.getElementById('apiKey').addEventListener('input', checkReady);

function setFile(f) {
  currentFile = f;
  document.getElementById('dropChosen').textContent = '✓ ' + f.name;
  drop.style.borderColor = 'var(--green)';
  checkReady();
}
function checkReady() {
  const ok = currentFile && document.getElementById('apiKey').value.trim().length > 10;
  document.getElementById('runBtn').disabled = !ok;
}

async function run() {
  const apiKey = document.getElementById('apiKey').value.trim();
  twinCsvB64   = null;
  twinFilename = null;
  hide('errBox'); hide('resultsBox');
  document.getElementById('btnDownload').style.display = 'none';
  document.getElementById('runBtn').disabled = true;
  document.getElementById('batchLog').innerHTML = '';
  document.getElementById('progressBar').style.width = '0%';
  show('streamBox');

  // Step 1: Upload file, get job_id
  const fd = new FormData();
  fd.append('file',    currentFile);
  fd.append('api_key', apiKey);

  let jobId;
  try {
    const r = await fetch('/start', { method: 'POST', body: fd });
    if (!r.ok) {
      const e = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(e.detail || 'Upload failed');
    }
    const d = await r.json();
    jobId = d.job_id;
  } catch(e) {
    showErr(e.message);
    document.getElementById('runBtn').disabled = false;
    hide('streamBox');
    return;
  }

  // Step 2: Stream SSE events
  const es = new EventSource('/stream/' + jobId);

  es.addEventListener('batch', e => {
    const d = JSON.parse(e.data);
    const pct = Math.round(d.batch / d.total * 100);
    document.getElementById('progressBar').style.width = pct + '%';
    const log = document.getElementById('batchLog');
    // Mark previous active as done
    const prev = log.querySelector('.active');
    if (prev) prev.className = 'done';
    const line = document.createElement('div');
    line.className = 'active';
    line.textContent = `Batch ${d.batch}/${d.total} twinned ✓ (${d.elapsed}s)`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  });

  es.addEventListener('training', e => {
    const log = document.getElementById('batchLog');
    const prev = log.querySelector('.active');
    if (prev) prev.className = 'done';
    const line = document.createElement('div');
    line.className = 'active';
    line.textContent = 'Training models on real and twin data…';
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  });

  es.addEventListener('result', e => {
    es.close();
    document.getElementById('progressBar').style.width = '100%';
    const prev = document.getElementById('batchLog').querySelector('.active');
    if (prev) prev.className = 'done';
    hide('streamBox');
    document.getElementById('runBtn').disabled = false;

    const d = JSON.parse(e.data);
    if (d.error) { showErr(d.error); return; }

    showResults(d);
    if (d.twin_csv_b64) {
      twinCsvB64  = d.twin_csv_b64;
      twinFilename = d.twin_filename;
      document.getElementById('btnDownload').style.display = 'block';
    }
  });

  es.onerror = () => {
    es.close();
    showErr('Connection lost. Please try again.');
    document.getElementById('runBtn').disabled = false;
    hide('streamBox');
  };
}

function showResults(d) {
  const agree = d.agreement.toFixed(1);
  document.getElementById('rAgreement').textContent = agree + '%';
  document.getElementById('rFile').textContent      = d.filename;
  document.getElementById('rRealAuc').textContent   = d.real_auc.toFixed(4);
  document.getElementById('rTwinAuc').textContent   = d.twin_auc.toFixed(4);
  const q = d.agreement >= 95 ? 'virtually identical'
          : d.agreement >= 90 ? 'nearly identical' : 'similar';
  document.getElementById('rVerdict').innerHTML =
    `Models trained on <strong>Codeastra twin data</strong> reached ${q} decisions ` +
    `to models trained on original data — <strong>${agree}%</strong> agreement ` +
    `on <strong>${d.n_test.toLocaleString()} held-out real records</strong> ` +
    `never sent to the API.`;
  show('resultsBox');
  document.getElementById('resultsBox').scrollIntoView({ behavior: 'smooth' });
}

function downloadTwins() {
  if (!twinCsvB64) return;
  const bytes  = atob(twinCsvB64);
  const arr    = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  const blob   = new Blob([arr], { type: 'text/csv' });
  const url    = URL.createObjectURL(blob);
  const a      = document.createElement('a');
  a.href       = url;
  a.download   = twinFilename || 'codeastra_twins.csv';
  a.click();
  URL.revokeObjectURL(url);
}

function reset() {
  currentFile = null; twinCsvB64 = null; twinFilename = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('dropChosen').textContent = '';
  document.getElementById('drop').style.borderColor = '';
  document.getElementById('btnDownload').style.display = 'none';
  hide('resultsBox'); hide('errBox'); hide('streamBox');
  checkReady();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showErr(msg) {
  document.getElementById('errBox').textContent = '⚠  ' + msg;
  show('errBox');
}
function show(id) { document.getElementById(id).style.display = 'block'; }
function hide(id) { document.getElementById(id).style.display = 'none'; }
</script>
</body>
</html>
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_file(contents: bytes, filename: str) -> pd.DataFrame:
    base = filename.split("/")[-1].split("\\")[-1]
    if base.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(contents), engine="xlrd")
    elif base.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
    else:
        sample = contents[:4096].decode("utf-8", errors="ignore")
        sep    = ";" if sample.count(";") > sample.count(",") else ","
        df     = pd.read_csv(io.BytesIO(contents), sep=sep,
                             skipinitialspace=True, na_values="?")
        if all(isinstance(c, (int, float)) for c in df.columns):
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def prepare(df: pd.DataFrame):
    """
    Returns:
      X_raw       — DataFrame with original values (categoricals kept as strings)
                    → this is what gets sent to the Codeastra API
      X_encoded   — same but fully numeric → used for ML training
      y           — binary target
      target      — target column name
      cat_cols    — list of categorical column names
      encoders    — LabelEncoder per categorical column
    """
    target = df.columns[-1]
    df     = df.copy().dropna(subset=[target])

    # Encode target
    y_raw = df[target]
    if y_raw.dtype == object or str(y_raw.dtype) in ("category","string","bool"):
        classes = sorted(y_raw.astype(str).unique())
        y = (y_raw.astype(str) == classes[-1]).astype(int)
    else:
        try:
            y_num = pd.to_numeric(y_raw, errors="raise")
            y = (y_num > 0).astype(int) if y_num.max() > 1 else y_num.astype(int)
        except Exception:
            classes = sorted(y_raw.astype(str).unique())
            y = (y_raw.astype(str) == classes[-1]).astype(int)

    features = df.drop(columns=[target])

    # Identify categorical columns — keep raw for API, encode for ML
    cat_cols  = list(features.select_dtypes(include=["object","category","bool"]).columns)
    encoders  = {}
    X_raw     = features.copy()

    # Convert booleans to strings for API
    for col in cat_cols:
        X_raw[col] = X_raw[col].astype(str)

    # Build encoded version for sklearn
    X_encoded = features.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col]  = le
    X_encoded = X_encoded.select_dtypes(include="number").astype(float).fillna(0)

    return X_raw, X_encoded, y, target, cat_cols, encoders


def call_api_batch(records: list, api_key: str) -> list:
    clean = [
        {k.replace(" ","_").replace("-","_").lower(): str(v) for k,v in r.items()}
        for r in records
    ]
    resp = _requests.post(
        f"{CODEASTRA_BASE}/twin/think",
        headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        json={"data": clean, "task_type": "general", "agent_role": "assistant"},
        timeout=90,
    )
    if resp.status_code in (401, 403):
        raise ValueError("Invalid API key.")
    if resp.status_code != 200:
        raise ValueError(f"Codeastra API {resp.status_code}: {resp.text[:200]}")
    d = resp.json()
    t = d.get("twinned_data", [])
    if not t:
        raise ValueError("Codeastra API returned empty twinned_data.")
    return t if isinstance(t, list) else [t]


def records_to_df(
    twins:       list,
    original_df: pd.DataFrame,
    cat_cols:    list,
    encoders:    dict,
) -> tuple:
    """
    Convert API twin records into two DataFrames:
      twin_raw     — semantic values (Month=Aug, VisitorType=New_Visitor)
                     → used for the download CSV (human-readable)
      twin_encoded — numeric values for sklearn
                     → used for ML training

    Categorical fields twinned by the API as strings are preserved as-is
    in twin_raw, and label-encoded for twin_encoded.
    """
    cols    = list(original_df.columns)
    col_map = {c: c.replace(" ","_").replace("-","_").lower() for c in cols}

    # Collect all unique twin string values per categorical field
    # to build stable encoders
    new_string_vals: dict[str, set] = {}
    for rec in twins:
        for orig, clean in col_map.items():
            if orig not in cat_cols:
                continue
            val = rec.get(clean)
            if val is not None:
                new_string_vals.setdefault(orig, set()).add(str(val))

    # Extend original encoders with any new categories from twin
    extended_encoders = {}
    for col in cat_cols:
        orig_classes = list(encoders[col].classes_) if col in encoders else []
        new_classes  = list(new_string_vals.get(col, set()))
        all_classes  = sorted({str(v) for v in set(orig_classes) | set(new_classes)})
        le = LabelEncoder()
        le.fit(all_classes)
        extended_encoders[col] = le

    raw_rows = []
    enc_rows = []

    for i, rec in enumerate(twins):
        raw_row = {}
        enc_row = {}
        for orig, clean in col_map.items():
            val = rec.get(clean)
            if val is None:
                raise ValueError(f"API did not return field '{orig}' in twin {i}.")

            if orig in cat_cols:
                # Keep semantic string value for raw
                str_val      = str(val)
                raw_row[orig] = str_val
                # Encode to int for ML
                enc_row[orig] = float(
                    extended_encoders[orig].transform([str_val])[0]
                )
            else:
                # Numeric field — but API may return a formatted string
                # (e.g. NDC code 59148-0017-13, date 2024-03-15)
                # Try float first; if it's a string, hash-encode it.
                val_str = str(val).replace(",","").replace("$","").strip()
                try:
                    parsed = float(val_str)
                    if math.isnan(parsed):
                        parsed = 0.0
                    raw_row[orig] = parsed
                    enc_row[orig] = parsed
                except (ValueError, TypeError):
                    # API returned a semantic string for this field
                    # Use stable hash so the same string always → same int
                    import hashlib
                    code = int(hashlib.md5(val_str.encode()).hexdigest()[:8], 16) % 100000
                    raw_row[orig] = val_str   # human-readable in download
                    enc_row[orig] = float(code)  # stable int for sklearn

        raw_rows.append(raw_row)
        enc_rows.append(enc_row)

    twin_raw     = pd.DataFrame(raw_rows, columns=cols)
    twin_encoded = pd.DataFrame(enc_rows, columns=cols).astype(float)
    return twin_raw, twin_encoded


# ── Background job ────────────────────────────────────────────────────────────

async def run_benchmark_job(job_id: str, contents: bytes, filename: str, api_key: str):
    q = _jobs[job_id]

    def send(event: str, data: dict):
        q.put_nowait({"event": event, "data": data})

    try:
        # Read + prepare
        df = read_file(contents, filename)
        if len(df) < 50:
            send("result", {"error": "Need at least 50 rows."}); return

        X_raw, X_encoded, y, target, cat_cols, encoders = prepare(df)
        if X_encoded.shape[1] == 0:
            send("result", {"error": "No feature columns found."}); return

        # Split — keep raw and encoded in sync
        idx_tr, idx_te = train_test_split(
            range(len(X_raw)), test_size=0.20, stratify=y, random_state=RANDOM_STATE
        )
        X_tr_raw = X_raw.iloc[idx_tr].reset_index(drop=True)
        X_te_enc = X_encoded.iloc[idx_te].reset_index(drop=True)
        X_tr_enc = X_encoded.iloc[idx_tr].reset_index(drop=True)
        y_tr     = y.iloc[idx_tr].reset_index(drop=True)
        y_te     = y.iloc[idx_te].reset_index(drop=True)

        # Twin in batches — send RAW values so API twins semantically
        cols    = list(X_tr_raw.columns)
        col_map = {c: c.replace(" ","_").replace("-","_").lower() for c in cols}
        records = X_tr_raw.rename(columns=col_map).to_dict(orient="records")
        n       = len(records)
        n_b     = math.ceil(n / BATCH_SIZE)
        all_twins = []

        for i in range(0, n, BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            t0 = time.time()
            # Run sync call in thread pool so we don't block the event loop
            loop = asyncio.get_event_loop()
            batch_twins = await loop.run_in_executor(
                None, call_api_batch, records[i:i+BATCH_SIZE], api_key
            )
            if len(batch_twins) != len(records[i:i+BATCH_SIZE]):
                raise ValueError(
                    f"API returned {len(batch_twins)} twins for {len(records[i:i+BATCH_SIZE])} records."
                )
            all_twins.extend(batch_twins)
            send("batch", {
                "batch":   batch_num,
                "total":   n_b,
                "elapsed": round(time.time() - t0, 1),
            })

        # Convert twins — get semantic (for download) + encoded (for ML)
        twin_raw, X_tw_enc = records_to_df(all_twins, X_tr_raw, cat_cols, encoders)

        # Train models
        send("training", {})

        def make_model():
            return Pipeline([
                ("sc", StandardScaler()),
                ("rf", RandomForestClassifier(
                    200, max_depth=8, min_samples_leaf=5,
                    random_state=RANDOM_STATE, n_jobs=-1
                ))
            ])

        loop = asyncio.get_event_loop()

        def train_both():
            mr = make_model(); mr.fit(X_tr_enc, y_tr)
            mt = make_model(); mt.fit(X_tw_enc, y_tr)
            pred_r = mr.predict(X_te_enc)
            pred_t = mt.predict(X_te_enc)
            auc_r  = roc_auc_score(y_te, mr.predict_proba(X_te_enc)[:,1])
            auc_t  = roc_auc_score(y_te, mt.predict_proba(X_te_enc)[:,1])
            return pred_r, pred_t, auc_r, auc_t

        pred_r, pred_t, auc_r, auc_t = await loop.run_in_executor(None, train_both)

        agreement = float(np.mean(pred_r == pred_t)) * 100

        # Build twin CSV — semantic values (Month=Aug, not 1.6)
        twin_df_full = twin_raw.copy()
        twin_df_full[target] = y_tr.values
        csv_bytes  = twin_df_full.to_csv(index=False).encode()
        csv_b64    = base64.b64encode(csv_bytes).decode()
        stem       = filename.rsplit(".", 1)[0]
        twin_fname = f"{stem}_codeastra_twins.csv"

        send("result", {
            "filename":      filename,
            "real_auc":      round(auc_r, 4),
            "twin_auc":      round(auc_t, 4),
            "agreement":     round(agreement, 1),
            "n_train":       len(X_tr_raw),
            "n_test":        len(X_te_enc),
            "twin_csv_b64":  csv_b64,
            "twin_filename": twin_fname,
        })

    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"benchmark_job {job_id} FAILED:\n{tb}")
        send("result", {"error": str(e), "detail": tb[-500:]})
    finally:
        # Clean up job after a delay
        await asyncio.sleep(300)
        _jobs.pop(job_id, None)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return HTMLResponse(HTML)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/start")
async def start(
    file:    UploadFile = File(...),
    api_key: str        = Form(...),
):
    """Upload file + key → returns job_id immediately."""
    key = api_key.strip()
    if not key or len(key) < 20:
        raise HTTPException(401, "Invalid API key.")

    contents = await file.read()
    job_id   = str(uuid.uuid4())
    _jobs[job_id] = asyncio.Queue()

    asyncio.create_task(
        run_benchmark_job(job_id, contents, file.filename, key)
    )
    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream(job_id: str):
    """SSE stream for a running benchmark job."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found.")

    q = _jobs[job_id]

    async def event_generator():
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=120)
                yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"
                if msg["event"] == "result":
                    break
            except asyncio.TimeoutError:
                yield "event: ping\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
