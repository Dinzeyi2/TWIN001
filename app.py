"""
Codeastra Decision Fidelity Benchmark — Railway Web App
"""
import os, io, json, time, math, warnings
from datetime import datetime

import numpy  as np
import pandas as pd
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import roc_auc_score
from sklearn.pipeline        import Pipeline

import requests as _requests
from fastapi           import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

CODEASTRA_BASE = "https://app.codeastra.dev"
BATCH_SIZE     = 100
RANDOM_STATE   = 42

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
  --bg:       #06060e;
  --card:     #0c0c18;
  --border:   #16163a;
  --blue:     #4f7fff;
  --green:    #00d68f;
  --gold:     #f5c842;
  --red:      #ff4d6d;
  --text:     #eeeef8;
  --muted:    #52527a;
  --mono:     'DM Mono', monospace;
  --serif:    'Instrument Serif', serif;
  --sans:     'DM Sans', sans-serif;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
}

/* ── Header ── */
header {
  width: 100%;
  max-width: 620px;
  padding: 72px 24px 0;
  text-align: center;
}
.eyebrow {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.22em;
  color: var(--blue);
  text-transform: uppercase;
  margin-bottom: 24px;
}
h1 {
  font-family: var(--serif);
  font-size: clamp(36px, 6vw, 56px);
  font-weight: 400;
  line-height: 1.1;
  margin-bottom: 20px;
  letter-spacing: -0.01em;
}
h1 em {
  font-style: italic;
  color: var(--green);
}
.sub {
  font-size: 15px;
  color: var(--muted);
  line-height: 1.75;
  margin-bottom: 60px;
  max-width: 460px;
  margin-left: auto;
  margin-right: auto;
}

/* ── Upload card ── */
.card {
  width: 100%;
  max-width: 560px;
  padding: 0 24px;
  margin-bottom: 20px;
}
.card-inner {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 36px;
}

/* ── Drop zone ── */
.drop {
  border: 1.5px dashed var(--border);
  border-radius: 10px;
  padding: 44px 24px;
  text-align: center;
  cursor: pointer;
  position: relative;
  transition: border-color .2s, background .2s;
  margin-bottom: 20px;
}
.drop:hover, .drop.over {
  border-color: var(--blue);
  background: rgba(79,127,255,.04);
}
.drop input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
  width: 100%;
  height: 100%;
}
.drop-icon { font-size: 28px; margin-bottom: 10px; }
.drop-main { font-size: 14px; font-weight: 500; margin-bottom: 4px; }
.drop-hint { font-size: 11px; font-family: var(--mono); color: var(--muted); }
.drop-chosen {
  margin-top: 10px;
  font-size: 12px;
  font-family: var(--mono);
  color: var(--green);
}

/* ── Key input ── */
.key-wrap { position: relative; margin-bottom: 24px; }
.key-label {
  display: block;
  font-size: 11px;
  font-family: var(--mono);
  color: var(--muted);
  letter-spacing: .06em;
  margin-bottom: 8px;
}
.key-input {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 13px 14px;
  color: var(--text);
  font-family: var(--mono);
  font-size: 13px;
  outline: none;
  transition: border-color .2s;
}
.key-input:focus { border-color: var(--blue); }
.key-input::placeholder { color: var(--muted); }

/* ── Run button ── */
.btn {
  width: 100%;
  padding: 15px;
  background: var(--blue);
  color: #fff;
  font-family: var(--sans);
  font-size: 15px;
  font-weight: 600;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: opacity .18s;
  letter-spacing: .01em;
}
.btn:hover:not(:disabled) { opacity: .85; }
.btn:disabled { opacity: .35; cursor: not-allowed; }

/* ── State bar ── */
.state-bar {
  width: 100%;
  max-width: 560px;
  padding: 0 24px;
  margin-bottom: 20px;
  display: none;
}
.state-inner {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 24px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 12px;
}
.spinner {
  width: 18px; height: 18px;
  border: 2px solid var(--border);
  border-top-color: var(--blue);
  border-radius: 50%;
  animation: spin .7s linear infinite;
  flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Error ── */
.err {
  width: 100%;
  max-width: 560px;
  padding: 0 24px;
  margin-bottom: 20px;
  display: none;
}
.err-inner {
  background: rgba(255,77,109,.07);
  border: 1px solid rgba(255,77,109,.25);
  border-radius: 10px;
  padding: 16px 20px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--red);
  line-height: 1.6;
}

/* ── Results ── */
.results {
  width: 100%;
  max-width: 560px;
  padding: 0 24px;
  margin-bottom: 60px;
  display: none;
}
.results-inner {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 36px;
}

/* Hero agreement */
.hero {
  text-align: center;
  padding: 32px 0 36px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 32px;
}
.hero-num {
  font-family: var(--serif);
  font-size: 88px;
  line-height: 1;
  color: var(--green);
  margin-bottom: 8px;
  letter-spacing: -0.04em;
}
.hero-label {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: .18em;
  color: var(--muted);
  text-transform: uppercase;
}
.hero-file {
  margin-top: 8px;
  font-size: 12px;
  font-family: var(--mono);
  color: var(--muted);
}

/* AUC row */
.auc-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 28px;
}
.auc-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
  text-align: center;
}
.auc-val {
  font-family: var(--mono);
  font-size: 26px;
  font-weight: 500;
  margin-bottom: 6px;
}
.auc-lbl {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  letter-spacing: .1em;
  text-transform: uppercase;
}

/* Verdict */
.verdict {
  font-size: 14px;
  line-height: 1.8;
  color: var(--muted);
  text-align: center;
  padding: 24px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  margin-bottom: 20px;
}
.verdict strong { color: var(--gold); font-weight: 500; }

/* Run again */
.run-again {
  display: block;
  width: 100%;
  padding: 13px;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--muted);
  font-family: var(--sans);
  font-size: 13px;
  cursor: pointer;
  text-align: center;
  transition: border-color .2s, color .2s;
}
.run-again:hover { border-color: var(--blue); color: var(--text); }

footer {
  padding: 0 0 40px;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  letter-spacing: .08em;
}
</style>
</head>
<body>

<header>
  <div class="eyebrow">Codeastra</div>
  <h1>Does your<br>twin data make<br><em>the same decisions?</em></h1>
  <p class="sub">Upload any dataset. We twin it, train two models,
  and measure how often they agree — without ever exposing the real data.</p>
</header>

<!-- Upload card -->
<div class="card">
  <div class="card-inner">

    <div class="drop" id="drop">
      <input type="file" id="fileInput" accept=".csv,.xls,.xlsx">
      <div class="drop-icon">↑</div>
      <div class="drop-main">Drop your file here</div>
      <div class="drop-hint">CSV · XLS · XLSX</div>
      <div class="drop-chosen" id="dropChosen"></div>
    </div>

    <div class="key-wrap">
      <label class="key-label" for="apiKey">Codeastra API Key</label>
      <input class="key-input" type="password" id="apiKey"
             placeholder="sk-guard-...">
    </div>

    <button class="btn" id="runBtn" onclick="run()" disabled>
      Run Benchmark
    </button>

  </div>
</div>

<!-- State -->
<div class="state-bar" id="stateBar">
  <div class="state-inner">
    <div class="spinner"></div>
    <span id="stateMsg">Twinning your data via Codeastra API…</span>
  </div>
</div>

<!-- Error -->
<div class="err" id="errBox">
  <div class="err-inner" id="errMsg"></div>
</div>

<!-- Results -->
<div class="results" id="resultsBox">
  <div class="results-inner">

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

    <button class="run-again" onclick="reset()">Run another dataset</button>

  </div>
</div>

<footer>codeastra.dev</footer>

<script>
let currentFile = null;

const drop = document.getElementById('drop');
drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
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
  document.getElementById('fileInput').files;
  document.getElementById('dropChosen').textContent = '✓ ' + f.name;
  drop.style.borderColor = 'var(--green)';
  checkReady();
}

function checkReady() {
  const ok = currentFile && document.getElementById('apiKey').value.trim().length > 10;
  document.getElementById('runBtn').disabled = !ok;
}

function setState(msg) {
  document.getElementById('stateMsg').textContent = msg;
  document.getElementById('stateBar').style.display = 'block';
}

async function run() {
  const apiKey = document.getElementById('apiKey').value.trim();
  hide('errBox'); hide('resultsBox');
  document.getElementById('runBtn').disabled = true;

  setState('Twinning your data via Codeastra API…');

  const fd = new FormData();
  fd.append('file',    currentFile);
  fd.append('api_key', apiKey);

  try {
    setState('Twinning your data via Codeastra API…');
    const r = await fetch('/benchmark', { method: 'POST', body: fd });
    if (!r.ok) {
      const e = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(e.detail || 'Benchmark failed');
    }
    setState('Training models…');
    const data = await r.json();
    show(data);
  } catch (e) {
    document.getElementById('errMsg').textContent = '⚠  ' + (e.message || String(e));
    show_el('errBox');
  } finally {
    hide('stateBar');
    document.getElementById('runBtn').disabled = false;
  }
}

function show(data) {
  const agree = data.agreement.toFixed(1);
  document.getElementById('rAgreement').textContent = agree + '%';
  document.getElementById('rFile').textContent      = data.filename;
  document.getElementById('rRealAuc').textContent   = data.real_auc.toFixed(4);
  document.getElementById('rTwinAuc').textContent   = data.twin_auc.toFixed(4);

  const auc_diff = data.twin_auc - data.real_auc;
  const q = data.agreement >= 95 ? 'virtually identical'
          : data.agreement >= 90 ? 'nearly identical'
          : 'similar';
  document.getElementById('rVerdict').innerHTML =
    `Models trained on <strong>Codeastra twin data</strong> reached ${q} decisions
     to models trained on the original data —
     <strong>${agree}%</strong> agreement on
     <strong>${data.n_test.toLocaleString()} held-out real records</strong>
     that were never sent to the API.`;

  show_el('resultsBox');
  document.getElementById('resultsBox').scrollIntoView({ behavior: 'smooth' });
}

function reset() {
  currentFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('dropChosen').textContent = '';
  document.getElementById('drop').style.borderColor = '';
  hide('resultsBox'); hide('errBox');
  checkReady();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function hide(id)    { document.getElementById(id).style.display = 'none'; }
function show_el(id) { document.getElementById(id).style.display = 'block'; }
</script>
</body>
</html>"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_file(contents: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith((".xls",)):
        return pd.read_excel(io.BytesIO(contents), engine="xlrd")
    if filename.endswith((".xlsx",)):
        return pd.read_excel(io.BytesIO(contents), engine="openpyxl")
    # CSV — detect separator
    sample = contents[:4096].decode("utf-8", errors="ignore")
    sep    = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(io.BytesIO(contents), sep=sep)


def prepare(df: pd.DataFrame):
    """Auto-detect target (last column), encode, return X, y."""
    target = df.columns[-1]
    df     = df.copy().dropna(subset=[target])

    for col in df.select_dtypes(include=["object","category","bool"]).columns:
        if col != target:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    y_raw = df[target]
    if y_raw.dtype == object or str(y_raw.dtype) in ("category","string"):
        classes  = sorted(y_raw.astype(str).unique())
        y = (y_raw.astype(str) == classes[-1]).astype(int)
    elif y_raw.nunique() <= 10:
        y = y_raw.astype(int)
        # binarize if multi-class
        if y.max() > 1:
            y = (y > 0).astype(int)
    else:
        y = (y_raw > y_raw.median()).astype(int)

    X = df.drop(columns=[target]).select_dtypes(include="number").astype(float)
    X = X.fillna(X.median())
    return X, y, target


def call_api(records: list, api_key: str) -> list:
    """Call Codeastra /twin/think. Raises immediately on any error — no retries, no fallbacks."""
    clean = [
        {k.replace(" ","_").replace("-","_").lower(): str(v)
         for k, v in r.items()}
        for r in records
    ]
    resp = _requests.post(
        f"{CODEASTRA_BASE}/twin/think",
        headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        json={"data": clean, "task_type": "general", "agent_role": "assistant"},
        timeout=90,
    )
    if resp.status_code == 401 or resp.status_code == 403:
        raise HTTPException(401, "Invalid API key. Check your Codeastra API key and try again.")
    if resp.status_code != 200:
        raise HTTPException(502, f"Codeastra API returned {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    twinned = data.get("twinned_data", [])
    if not twinned:
        raise HTTPException(502, "Codeastra API returned empty twinned_data.")
    return twinned if isinstance(twinned, list) else [twinned]


def twin_df(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Twin every row via the Codeastra API. No fallbacks.
    If the API fails or returns an unparseable value, we raise — never silently use real data.
    """
    cols    = list(df.columns)
    col_map = {c: c.replace(" ","_").replace("-","_").lower() for c in cols}
    records = df.rename(columns=col_map).to_dict(orient="records")
    twins   = []

    for i in range(0, len(records), BATCH_SIZE):
        batch_twins = call_api(records[i: i + BATCH_SIZE], api_key)
        if len(batch_twins) != len(records[i: i + BATCH_SIZE]):
            raise HTTPException(502,
                f"Codeastra API returned {len(batch_twins)} twins "
                f"for {len(records[i:i+BATCH_SIZE])} records.")
        twins.extend(batch_twins)

    rows = []
    for i, rec in enumerate(twins):
        row = {}
        for orig, clean in col_map.items():
            val = rec.get(clean)
            if val is None:
                raise HTTPException(502,
                    f"Codeastra API did not return field '{orig}' in twin record {i}.")
            try:
                parsed = float(str(val).replace(",","").replace("$","").strip())
            except (ValueError, TypeError):
                raise HTTPException(502,
                    f"Codeastra API returned non-numeric value '{val}' "
                    f"for numeric field '{orig}'.")
            if math.isnan(parsed):
                raise HTTPException(502,
                    f"Codeastra API returned NaN for field '{orig}'.")
            row[orig] = parsed
        rows.append(row)

    return pd.DataFrame(rows, columns=cols)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/benchmark")
async def benchmark(
    file:    UploadFile = File(...),
    api_key: str        = Form(...),
):
    contents = await file.read()

    try:
        df = read_file(contents, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Cannot read file: {e}")

    if len(df) < 50:
        raise HTTPException(400, "Need at least 50 rows.")

    try:
        X, y, target = prepare(df)
    except Exception as e:
        raise HTTPException(400, f"Data preparation failed: {e}")

    if X.shape[1] == 0:
        raise HTTPException(400, "No numeric feature columns found.")

    # Split — test set NEVER touches the API
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Twin training set
    try:
        X_tw = twin_df(X_tr, api_key.strip())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Codeastra API error: {e}")

    # Train
    def model():
        return Pipeline([
            ("sc", StandardScaler()),
            ("rf", RandomForestClassifier(
                200, max_depth=8, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ])

    mr = model(); mr.fit(X_tr, y_tr)
    mt = model(); mt.fit(X_tw, y_tr)

    pred_r = mr.predict(X_te)
    pred_t = mt.predict(X_te)
    auc_r  = roc_auc_score(y_te, mr.predict_proba(X_te)[:,1])
    auc_t  = roc_auc_score(y_te, mt.predict_proba(X_te)[:,1])

    return {
        "filename":  file.filename,
        "target":    target,
        "real_auc":  round(auc_r, 4),
        "twin_auc":  round(auc_t, 4),
        "agreement": round(float(np.mean(pred_r == pred_t)) * 100, 1),
        "n_train":   len(X_tr),
        "n_test":    len(X_te),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("benchmark_app:app", host="0.0.0.0", port=port)
