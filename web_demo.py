"""
web_demo.py
===========
Real-time web cockpit and interactive dashboard for rag_guard.
Provides an interactive browser UI with live token streaming, dynamic
faithfulness gauges, entropy telemetries, entity grounding inspectors,
and real-time GPU early interception toggles.

Usage:
    python3 web_demo.py
    # Then open http://localhost:8000 in your browser
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rag_guard.vllm_hook import (
    AsyncRAGGuard,
    DEFAULT_TOP_K_LOGPROBS,
    MockAsyncLLMEngine,
    StreamLogprobEntry,
    StreamRequestOutput,
)

logger = logging.getLogger("rag_guard.web_demo")
app = FastAPI(title="rag_guard Real-Time Cockpit")

# ---------------------------------------------------------------------------
# Pre-loaded Academic Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "apple_grounded": {
        "title": "Apple Q1 2024 Earnings (Grounded & Faithful)",
        "context": "Apple Inc. reported quarterly revenue of $119.58 billion in Q1 2024, up 2 percent year-over-year. iPhone revenue was $69.7 billion, while Services revenue reached an all-time high of $23.12 billion.",
        "prompt": "Based on the context, what were Apple's financial results for Q1 2024?",
        "mode": "grounded",
        "custom_tokens": [
            ("Apple", -0.10, 5.0),
            (" Inc.", -0.12, 4.8),
            (" reported", -0.15, 4.5),
            (" $119.58B", -0.05, 6.0),
            (" in", -0.08, 5.2),
            (" revenue", -0.11, 4.9),
            (" for", -0.09, 5.1),
            (" Q1", -0.07, 5.4),
            (" 2024.", -0.04, 6.2),
            (" iPhone", -0.14, 4.6),
            (" revenue", -0.10, 5.0),
            (" was", -0.06, 5.5),
            (" $69.7B.", -0.05, 6.1),
        ],
    },
    "tesla_conflict": {
        "title": "Tesla Q3 Vehicle Deliveries (Evident Numerical Conflict)",
        "context": "In Q3 2023, Tesla delivered 435,059 vehicles worldwide and produced 430,488 vehicles, generating $23.35 billion in revenue against analysts' consensus of $24.1 billion.",
        "prompt": "How many vehicles did Tesla deliver in Q3 2023?",
        "mode": "conflict",
        "custom_tokens": [
            ("In", -0.10, 4.8),
            (" Q3", -0.12, 4.5),
            (" 2023,", -0.15, 4.2),
            (" Tesla", -0.18, 3.8),
            (" delivered", -0.25, 3.2),
            (" 850,000", -3.85, 0.15),  # Numerical contradiction: 850k vs 435k
            (" vehicles", -3.20, 0.20),
            (" and", -2.40, 0.35),
            (" achieved", -3.10, 0.18),
            (" $55.00B", -4.20, 0.12),  # Dollar contradiction: $55B vs $23B
            (" revenue.", -3.90, 0.14),
        ],
    },
    "openai_extrapolation": {
        "title": "OpenAI Enterprise Model (Baseless Extrapolation)",
        "context": "OpenAI announced GPT-4 on March 14, 2023. It passed a simulated bar exam with a score around the top 10% of test takers, representing a major leap over GPT-3.5.",
        "prompt": "What major milestones did OpenAI announce for GPT-4?",
        "mode": "extrapolation",
        "custom_tokens": [
            ("OpenAI", -0.10, 4.5),
            (" announced", -0.15, 4.2),
            (" GPT-4", -0.18, 3.9),
            (" on", -0.22, 3.5),
            (" March", -0.25, 3.1),
            (" 14,", -0.20, 3.4),
            (" 2023.", -0.15, 4.0),
            (" Following", -1.80, 0.35),
            (" release,", -2.20, 0.25),
            (" OpenAI", -2.90, 0.20),
            (" signed", -3.60, 0.15),
            (" a", -2.40, 0.22),
            (" $15B", -4.10, 0.10),   # External fabricated claim
            (" classified", -3.80, 0.12),
            (" defense", -3.50, 0.14),
            (" contract.", -3.90, 0.11),
        ],
    },
}


class StreamRequestPayload(BaseModel):
    preset_key: Optional[str] = "apple_grounded"
    context: str
    prompt: str
    token_delay_ms: float = 40.0
    early_stop_threshold: float = 0.40
    abort_on_hallucination: bool = True
    mode: str = "grounded"


class CustomMockEngine(MockAsyncLLMEngine):
    """Engine supporting preset-specific token scripts with calibrated logprobs."""

    def __init__(self, script: List[tuple], token_delay_ms: float = 40.0) -> None:
        super().__init__(token_delay_ms=token_delay_ms)
        self.script = script

    async def generate(
        self,
        prompt: str,
        sampling_params: Optional[Any] = None,
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamRequestOutput]:
        import uuid
        import numpy as np

        req_id = request_id or str(uuid.uuid4())
        rng = np.random.default_rng(seed=42)
        accumulated_text = ""
        logprobs_history = []

        for idx, (token_str, mean_lp, gap_scale) in enumerate(self.script):
            if req_id in self.aborted_requests:
                break

            if self.token_delay_s > 0:
                await asyncio.sleep(self.token_delay_s)

            accumulated_text += token_str

            top_k_dict = {}
            top_k_dict[token_str] = StreamLogprobEntry(logprob=float(mean_lp), rank=0, decoded_token=token_str)
            for k in range(1, DEFAULT_TOP_K_LOGPROBS):
                alt_lp = float(mean_lp - (gap_scale * k + rng.exponential(scale=0.1)))
                top_k_dict[f"<alt_{k}>"] = StreamLogprobEntry(logprob=alt_lp, rank=k, decoded_token=f"<alt_{k}>")

            logprobs_history.append(top_k_dict)
            is_last = (idx == len(self.script) - 1)

            yield StreamRequestOutput(
                request_id=req_id,
                prompt=prompt,
                outputs=[
                    type(
                        "StreamCompletionOutput",
                        (),
                        {
                            "text": accumulated_text,
                            "logprobs": list(logprobs_history),
                            "token_ids": list(range(idx + 1)),
                            "finish_reason": "stop" if is_last else None,
                        },
                    )()
                ],
                finished=is_last,
            )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/presets")
def get_presets():
    return {
        k: {
            "title": v["title"],
            "context": v["context"],
            "prompt": v["prompt"],
            "mode": v["mode"],
        }
        for k, v in PRESETS.items()
    }


@app.post("/api/stream")
async def stream_rag_guard(payload: StreamRequestPayload):
    # Select tokens from preset or fallback
    if payload.preset_key in PRESETS:
        script = PRESETS[payload.preset_key]["custom_tokens"]
    else:
        # Fallback script for custom prompt
        script = PRESETS["apple_grounded"]["custom_tokens"]

    engine = CustomMockEngine(script=script, token_delay_ms=payload.token_delay_ms)
    guard = AsyncRAGGuard(
        engine=engine,
        early_stop_threshold=payload.early_stop_threshold,
        abort_on_hallucination=payload.abort_on_hallucination,
    )

    async def event_generator():
        try:
            async for tok, step_res, final_res in guard.stream_generate(
                prompt=payload.prompt,
                context=payload.context,
            ):
                if tok:
                    payload_dict = {
                        "type": "token",
                        "token": tok,
                        "step": step_res.to_dict(),
                        "context_entities": sorted(list(guard.detector.weights_path.parent.name and set())),
                    }
                    yield f"data: {json.dumps(payload_dict)}\n\n"

                if final_res:
                    final_dict = {
                        "type": "finish",
                        "final": final_res.to_dict(),
                    }
                    yield f"data: {json.dumps(final_dict)}\n\n"
                    break

        except Exception as exc:
            logger.exception("Error in stream: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Cockpit Frontend UI (HTML5 + Tailwind CSS + Vanilla JS)
# ---------------------------------------------------------------------------
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>rag_guard — Real-Time RAG Hallucination Detection Cockpit</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @keyframes pulse-fast { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    .animate-stream { animation: pulse-fast 1s infinite; }
    .gauge-circle { transition: stroke-dashoffset 0.25s ease-out, stroke 0.25s ease; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased flex flex-col">

  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="h-9 w-9 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 font-bold text-lg">
        <i class="fa-solid fa-shield-halved"></i>
      </div>
      <div>
        <h1 class="font-bold text-lg tracking-tight flex items-center gap-2">
          rag_guard
          <span class="text-xs font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded">v0.1.0 Streaming</span>
        </h1>
        <p class="text-xs text-slate-400">Intrinsic Uncertainty Quantification & Deterministic Entity Verification</p>
      </div>
    </div>
    <div class="flex items-center space-x-6 text-xs text-slate-400 font-mono">
      <div class="flex items-center gap-2"><span class="h-2 w-2 rounded-full bg-emerald-400"></span> Budget: &lt;20 ms</div>
      <div class="flex items-center gap-2"><span class="h-2 w-2 rounded-full bg-emerald-400"></span> API Cost: $0.00</div>
      <div class="flex items-center gap-2"><span class="h-2 w-2 rounded-full bg-emerald-400"></span> NSUT CSDA 2026</div>
    </div>
  </header>

  <!-- Main Grid -->
  <main class="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 max-w-7xl mx-auto w-full">

    <!-- Column 1: Controls & Setup (4 cols) -->
    <section class="lg:col-span-4 flex flex-col gap-5">
      
      <!-- Preset Selector Card -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm">
        <label class="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Evaluation Scenario</label>
        <select id="presetSelect" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500">
          <option value="apple_grounded">🟢 Scenario 1: Apple Q1 2024 (Grounded / Faithful)</option>
          <option value="tesla_conflict">🔴 Scenario 2: Tesla Deliveries (Evident Conflict)</option>
          <option value="openai_extrapolation">🔴 Scenario 3: OpenAI Defense (Baseless Extrapolation)</option>
        </select>
      </div>

      <!-- Context Box -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 flex flex-col gap-3 shadow-sm flex-1">
        <div class="flex justify-between items-center">
          <label class="text-xs font-bold text-slate-300 uppercase tracking-wider">Retrieved Context (C)</label>
          <span id="entityCountBadge" class="text-[10px] font-mono bg-slate-800 text-slate-400 px-2 py-0.5 rounded">Pre-caching Ec...</span>
        </div>
        <textarea id="contextInput" rows="5" class="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-300 leading-relaxed focus:outline-none focus:border-emerald-500 resize-none"></textarea>

        <label class="text-xs font-bold text-slate-300 uppercase tracking-wider mt-1">Query Prompt (Q)</label>
        <input id="promptInput" type="text" class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500">

        <!-- Controls -->
        <div class="pt-3 border-t border-slate-800/80 flex flex-col gap-3">
          <div class="flex items-center justify-between">
            <span class="text-xs text-slate-300 flex items-center gap-1.5">
              <i class="fa-solid fa-bolt text-amber-400"></i> Auto-Abort on Alert
            </span>
            <input type="checkbox" id="autoAbortToggle" checked class="h-4 w-4 rounded bg-slate-950 border-slate-700 text-emerald-500 focus:ring-0">
          </div>
          <div class="flex items-center justify-between text-xs text-slate-400">
            <span>Stream Pace</span>
            <span id="paceLabel" class="font-mono text-slate-300">40 ms/tok</span>
          </div>
          <input type="range" id="paceRange" min="5" max="100" value="40" class="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500">

          <button id="streamBtn" class="mt-2 w-full bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold py-2.5 rounded-lg text-sm flex items-center justify-center gap-2 transition-all shadow-lg shadow-emerald-500/10">
            <i class="fa-solid fa-play"></i> Start Streaming Guardrail
          </button>
        </div>
      </div>

    </section>

    <!-- Column 2: Streaming Output & Interception Banner (5 cols) -->
    <section class="lg:col-span-5 flex flex-col gap-5">
      
      <!-- Streaming Answer Container -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 flex flex-col h-full shadow-sm">
        <div class="flex items-center justify-between mb-3 border-b border-slate-800 pb-3">
          <div class="flex items-center gap-2">
            <span id="statusIndicator" class="h-2.5 w-2.5 rounded-full bg-slate-600"></span>
            <span id="streamStatusText" class="text-xs font-mono font-medium text-slate-400 uppercase">Ready</span>
          </div>
          <div class="flex items-center gap-3 text-xs font-mono text-slate-400">
            <span>Tokens: <span id="tokenCounter" class="text-slate-200">0</span></span>
            <span>Overhead: <span id="latencyTimer" class="text-emerald-400">0.0 ms</span></span>
          </div>
        </div>

        <!-- Early Abort Alert Banner -->
        <div id="abortBanner" class="hidden mb-4 bg-red-500/10 border border-red-500/30 rounded-lg p-3.5 flex items-start gap-3">
          <div class="text-red-400 mt-0.5 text-base"><i class="fa-solid fa-triangle-exclamation"></i></div>
          <div>
            <div class="text-xs font-bold text-red-400 uppercase tracking-wide">GPU Early Abort Triggered!</div>
            <div id="abortReason" class="text-xs text-red-300/90 mt-0.5 leading-relaxed">Hallucination confidence exceeded safety threshold. Stream halted mid-flight to save GPU compute.</div>
          </div>
        </div>

        <!-- Live Streaming Output Text Box -->
        <div id="tokenStreamBox" class="flex-1 bg-slate-950 border border-slate-800/80 rounded-lg p-4 font-mono text-sm leading-relaxed overflow-y-auto min-h-[220px] text-slate-300">
          <span class="text-slate-600 italic">Click "Start Streaming Guardrail" to observe real-time token decoding...</span>
        </div>

        <!-- Entity Tag Badges Box -->
        <div class="mt-4 pt-3 border-t border-slate-800">
          <div class="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2 flex items-center justify-between">
            <span>Detected Entities in Answer (Ea)</span>
            <span id="groundingRatioDisplay" class="font-mono text-emerald-400 text-xs">R_entity: 1.00</span>
          </div>
          <div id="entityBadgesList" class="flex flex-wrap gap-1.5 min-h-[30px] items-center text-xs">
            <span class="text-slate-600 text-[11px] italic">No answer entities claimed yet...</span>
          </div>
        </div>
      </div>

    </section>

    <!-- Column 3: Telemetry & Live Faithfulness Gauges (3 cols) -->
    <section class="lg:col-span-3 flex flex-col gap-5">
      
      <!-- Faithfulness Gauge Card -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 flex flex-col items-center shadow-sm">
        <span class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Composite Faithfulness</span>
        
        <!-- SVG Gauge -->
        <div class="relative w-36 h-36 flex items-center justify-center my-2">
          <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" class="stroke-slate-800 fill-none" stroke-width="8" />
            <circle id="gaugeProgress" cx="50" cy="50" r="40" class="gauge-circle stroke-emerald-500 fill-none" stroke-width="8" stroke-dasharray="251.2" stroke-dashoffset="251.2" stroke-linecap="round" />
          </svg>
          <div class="absolute flex flex-col items-center">
            <span id="faithScoreLabel" class="text-2xl font-black font-mono tracking-tight text-slate-100">0%</span>
            <span id="verdictTag" class="text-[10px] font-bold uppercase tracking-wider text-slate-400 mt-0.5">STANDBY</span>
          </div>
        </div>

        <div class="grid grid-cols-2 w-full gap-2 mt-2 pt-3 border-t border-slate-800 text-center font-mono">
          <div class="bg-slate-950/60 p-2 rounded border border-slate-800/80">
            <div class="text-[10px] text-slate-400 uppercase">Hallucination</div>
            <div id="hallucinationScoreLabel" class="text-xs font-bold text-slate-300">0%</div>
          </div>
          <div class="bg-slate-950/60 p-2 rounded border border-slate-800/80">
            <div class="text-[10px] text-slate-400 uppercase">Threshold (tau)</div>
            <div id="thresholdLabel" class="text-xs font-bold text-slate-400">0.50</div>
          </div>
        </div>
      </div>

      <!-- Feature Breakdown Telemetry -->
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm flex-1 flex flex-col justify-between">
        <span class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">5D Signal Telemetry</span>
        
        <div class="space-y-3 font-mono text-xs">
          <div>
            <div class="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Mean Logprob (&ell;&#772;)</span>
              <span id="val_l_bar" class="text-slate-200">0.00</span>
            </div>
            <div class="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
              <div id="bar_l_bar" class="bg-emerald-500 h-full w-0 transition-all duration-200"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Hesitation (&ell;<sub>min</sub>)</span>
              <span id="val_l_min" class="text-slate-200">0.00</span>
            </div>
            <div class="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
              <div id="bar_l_min" class="bg-emerald-500 h-full w-0 transition-all duration-200"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Entropy (H&#772; bits)</span>
              <span id="val_h_bar" class="text-slate-200">0.00</span>
            </div>
            <div class="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
              <div id="bar_h_bar" class="bg-amber-400 h-full w-0 transition-all duration-200"></div>
            </div>
          </div>

          <div>
            <div class="flex justify-between text-[11px] text-slate-400 mb-1">
              <span>Entity Ratio (R<sub>ent</sub>)</span>
              <span id="val_r_ent" class="text-slate-200">1.00</span>
            </div>
            <div class="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
              <div id="bar_r_ent" class="bg-emerald-500 h-full w-full transition-all duration-200"></div>
            </div>
          </div>
        </div>

        <!-- Latency Budget Bar -->
        <div class="mt-4 pt-3 border-t border-slate-800">
          <div class="flex justify-between text-[11px] font-mono mb-1">
            <span class="text-slate-400">Latency Overhead (&Delta;t)</span>
            <span id="budgetPct" class="text-emerald-400 font-bold">&lt; 1.0 ms</span>
          </div>
          <div class="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden border border-slate-800">
            <div id="budgetBar" class="bg-emerald-500 h-full w-[5%] transition-all"></div>
          </div>
        </div>

      </div>

    </section>

  </main>

  <!-- Frontend Logic -->
  <script>
    let presetsData = {};
    const CIRCUMFERENCE = 251.2;

    async function loadPresets() {
      const res = await fetch('/api/presets');
      presetsData = await res.json();
      applyPreset('apple_grounded');
    }

    function applyPreset(key) {
      const p = presetsData[key];
      if (!p) return;
      document.getElementById('contextInput').value = p.context;
      document.getElementById('promptInput').value = p.prompt;
    }

    document.getElementById('presetSelect').addEventListener('change', (e) => {
      applyPreset(e.target.value);
    });

    document.getElementById('paceRange').addEventListener('input', (e) => {
      document.getElementById('paceLabel').textContent = `${e.target.value} ms/tok`;
    });

    function updateGauge(score, isHallucination) {
      const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
      const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;
      const progress = document.getElementById('gaugeProgress');
      const scoreLabel = document.getElementById('faithScoreLabel');
      const verdict = document.getElementById('verdictTag');

      progress.style.strokeDashoffset = offset;
      scoreLabel.textContent = `${pct}%`;

      if (isHallucination) {
        progress.classList.remove('stroke-emerald-500');
        progress.classList.add('stroke-red-500');
        scoreLabel.classList.add('text-red-400');
        verdict.textContent = "HALLUCINATION";
        verdict.className = "text-[10px] font-bold uppercase tracking-wider text-red-400 mt-0.5";
      } else {
        progress.classList.remove('stroke-red-500');
        progress.classList.add('stroke-emerald-500');
        scoreLabel.classList.remove('text-red-400');
        verdict.textContent = "GROUNDED";
        verdict.className = "text-[10px] font-bold uppercase tracking-wider text-emerald-400 mt-0.5";
      }

      document.getElementById('hallucinationScoreLabel').textContent = `${100 - pct}%`;
    }

    function updateTelemetry(step) {
      document.getElementById('val_l_bar').textContent = step.l_bar.toFixed(2);
      document.getElementById('val_l_min').textContent = step.l_min.toFixed(2);
      document.getElementById('val_h_bar').textContent = step.H_bar.toFixed(2);
      document.getElementById('val_r_ent').textContent = step.R_entity.toFixed(2);

      // Bars
      const lBarPct = Math.max(0, Math.min(100, (1 - Math.abs(step.l_bar) / 4) * 100));
      document.getElementById('bar_l_bar').style.w = `${lBarPct}%`;

      const hBarPct = Math.max(0, Math.min(100, (step.H_bar / 2.5) * 100));
      document.getElementById('bar_h_bar').style.width = `${hBarPct}%`;
      document.getElementById('bar_r_ent').style.width = `${step.R_entity * 100}%`;
    }

    // Streaming client
    document.getElementById('streamBtn').addEventListener('click', async () => {
      const btn = document.getElementById('streamBtn');
      const box = document.getElementById('tokenStreamBox');
      const abortBanner = document.getElementById('abortBanner');
      const statusIndicator = document.getElementById('statusIndicator');
      const statusText = document.getElementById('streamStatusText');
      const tokenCounter = document.getElementById('tokenCounter');
      const badgesList = document.getElementById('entityBadgesList');

      // Reset
      btn.disabled = true;
      box.innerHTML = '';
      badgesList.innerHTML = '';
      abortBanner.classList.add('hidden');
      statusIndicator.className = "h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse";
      statusText.textContent = "STREAMING...";
      statusText.className = "text-xs font-mono font-medium text-emerald-400 uppercase";

      const payload = {
        preset_key: document.getElementById('presetSelect').value,
        context: document.getElementById('contextInput').value,
        prompt: document.getElementById('promptInput').value,
        token_delay_ms: parseFloat(document.getElementById('paceRange').value),
        early_stop_threshold: 0.35,
        abort_on_hallucination: document.getElementById('autoAbortToggle').checked
      };

      const startTime = performance.now();
      let tokCount = 0;

      try {
        const response = await fetch('/api/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n\\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.substring(6));

              if (data.type === 'token') {
                tokCount++;
                tokenCounter.textContent = tokCount;
                const elapsed = (performance.now() - startTime).toFixed(1);
                document.getElementById('latencyTimer').textContent = `${elapsed} ms`;

                // Render token with color
                const span = document.createElement('span');
                span.textContent = data.token;
                if (data.step.is_hallucination) {
                  span.className = "text-red-400 font-bold bg-red-500/10 px-0.5 rounded";
                } else {
                  span.className = "text-slate-200";
                }
                box.appendChild(span);
                box.scrollTop = box.scrollHeight;

                updateGauge(data.step.faithfulness_score, data.step.is_hallucination);
                updateTelemetry(data.step);

                if (data.step.early_stop_recommended && payload.abort_on_hallucination) {
                  abortBanner.classList.remove('hidden');
                  statusIndicator.className = "h-2.5 w-2.5 rounded-full bg-red-500";
                  statusText.textContent = "ABORTED BY GUARDRAIL";
                  statusText.className = "text-xs font-mono font-medium text-red-400 uppercase";
                }
              } else if (data.type === 'finish') {
                statusIndicator.className = data.final.is_hallucination 
                  ? "h-2.5 w-2.5 rounded-full bg-red-500" 
                  : "h-2.5 w-2.5 rounded-full bg-emerald-400";
                statusText.textContent = data.final.is_hallucination ? "FLAGGED HALLUCINATION" : "VERIFIED FAITHFUL";
                
                // Populate badges
                badgesList.innerHTML = '';
                data.final.answer_entities.forEach(ent => {
                  const b = document.createElement('span');
                  const isMatch = data.final.context_entities.includes(ent);
                  b.className = isMatch 
                    ? "bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 px-2 py-0.5 rounded font-mono text-[11px]"
                    : "bg-red-500/15 border border-red-500/30 text-red-400 px-2 py-0.5 rounded font-mono text-[11px]";
                  b.textContent = ent + (isMatch ? " ✓" : " ✗");
                  badgesList.appendChild(b);
                });
              }
            }
          }
        }
      } catch (err) {
        console.error(err);
      } finally {
        btn.disabled = false;
      }
    });

    loadPresets();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return HTMLResponse(content=HTML_DASHBOARD)


def run_web_cockpit(port: int = 8000, host: str = "127.0.0.1"):
    print("\n" + "=" * 65)
    print("  rag_guard Real-Time Web Cockpit Initializing...")
    print(f"  Access Cockpit UI in browser: http://{host}:{port}")
    print("=" * 65 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_web_cockpit()
