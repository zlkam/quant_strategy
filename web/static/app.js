/**
 * Quant Strategy Dashboard — Frontend Logic
 *
 * Fetches data from the Python FastAPI backend and renders:
 *  - Combined candlestick + indicator Plotly chart
 *  - Signal cards for all 9 tickers
 *  - Position tracker, metrics, and Quick View buttons
 */

// ---- State ----
const STATE = {
  ticker: "QQQ",
  displayBars: 60,
  period: "3mo",
  refreshSec: 300,
  chartData: null,   // full OHLCV + indicator data for current ticker
  signals: null,     // latest signals for all tickers
  thresholds: null,  // hysteresis entry/exit levels
  positions: null,
  timerId: null,
};

const MAP = { 1: "LONG", 0: "FLAT", "-1": "SHORT" };
const COL = { LONG: "#00CC66", SHORT: "#FF4444", FLAT: "#888", ERROR: "#FF4444" };
const API = "";  // same-origin

// ---- Helpers ----

function $ (sel) { return document.querySelector(sel); }
function $$ (sel) { return document.querySelectorAll(sel); }

async function fetchJSON (url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ---- Initialisation ----

async function init () {
  await loadThresholds();
  await loadSignals();
  await loadPositions();
  await loadChart();
  renderQuickView();
  renderSignalCards();
  renderPositions();
  renderMetrics();
  scheduleRefresh();
}

// ---- API calls ----

async function loadThresholds () {
  STATE.thresholds = await fetchJSON(`${API}/api/thresholds`);
}

async function loadSignals () {
  STATE.signals = await fetchJSON(`${API}/api/signals?period=${STATE.period}`);
}

async function loadPositions () {
  try {
    STATE.positions = await fetchJSON(`${API}/api/positions`);
  } catch (_) {
    STATE.positions = {};
  }
}

async function loadChart () {
  const t = STATE.ticker;
  const p = STATE.period;
  const chartEl = $("#chart-price");
  const indEl = $("#chart-indicator");
  chartEl.innerHTML = '<div class="loading">Loading...</div>';
  indEl.innerHTML = '';

  try {
    const json = await fetchJSON(`${API}/api/chart_lightweight/${t}?period=${p}`);
    STATE.chartData = json;

    if (json.error) {
      chartEl.innerHTML = `<div class="loading">Error: ${json.error}</div>`;
      return;
    }
    renderChart(json);
  } catch (e) {
    chartEl.innerHTML = `<div class="loading">Failed to load chart: ${e.message}</div>`;
  }
}

// ---- Chart rendering (Plotly.js) ----

function renderChart (json) {
  const data = json.data;  // array of row objects

  // Extract arrays from row-oriented data
  const dates = data.map(r => r.date);
  const open  = data.map(r => r.Open);
  const high  = data.map(r => r.High);
  const low   = data.map(r => r.Low);
  const close = data.map(r => r.Close);
  const ama   = data.map(r => r.AMA);
  const smfi  = data.map(r => r.SMFI);
  const adx   = data.map(r => r.ADX);
  const eff   = data.map(r => r.effective_signal);
  const state = data.map(r => r.hysteresis_state);

  const thr = STATE.thresholds;
  const n = data.length;
  const bars = Math.min(STATE.displayBars, n);
  const viewStart = dates[Math.max(0, n - bars)];
  const viewEnd = dates[n - 1];

  // ---- Trace definitions ----

  // Row 1: Price chart
  const traceCandle = {
    x: dates, open: open, high: high, low: low, close: close,
    type: "candlestick",
    name: "Price",
    increasing: { line: { color: "#26A69A" } },
    decreasing: { line: { color: "#EF5350" } },
    xaxis: "x", yaxis: "y",
  };

  const traceAMA = {
    x: dates, y: ama,
    type: "scatter", mode: "lines",
    name: "AMA",
    line: { color: "#FFA726", width: 1.5 },
    xaxis: "x", yaxis: "y",
  };

  // Row 2: Indicator chart
  const traceEff = {
    x: dates, y: eff,
    type: "scatter", mode: "lines",
    name: "Eff Signal",
    line: { color: "#FFF", width: 1 },
    fill: "tozeroy",
    fillcolor: "rgba(100,100,255,0.15)",
    xaxis: "x2", yaxis: "y2",
  };

  const traceSMFI = {
    x: dates, y: smfi,
    type: "scatter", mode: "lines",
    name: "SMFI",
    line: { color: "#42A5F5", width: 1.2 },
    xaxis: "x2", yaxis: "y2",
  };

  const traceADX = {
    x: dates, y: adx,
    type: "scatter", mode: "lines",
    name: "ADX",
    line: { color: "#AB47BC", width: 1, dash: "dot" },
    xaxis: "x2", yaxis: "y2",
  };

  // Entry/exit markers
  const longX = [], longY = [];
  const shortX = [], shortY = [];
  for (let i = 1; i < n; i++) {
    if (state[i] !== state[i-1]) {
      if (state[i] === 1) {
        longX.push(dates[i]); longY.push(eff[i]);
      } else if (state[i] === -1) {
        shortX.push(dates[i]); shortY.push(eff[i]);
      }
    }
  }

  const traceLong = {
    x: longX, y: longY,
    type: "scatter", mode: "markers",
    name: "LONG",
    marker: { symbol: "triangle-up", size: 10, color: "#00CC66",
              line: { width: 1, color: "#FFF" } },
    xaxis: "x2", yaxis: "y2",
  };

  const traceShort = {
    x: shortX, y: shortY,
    type: "scatter", mode: "markers",
    name: "SHORT",
    marker: { symbol: "triangle-down", size: 10, color: "#FF4444",
              line: { width: 1, color: "#FFF" } },
    xaxis: "x2", yaxis: "y2",
  };

  // Threshold horizontal lines (shapes)
  const shapes = [];
  const annotations = [];
  const hlines = [
    { y: thr.long_entry,  label: "LONG Entry",   color: "rgba(0,204,102,0.5)" },
    { y: thr.long_exit,   label: "LONG Exit",    color: "rgba(0,204,102,0.25)" },
    { y: thr.short_exit,  label: "SHORT Cover",  color: "rgba(255,68,68,0.25)" },
    { y: thr.short_entry, label: "SHORT Entry",  color: "rgba(255,68,68,0.5)" },
  ];
  for (const hl of hlines) {
    shapes.push({
      type: "line", xref: "x2 domain", yref: "y2",
      x0: 0, x1: 1, y0: hl.y, y1: hl.y,
      line: { dash: "dash", color: hl.color },
    });
    annotations.push({
      xref: "x2 domain", yref: "y2",
      x: 1.01, y: hl.y, text: hl.label,
      showarrow: false, xanchor: "left",
      font: { size: 9, color: hl.color },
    });
  }

  const lastState = state.length ? MAP[String(state[state.length - 1])] || "FLAT" : "FLAT";
  const lastSig = eff.length ? eff[eff.length - 1] : 0;
  const title = `${json.ticker}  ${lastState}  Signal: ${lastSig.toFixed(1)}`;

  // ---- Layout ----
  const layout = {
    title: { text: title, x: 0.02, y: 0.98, xanchor: "left", font: { size: 16, color: "#FAFAFA" } },
    template: "plotly_dark",
    paper_bgcolor: "#1A1C23",
    plot_bgcolor: "#1A1C23",
    height: 720,
    dragmode: "pan",
    margin: { l: 10, r: 20, t: 50, b: 30 },
    legend: { orientation: "h", yanchor: "top", y: 1.08, xanchor: "center", x: 0.5,
              font: { size: 10 } },

    // Row 1: price
    xaxis: {
      domain: [0, 1], anchor: "y",
      range: [viewStart, viewEnd],
      rangeslider: { visible: false },
      constrain: "domain",
      showgrid: true, gridcolor: "rgba(128,128,128,0.12)",
    },
    yaxis: {
      domain: [0.42, 1], anchor: "x",
      title: "Price",
      fixedrange: false,
    },

    // Row 2: indicators
    xaxis2: {
      domain: [0, 1], anchor: "y2",
      range: [viewStart, viewEnd],
      rangeslider: { visible: true, thickness: 0.06,
                     range: [dates[0], dates[dates.length - 1]] },
      constrain: "domain",
      showgrid: true, gridcolor: "rgba(128,128,128,0.12)",
    },
    yaxis2: {
      domain: [0, 0.36], anchor: "x2",
      title: "Signal / Value",
      range: [-120, 120],
      fixedrange: false,
      showgrid: true, gridcolor: "rgba(128,128,128,0.15)",
    },

    shapes: shapes,
    annotations: annotations,
  };

  const allTraces = [traceCandle, traceAMA, traceEff, traceSMFI, traceADX, traceLong, traceShort];

  const config = {
    scrollZoom: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    displaylogo: false,
    responsive: true,
  };

  // Render both charts as a single combined figure inside #chart-price
  Plotly.newPlot("chart-price", allTraces, layout, config);
  // Hide the indicator container — it's now part of the combined figure
  $("#chart-indicator").style.display = "none";

  // Make the combined chart fill both containers
  adjustChartSize();
}

function adjustChartSize () {
  const main = $("#main");
  const priceEl = $("#chart-price");
  if (main && priceEl) {
    const h = main.clientHeight - 180; // minus cards + metrics
    priceEl.style.minHeight = Math.max(620, h) + "px";
  }
}

// ---- Signal Cards ----

function renderSignalCards () {
  const container = $("#signal-cards");
  if (!container) return;
  const s = STATE.signals;
  const thr = STATE.thresholds;
  if (!s || !thr) return;

  container.innerHTML = "";
  const tickers = Object.keys(s).sort();

  for (const t of tickers) {
    const sig = s[t];
    const state = sig.state || "FLAT";
    const color = COL[state] || "#888";
    const delta = sig.signal - sig.prev_signal;
    const deltaStr = delta !== 0 ? `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}` : "—";

    const card = document.createElement("div");
    card.className = "signal-card";
    card.onclick = () => selectTicker(t);
    card.innerHTML = `
      <div class="top">
        <span class="ticker-name">${t}</span>
        <span class="state-badge ${state}" style="color:${color}">${state}</span>
      </div>
      <div class="sig-row">
        Sig: ${sig.signal >= 0 ? "+" : ""}${sig.signal.toFixed(1)}
        <span class="delta">(${deltaStr})</span>
      </div>
      <div class="thresh">
        L&gt;${thr.long_entry}/${thr.long_exit}
        S&lt;${thr.short_entry}/${thr.short_exit}
      </div>
      <div style="font-size:0.8em;color:#CCC;">$${sig.close.toFixed(2)}</div>
      <div class="indicators">
        ADX:${sig.adx.toFixed(0)} SMFI:${sig.smfi.toFixed(0)} CI:${sig.choppiness.toFixed(0)}
      </div>`;
    container.appendChild(card);
  }

  // Summary bar
  let longC = 0, shortC = 0, flatC = 0;
  for (const t of tickers) {
    const st = s[t].state;
    if (st === "LONG") longC++;
    else if (st === "SHORT") shortC++;
    else flatC++;
  }
  $("#status-bar").innerHTML =
    `<span style="color:var(--green)">${longC} LONG</span> &nbsp;|&nbsp; ` +
    `<span style="color:var(--red)">${shortC} SHORT</span> &nbsp;|&nbsp; ` +
    `<span style="color:var(--text3)">${flatC} FLAT</span> &nbsp;|&nbsp; ` +
    `Updated: ${new Date().toLocaleTimeString()}`;
}

// ---- Metrics Row ----

function renderMetrics () {
  const container = $("#metrics-row");
  if (!container) return;
  const s = STATE.signals;
  const t = STATE.ticker;
  if (!s || !s[t]) return;
  const sig = s[t];
  const cd = STATE.chartData;
  let prevClose = 0;
  if (cd && cd.data && cd.data.length >= 2) {
    prevClose = cd.data[cd.data.length - 2].Close || 0;
  }

  const closeDelta = sig.close - prevClose;
  const sigDelta = sig.signal - sig.prev_signal;

  container.innerHTML = `
    <div class="metric-card">
      <div class="label">Close</div>
      <div class="value">$${sig.close.toFixed(2)}</div>
      <div class="delta ${closeDelta >= 0 ? "up" : "down"}">${closeDelta >= 0 ? "+" : ""}${closeDelta.toFixed(2)}</div>
    </div>
    <div class="metric-card">
      <div class="label">Signal</div>
      <div class="value">${sig.signal >= 0 ? "+" : ""}${sig.signal.toFixed(1)}</div>
      <div class="delta ${sigDelta >= 0 ? "up" : "down"}">${sigDelta >= 0 ? "+" : ""}${sigDelta.toFixed(1)}</div>
    </div>
    <div class="metric-card">
      <div class="label">AMA</div>
      <div class="value">$${sig.ama.toFixed(2)}</div>
      <div class="delta"></div>
    </div>
    <div class="metric-card">
      <div class="label">ATR</div>
      <div class="value">$${sig.atr.toFixed(2)}</div>
      <div class="delta"></div>
    </div>
    <div class="metric-card">
      <div class="label">SMFI</div>
      <div class="value">${sig.smfi.toFixed(1)}</div>
      <div class="delta"></div>
    </div>
    <div class="metric-card">
      <div class="label">ADX</div>
      <div class="value">${sig.adx.toFixed(0)}</div>
      <div class="delta"></div>
    </div>`;
}

// ---- Quick View ----

function renderQuickView () {
  const container = $("#quick-view");
  if (!container) return;
  container.innerHTML = "";
  const tickers = Object.keys(STATE.signals || {}).sort();
  if (!tickers.length) {
    // Fallback: use the hardcoded list
    for (const t of ["QQQ","SPY","AAPL","AMZN","GOOGL","META","MSFT","NVDA","TSLA"]) {
      const btn = document.createElement("button");
      btn.className = "qv-btn" + (t === STATE.ticker ? " active" : "");
      btn.textContent = t;
      btn.onclick = () => selectTicker(t);
      container.appendChild(btn);
    }
    return;
  }
  for (const t of tickers) {
    const btn = document.createElement("button");
    btn.className = "qv-btn" + (t === STATE.ticker ? " active" : "");
    btn.textContent = t;
    btn.onclick = () => selectTicker(t);
    container.appendChild(btn);
  }
}

// ---- Positions ----

function renderPositions () {
  const container = $("#positions-panel");
  if (!container) return;
  const pos = STATE.positions || {};
  const active = Object.entries(pos).filter(([_, v]) => v.state !== "FLAT");
  if (!active.length) {
    container.innerHTML = '<div style="font-size:0.75em;color:var(--text3)">No active positions</div>';
    return;
  }
  container.innerHTML = active.map(([t, p]) => `
    <div class="pos-item ${p.state}">
      <span class="ticker">${t}</span> ${p.state}
      &nbsp;Entry: ${p.entry_price || "-"}
      &nbsp;Stop: ${p.stop_level || "-"}
    </div>`).join("");
}

// ---- Ticker switching ----

function selectTicker (t) {
  STATE.ticker = t;
  $("#ticker-select").value = t;
  loadChart();
  renderMetrics();
  renderQuickView();
  // Update URL hash so the page can be bookmarked
  window.location.hash = t;
}

// ---- Event bindings ----

function bindEvents () {
  $("#ticker-select").addEventListener("change", function () {
    selectTicker(this.value);
  });

  $("#bars-select").addEventListener("change", function () {
    STATE.displayBars = parseInt(this.value);
    if (STATE.chartData) renderChart(STATE.chartData);
  });

  $("#period-select").addEventListener("change", function () {
    STATE.period = this.value;
    loadSignals().then(() => {
      renderSignalCards();
      renderMetrics();
      renderQuickView();
    });
    loadChart();
  });

  $("#refresh-select").addEventListener("change", function () {
    STATE.refreshSec = parseInt(this.value);
    scheduleRefresh();
  });

  $("#refresh-btn").addEventListener("click", function () {
    Promise.all([loadSignals(), loadPositions(), loadChart()]).then(() => {
      renderSignalCards();
      renderPositions();
      renderMetrics();
      renderQuickView();
    });
  });

  window.addEventListener("resize", adjustChartSize);
}

// ---- Auto-refresh ----

function scheduleRefresh () {
  if (STATE.timerId) clearInterval(STATE.timerId);
  if (STATE.refreshSec > 0) {
    STATE.timerId = setInterval(autoRefresh, STATE.refreshSec * 1000);
  }
}

async function autoRefresh () {
  await loadSignals();
  await loadPositions();
  await loadChart();
  renderSignalCards();
  renderPositions();
  renderMetrics();
  renderQuickView();
}

// ---- Bootstrap ----

function populateTickerSelect () {
  const sel = $("#ticker-select");
  if (!sel) return;
  const tickers = ["QQQ", "SPY", "AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"];
  sel.innerHTML = tickers.map(t =>
    `<option value="${t}" ${t === STATE.ticker ? "selected" : ""}>${t}</option>`
  ).join("");
}

window.addEventListener("DOMContentLoaded", () => {
  populateTickerSelect();
  bindEvents();
  init();
});
