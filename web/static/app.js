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
	period: "3mo", // always 3mo for signals (fast)
	refreshSec: 300,
	chartData: null, // full OHLCV + indicator data for current ticker
	fullData: null, // 2y data loaded in background
	fullDataLoaded: false, // whether 2y data has been merged
	signals: null, // latest signals for all tickers
	thresholds: null, // hysteresis entry/exit levels
	positions: null,
	timerId: null,
	cardsVisible: true,
};

const MAP = { 1: "LONG", 0: "FLAT", "-1": "SHORT" };
const COL = {
	LONG: "#00FF88",
	SHORT: "#FF3355",
	FLAT: "#888",
	ERROR: "#FF3355",
};
const API = ""; // same-origin

// ---- Helpers ----

function $(sel) {
	return document.querySelector(sel);
}
function $$(sel) {
	return document.querySelectorAll(sel);
}

async function fetchJSON(url) {
	const r = await fetch(url);
	if (!r.ok) throw new Error(`HTTP ${r.status}`);
	return r.json();
}

// ---- Initialisation ----

async function init() {
	await loadThresholds();
	await loadSignals();
	await loadPositions();
	await loadChart();
	renderQuickView();
	renderSignalCards();
	renderPositions();
	renderMetrics();
	pulseUpdate();
	scheduleRefresh();
}

// ---- API calls ----

async function loadThresholds() {
	STATE.thresholds = await fetchJSON(`${API}/api/thresholds`);
}

async function loadSignals() {
	STATE.signals = await fetchJSON(
		`${API}/api/signals?period=${STATE.period}`,
	);
}

async function loadPositions() {
	try {
		STATE.positions = await fetchJSON(`${API}/api/positions`);
	} catch (_) {
		STATE.positions = {};
	}
}

async function loadChart() {
	const t = STATE.ticker;
	STATE.fullDataLoaded = false;
	STATE.fullData = null;
	const chartEl = $("#chart-price");
	chartEl.innerHTML = '<div class="loading">Loading...</div>';

	try {
		const json = await fetchJSON(
			`${API}/api/chart_lightweight/${t}?period=3mo`,
		);
		STATE.chartData = json;

		if (json.error) {
			chartEl.innerHTML = `<div class="loading">Error: ${json.error}</div>`;
			return;
		}
		renderChart(json);
		// Background: load 2y data for seamless panning
		loadFullData(t);
	} catch (e) {
		chartEl.innerHTML = `<div class="loading">Failed to load chart: ${e.message}</div>`;
	}
}

async function loadFullData(ticker) {
	try {
		const json = await fetchJSON(
			`${API}/api/chart_lightweight/${ticker}?period=2y`,
		);
		if (json.error) return;
		STATE.fullData = json;
		mergeFullData(json);
	} catch (_) {
		/* silently ignore background load failures */
	}
}

function mergeFullData(json) {
	if (!STATE.fullDataLoaded) {
		// First merge: update traces with full data, preserve viewport
		const cd = STATE.chartData;
		if (!cd || !cd.data) return;
		const existingDates = new Set(cd.data.map((r) => r.date));
		const newRows = json.data.filter((r) => !existingDates.has(r.date));
		cd.data = [...newRows, ...cd.data].sort((a, b) =>
			a.date.localeCompare(b.date),
		);
		STATE.chartData = cd;
	}
	STATE.fullDataLoaded = true;
	updateChartTraces();
}

function updateChartTraces() {
	const cd = STATE.chartData;
	if (!cd || !cd.data) return;

	// Snapshot current viewport BEFORE touching the chart
	const chartEl = $("#chart-price");
	let snapRange = null;
	if (chartEl && chartEl._fullLayout && chartEl._fullLayout.xaxis) {
		const xr = chartEl._fullLayout.xaxis.range;
		if (xr && xr.length === 2) snapRange = [xr[0], xr[1]];
	}

	const data = cd.data;
	const dates = data.map((r) => r.date);
	const open = data.map((r) => r.Open);
	const high = data.map((r) => r.High);
	const low = data.map((r) => r.Low);
	const close = data.map((r) => r.Close);
	const ama = data.map((r) => r.AMA);
	const smfi = data.map((r) => r.SMFI);
	const adx = data.map((r) => r.ADX);
	const eff = data.map((r) => r.effective_signal);
	const state = data.map((r) => r.hysteresis_state);

	// Recompute entry markers over full data
	const longX = [],
		longY = [];
	const shortX = [],
		shortY = [];
	for (let i = 1; i < data.length; i++) {
		if (state[i] !== state[i - 1]) {
			if (state[i] === 1) {
				longX.push(dates[i]);
				longY.push(eff[i]);
			} else if (state[i] === -1) {
				shortX.push(dates[i]);
				shortY.push(eff[i]);
			}
		}
	}

	// Suppress relayout clamping while we update
	_clamping = true;

	Plotly.update(
		"chart-price",
		{
			x: [dates, dates, dates, dates, dates, longX, shortX],
			open: [open],
			high: [high],
			low: [low],
			close: [close],
			y: [null, ama, eff, smfi, adx, longY, shortY],
		},
		{
			"xaxis2.rangeslider.range": [dates[0], dates[dates.length - 1]],
		},
		[0, 1, 2, 3, 4, 5, 6],
	).then(function () {
		// Restore saved viewport
		if (snapRange) {
			Plotly.relayout("chart-price", {
				"xaxis.range": snapRange,
				"xaxis2.range": snapRange,
			}).then(function () {
				_clamping = false;
			});
		} else {
			_clamping = false;
		}
	});
}

// ---- Chart rendering (Plotly.js) ----

function renderChart(json) {
	const data = json.data; // array of row objects

	// Extract arrays from row-oriented data
	const dates = data.map((r) => r.date);
	const open = data.map((r) => r.Open);
	const high = data.map((r) => r.High);
	const low = data.map((r) => r.Low);
	const close = data.map((r) => r.Close);
	const ama = data.map((r) => r.AMA);
	const smfi = data.map((r) => r.SMFI);
	const adx = data.map((r) => r.ADX);
	const eff = data.map((r) => r.effective_signal);
	const state = data.map((r) => r.hysteresis_state);

	const thr = STATE.thresholds;
	const n = data.length;
	const bars = Math.min(STATE.displayBars, n);
	const viewStart = dates[Math.max(0, n - bars)];
	const viewEnd = dates[n - 1];

	// ---- Trace definitions ----

	// Row 1: Price chart
	const traceCandle = {
		x: dates,
		open: open,
		high: high,
		low: low,
		close: close,
		type: "candlestick",
		name: "Price",
		increasing: { line: { color: "#00FF88" } },
		decreasing: { line: { color: "#FF3355" } },
		xaxis: "x",
		yaxis: "y",
	};

	const traceAMA = {
		x: dates,
		y: ama,
		type: "scatter",
		mode: "lines",
		name: "AMA",
		line: { color: "#FFB300", width: 1.5 },
		xaxis: "x",
		yaxis: "y",
	};

	// Row 2: Indicator chart
	const traceEff = {
		x: dates,
		y: eff,
		type: "scatter",
		mode: "lines",
		name: "Eff Signal",
		line: { color: "#FFF", width: 1 },
		fill: "tozeroy",
		fillcolor: "rgba(100,100,255,0.15)",
		xaxis: "x2",
		yaxis: "y2",
	};

	const traceSMFI = {
		x: dates,
		y: smfi,
		type: "scatter",
		mode: "lines",
		name: "SMFI",
		line: { color: "#00E5FF", width: 1.2 },
		xaxis: "x2",
		yaxis: "y2",
	};

	const traceADX = {
		x: dates,
		y: adx,
		type: "scatter",
		mode: "lines",
		name: "ADX",
		line: { color: "#AB47BC", width: 1, dash: "dot" },
		xaxis: "x2",
		yaxis: "y2",
	};

	// Entry/exit markers
	const longX = [],
		longY = [];
	const shortX = [],
		shortY = [];
	for (let i = 1; i < n; i++) {
		if (state[i] !== state[i - 1]) {
			if (state[i] === 1) {
				longX.push(dates[i]);
				longY.push(eff[i]);
			} else if (state[i] === -1) {
				shortX.push(dates[i]);
				shortY.push(eff[i]);
			}
		}
	}

	const traceLong = {
		x: longX,
		y: longY,
		type: "scatter",
		mode: "markers",
		name: "LONG",
		marker: {
			symbol: "triangle-up",
			size: 10,
			color: "#00FF88",
			line: { width: 1, color: "#FFF" },
		},
		xaxis: "x2",
		yaxis: "y2",
	};

	const traceShort = {
		x: shortX,
		y: shortY,
		type: "scatter",
		mode: "markers",
		name: "SHORT",
		marker: {
			symbol: "triangle-down",
			size: 10,
			color: "#FF3355",
			line: { width: 1, color: "#FFF" },
		},
		xaxis: "x2",
		yaxis: "y2",
	};

	// Threshold horizontal lines (shapes)
	const shapes = [];
	const annotations = [];
	const hlines = [
		{
			y: thr.long_entry,
			label: "LONG Entry",
			color: "rgba(0,255,136,0.45)",
		},
		{ y: thr.long_exit, label: "LONG Exit", color: "rgba(0,255,136,0.22)" },
		{
			y: thr.short_exit,
			label: "SHORT Cover",
			color: "rgba(255,51,85,0.22)",
		},
		{
			y: thr.short_entry,
			label: "SHORT Entry",
			color: "rgba(255,51,85,0.45)",
		},
	];
	for (const hl of hlines) {
		shapes.push({
			type: "line",
			xref: "x2 domain",
			yref: "y2",
			x0: 0,
			x1: 1,
			y0: hl.y,
			y1: hl.y,
			line: { dash: "dash", color: hl.color },
		});
		annotations.push({
			xref: "x2 domain",
			yref: "y2",
			x: 1.01,
			y: hl.y,
			text: hl.label,
			showarrow: false,
			xanchor: "left",
			font: { size: 9, color: hl.color },
		});
	}

	const lastState = state.length
		? MAP[String(state[state.length - 1])] || "FLAT"
		: "FLAT";
	const lastSig = eff.length ? eff[eff.length - 1] : 0;
	const title = `${json.ticker}  ${lastState}  Signal: ${lastSig.toFixed(1)}`;

	// ---- Layout ----
	const layout = {
		title: {
			text: title,
			x: 0.02,
			y: 0.98,
			xanchor: "left",
			font: { size: 15, color: "#F0F0F5", family: "Chakra Petch" },
		},
		template: "plotly_dark",
		paper_bgcolor: "#0E0E14",
		plot_bgcolor: "#0E0E14",
		height: 720,
		dragmode: "pan",
		margin: { l: 10, r: 20, t: 50, b: 30 },
		legend: {
			orientation: "h",
			yanchor: "top",
			y: 1.08,
			xanchor: "center",
			x: 0.5,
			font: { size: 10, family: "JetBrains Mono" },
		},

		// Row 1: price
		xaxis: {
			domain: [0, 1],
			anchor: "y",
			range: [viewStart, viewEnd],
			autorange: false,
			rangeslider: { visible: false },
			constrain: "domain",
			showgrid: true,
			gridcolor: "rgba(255,255,255,0.04)",
		},
		yaxis: {
			domain: [0.42, 1],
			anchor: "x",
			title: "Price",
			fixedrange: false,
		},

		// Row 2: indicators (synced to price x-axis)
		xaxis2: {
			domain: [0, 1],
			anchor: "y2",
			range: [viewStart, viewEnd],
			autorange: false,
			matches: "x",
			rangeslider: {
				visible: true,
				thickness: 0.06,
				range: [dates[0], dates[dates.length - 1]],
			},
			constrain: "domain",
			showgrid: true,
			gridcolor: "rgba(255,255,255,0.05)",
		},
		yaxis2: {
			domain: [0, 0.36],
			anchor: "x2",
			title: "Signal / Value",
			range: [-120, 120],
			fixedrange: false,
			showgrid: true,
			gridcolor: "rgba(255,255,255,0.06)",
		},

		shapes: shapes,
		annotations: annotations,
	};

	const allTraces = [
		traceCandle,
		traceAMA,
		traceEff,
		traceSMFI,
		traceADX,
		traceLong,
		traceShort,
	];

	const config = {
		scrollZoom: true,
		displayModeBar: true,
		modeBarButtonsToRemove: ["lasso2d", "select2d"],
		displaylogo: false,
		responsive: true,
	};

	// Render combined figure inside #chart-price
	Plotly.newPlot("chart-price", allTraces, layout, config).then(function () {
		setupRelayoutHandler();
		adjustChartSize();
	});
}

function adjustChartSize() {
	const main = $("#main");
	const metricsEl = $("#metrics-row");
	const cardsEl = $("#signal-cards");
	if (!main) return;

	const topBarH = 42;
	const mainPad = 32; // 16px top + 16px bottom
	const metricsH = metricsEl ? metricsEl.offsetHeight : 0;
	const cardsH =
		cardsEl && !cardsEl.classList.contains("hidden")
			? cardsEl.offsetHeight
			: 0;
	const gaps = 20; // gap between chart/metrics + metrics/cards

	const available =
		window.innerHeight - topBarH - mainPad - metricsH - cardsH - gaps;
	const h = Math.max(400, available);
	Plotly.relayout("chart-price", { height: h });
}

// ---- Auto-fit y-axis to visible candles (debounced) ----
let _autoFitTimer = null;
let _clamping = false; // guard against recursive relayout

function setupRelayoutHandler() {
	const chartEl = $("#chart-price");
	if (!chartEl) return;

	chartEl.on("plotly_relayout", function (evt) {
		// Extract x-axis range from either event format (xaxis or xaxis2)
		let r0, r1;
		const prefix =
			evt["xaxis.range"] != null
				? "xaxis"
				: evt["xaxis2.range"] != null
					? "xaxis2"
					: null;
		if (prefix) {
			r0 = evt[prefix + ".range"][0];
			r1 = evt[prefix + ".range"][1];
		}
		if (r0 == null) {
			// Try dotted keys
			for (const p of ["xaxis", "xaxis2"]) {
				if (evt[p + ".range[0]"] !== undefined) {
					r0 = evt[p + ".range[0]"];
					r1 = evt[p + ".range[1]"];
					break;
				}
			}
		}
		if (r0 == null || r1 == null) return;

		// Clamp to data boundaries (skip if we're the ones triggering the relayout)
		if (!_clamping) {
			clampViewport(r0, r1);
		}

		// Debounced auto-fit y-axis
		clearTimeout(_autoFitTimer);
		_autoFitTimer = setTimeout(autoFitYAxis, 150);
	});
}

function clampViewport(r0, r1) {
	const cd = STATE.chartData;
	if (!cd || !cd.data || cd.data.length === 0) return;
	const dates = cd.data.map((r) => r.date);
	const dataMin = dates[0];
	const dataMax = dates[dates.length - 1];

	// Convert to timestamps for arithmetic (ISO strings don't subtract)
	const t0 = new Date(r0).getTime();
	const t1 = new Date(r1).getTime();
	const span = t1 - t0;
	const tMin = new Date(dataMin).getTime();
	const tMax = new Date(dataMax).getTime();
	const dataSpan = tMax - tMin;

	// Right boundary: allow 50% extension past latest data
	const tMaxExtended = tMax + span * 0.5;

	let clamped = false;
	let nt0 = t0,
		nt1 = t1;

	// Left: hard clamp — no panning before earliest data
	if (t0 < tMin) {
		nt0 = tMin;
		nt1 = Math.min(tMaxExtended, tMin + span);
		clamped = true;
	}
	// Right: soft clamp — allow some extension past latest data
	if (t1 > tMaxExtended) {
		nt1 = tMaxExtended;
		nt0 = Math.max(tMin, tMaxExtended - span);
		clamped = true;
	}
	// Prevent zooming out beyond full data range
	if (span > dataSpan) {
		nt0 = tMin;
		nt1 = tMax;
		clamped = true;
	}

	if (clamped) {
		_clamping = true;
		const newR0 = new Date(nt0).toISOString();
		const newR1 = new Date(nt1).toISOString();
		Plotly.relayout("chart-price", { "xaxis.range": [newR0, newR1] }).then(
			function () {
				_clamping = false;
			},
		);
	}
}

function autoFitYAxis() {
	const cd = STATE.chartData;
	if (!cd || !cd.data || cd.data.length === 0) return;

	const chartEl = $("#chart-price");
	if (!chartEl) return;

	const layout = chartEl._fullLayout;
	if (!layout || !layout.xaxis || !layout.xaxis.range) return;

	const [x0, x1] = layout.xaxis.range;
	const data = cd.data;

	// Find visible OHLC bars
	let visHigh = -Infinity,
		visLow = Infinity,
		found = false;
	for (const r of data) {
		const d = r.date;
		if (d >= x0 && d <= x1) {
			if (r.High != null && r.High > visHigh) visHigh = r.High;
			if (r.Low != null && r.Low < visLow) visLow = r.Low;
			found = true;
		}
	}
	if (!found || visHigh === -Infinity || visLow === Infinity) return;

	const pad = (visHigh - visLow) * 0.1;
	const yLow = visLow - pad;
	const yHigh = visHigh + pad;

	// Don't update if y-range is already close (avoid feedback loops)
	const currYRange = layout.yaxis && layout.yaxis.range;
	if (
		currYRange &&
		Math.abs(currYRange[0] - yLow) < 1 &&
		Math.abs(currYRange[1] - yHigh) < 1
	)
		return;

	Plotly.relayout("chart-price", { "yaxis.range": [yLow, yHigh] });
}

// ---- Signal Cards ----

function renderSignalCards() {
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
		const deltaStr =
			delta !== 0 ? `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}` : "—";

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
      <div class="price-line">$${sig.close.toFixed(2)}</div>
      <div class="indicators">
        ADX:${sig.adx.toFixed(0)} SMFI:${sig.smfi.toFixed(0)} CI:${sig.choppiness.toFixed(0)}
      </div>`;
		container.appendChild(card);
	}

	// Summary bar
	let longC = 0,
		shortC = 0,
		flatC = 0;
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

function renderMetrics() {
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

function renderQuickView() {
	const container = $("#quick-view");
	if (!container) return;
	container.innerHTML = "";
	const tickers = Object.keys(STATE.signals || {}).sort();
	if (!tickers.length) {
		// Fallback: use the hardcoded list
		for (const t of [
			"QQQ",
			"SPY",
			"AAPL",
			"AMZN",
			"GOOGL",
			"META",
			"MSFT",
			"NVDA",
			"TSLA",
		]) {
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

function renderPositions() {
	const container = $("#positions-panel");
	if (!container) return;
	const pos = STATE.positions || {};
	const active = Object.entries(pos).filter(([_, v]) => v.state !== "FLAT");
	if (!active.length) {
		container.innerHTML =
			'<div style="font-size:0.75em;color:var(--text3)">No active positions</div>';
		return;
	}
	container.innerHTML = active
		.map(
			([t, p]) =>
				`<details class="pos-item ${p.state}">
      <summary>
        <span class="pos-summary-ticker">${t}</span>
        <span class="pos-summary-state ${p.state}">${p.state}</span>
        <span class="pos-summary-signal">${p.signal != null ? (p.signal >= 0 ? "+" : "") + p.signal.toFixed(1) : "-"}</span>
      </summary>
      <div class="pos-detail">
        <span><span class="dlbl">Entry</span> ${p.entry_price || "-"}</span>
        <span><span class="dlbl">Stop</span> ${p.stop_level || "-"}</span>
        <span><span class="dlbl">Size</span> ${p.position_size != null ? p.position_size : "-"}</span>
      </div>
    </details>`,
		)
		.join("");
}

// ---- Ticker switching ----

function selectTicker(t) {
	STATE.ticker = t;
	$("#ticker-select").value = t;
	loadChart();
	renderMetrics();
	renderQuickView();
	// Update URL hash so the page can be bookmarked
	window.location.hash = t;
}

// ---- Event bindings ----

function bindEvents() {
	$("#ticker-select").addEventListener("change", function () {
		selectTicker(this.value);
	});

	$("#bars-select").addEventListener("change", function () {
		STATE.displayBars = parseInt(this.value);
		if (STATE.chartData) renderChart(STATE.chartData);
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
			pulseUpdate();
		});
	});

	$("#toggle-cards-btn").addEventListener("click", toggleSignalCards);

	$("#lang-toggle").addEventListener("click", function (e) {
		e.stopPropagation(); // don't toggle the <details>
		toggleLanguage();
	});

	window.addEventListener("resize", adjustChartSize);
}

// ---- Language Toggle ----

function toggleLanguage() {
	const guide = $("#guide-content");
	const label = $("#lang-label");
	if (!guide || !label) return;
	if (guide.classList.contains("guide-cn")) {
		guide.classList.remove("guide-cn");
		guide.classList.add("guide-en");
		label.textContent = "中文";
	} else {
		guide.classList.remove("guide-en");
		guide.classList.add("guide-cn");
		label.textContent = "English";
	}
}

// ---- Toggle Signal Cards ----

function toggleSignalCards() {
	const cards = $("#signal-cards");
	const btn = $("#toggle-cards-btn");
	if (!cards || !btn) return;
	STATE.cardsVisible = !STATE.cardsVisible;
	if (STATE.cardsVisible) {
		cards.classList.remove("hidden");
		btn.classList.remove("cards-off");
	} else {
		cards.classList.add("hidden");
		btn.classList.add("cards-off");
	}
	adjustChartSize();
}

// ---- Auto-refresh ----

function scheduleRefresh() {
	if (STATE.timerId) clearInterval(STATE.timerId);
	if (STATE.refreshSec > 0) {
		STATE.timerId = setInterval(autoRefresh, STATE.refreshSec * 1000);
	}
}

function pulseUpdate() {
	const dot = $("#update-pulse");
	if (dot) {
		dot.classList.remove("live");
		void dot.offsetWidth; // force reflow
		dot.classList.add("live");
	}
	const timeEl = $("#update-time");
	if (timeEl) {
		timeEl.textContent = new Date().toLocaleTimeString();
	}
}

async function autoRefresh() {
	await loadSignals();
	await loadPositions();
	await loadChart();
	renderSignalCards();
	renderPositions();
	renderMetrics();
	renderQuickView();
	pulseUpdate();
}

// ---- Bootstrap ----

function populateTickerSelect() {
	const sel = $("#ticker-select");
	if (!sel) return;
	const tickers = [
		"QQQ",
		"SPY",
		"AAPL",
		"AMZN",
		"GOOGL",
		"META",
		"MSFT",
		"NVDA",
		"TSLA",
	];
	sel.innerHTML = tickers
		.map(
			(t) =>
				`<option value="${t}" ${t === STATE.ticker ? "selected" : ""}>${t}</option>`,
		)
		.join("");
}

window.addEventListener("DOMContentLoaded", () => {
	populateTickerSelect();
	bindEvents();
	init();
	// Fallback: resize after everything settles
	setTimeout(adjustChartSize, 800);
});
