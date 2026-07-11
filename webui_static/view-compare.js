/* Compare view: pick saved runs, interactive metric charts, values table,
   batch-sweep charts and the export group (ZIP / HTML / PDF). */

(function () {
  "use strict";

  const {
    state, esc, fmt, seriesColor, ctxNum, fmtDate, pageHead,
    resultName, resultSubtitle, seriesLabel, matchesFilter, metricsForKeys,
    ensureResults, ensureDetail, toggleCompare,
  } = CB;

  async function renderCompareView() {
    const root = document.getElementById("view-compare");
    root.innerHTML = `${pageHead("Bench · Analysis", "Compare",
      "Pick up to 8 saved runs — each keeps its color while selected.")}
    <div class="compare-layout">
      <div class="panel">
        <div class="panel-head"><span class="panel-title">Runs</span>
          <button class="btn small" id="cmpClear">Clear</button></div>
        <input type="text" id="cmpFilter" placeholder="Filter…" style="width:100%;margin-bottom:8px"
          class="num">
        <div class="pick-list" id="cmpPickList"></div>
      </div>
      <div>
        <div class="metric-tabs" id="cmpTabs"></div>
        <div class="chart-controls">
          <label class="check"><input type="checkbox" id="cmpGrid" ${state.compare.grid ? "checked" : ""}> All metrics grid</label>
          <span style="flex:1"></span>
          ${CB.report.exportGroupHtml()}
        </div>
        <div id="cmpCharts"></div>
      </div>
    </div>`;

    document.getElementById("cmpClear").addEventListener("click", () => {
      state.compare.slots.clear();
      renderCompareView();
    });
    document.getElementById("cmpFilter").addEventListener("input", renderPickList);
    document.getElementById("cmpGrid").addEventListener("change", e => {
      state.compare.grid = e.target.checked;
      renderCompareCharts();
    });
    CB.report.wireExportGroup(root, compareEntries, "context-bench-comparison",
      entries => entries.length === 1 ? resultName(entries[0].detail.summary) : "Benchmark comparison");

    try { await ensureResults(); } catch (e) {
      root.querySelector("#cmpPickList").innerHTML = `<div class="empty">${esc(e.message)}</div>`;
      return;
    }
    renderPickList();
    renderCompareCharts();
  }

  async function compareEntries() {
    const entries = [];
    for (const [folder, slot] of [...state.compare.slots.entries()].sort((a, b) => a[1] - b[1])) {
      const detail = await ensureDetail(folder);
      entries.push({ detail, slot, name: seriesLabel(detail.summary) });
    }
    return entries;
  }

  function renderPickList() {
    const list = document.getElementById("cmpPickList");
    if (!list) return;
    const filterNode = document.getElementById("cmpFilter");
    const filter = filterNode ? filterNode.value : "";
    const rows = state.results.filter(r => !r.error && matchesFilter(r, filter));
    if (!rows.length) {
      list.innerHTML = `<div class="empty">No saved results.</div>`;
      return;
    }
    list.innerHTML = rows.map(r => {
      const slot = state.compare.slots.get(r.folder);
      const on = slot != null;
      return `<div class="pick-row ${on ? "on" : ""}" data-folder="${esc(r.folder)}" role="checkbox"
          aria-checked="${on}" tabindex="0">
        <span class="pick-swatch" style="${on ? `background:${seriesColor(slot)}` : ""}"></span>
        <div style="min-width:0">
          <div class="pick-name">${esc(resultName(r))}</div>
          <div class="pick-sub">${esc(resultSubtitle(r))} · ${esc(fmtDate(r.timestamp))}</div>
        </div>
      </div>`;
    }).join("");
    list.querySelectorAll(".pick-row").forEach(row => {
      const toggle = () => {
        const folder = row.dataset.folder;
        toggleCompare(folder, !state.compare.slots.has(folder));
        renderPickList();
        renderCompareCharts();
      };
      row.addEventListener("click", toggle);
      row.addEventListener("keydown", e => {
        if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); }
      });
    });
  }

  async function compareSeriesFor(metricKey) {
    const series = [];
    for (const [folder, slot] of state.compare.slots.entries()) {
      let detail;
      try { detail = await ensureDetail(folder); } catch (e) { continue; }
      series.push({
        name: seriesLabel(detail.summary),
        color: seriesColor(slot),
        slot,
        points: detail.results.map(r => [ctxNum(r.context_size), r[metricKey]]),
      });
    }
    series.sort((a, b) => a.slot - b.slot);
    return series;
  }

  function selectedMetrics() {
    const keys = new Set();
    for (const folder of state.compare.slots.keys()) {
      const summary = state.results.find(r => r.folder === folder);
      if (summary) summary.columns.forEach(c => keys.add(c));
    }
    return metricsForKeys(keys);
  }

  async function renderCompareCharts() {
    const tabs = document.getElementById("cmpTabs");
    const container = document.getElementById("cmpCharts");
    if (!tabs || !container) return;

    if (!state.compare.slots.size) {
      tabs.innerHTML = "";
      container.innerHTML = `<div class="empty"><strong>Nothing selected</strong>
        Pick runs on the left (or select them under Results).</div>`;
      return;
    }

    const metrics = selectedMetrics();
    if (!metrics.find(m => m.key === state.compare.metric) && metrics.length) {
      state.compare.metric = metrics[0].key;
    }
    tabs.innerHTML = metrics.map(m =>
      `<button class="metric-tab ${m.key === state.compare.metric ? "active" : ""}" data-metric="${esc(m.key)}">
        ${esc(m.label)} <span class="unit">${esc(m.unit)}</span></button>`).join("");
    tabs.querySelectorAll(".metric-tab").forEach(btn =>
      btn.addEventListener("click", () => {
        state.compare.metric = btn.dataset.metric;
        renderCompareCharts();
      }));

    if (state.compare.grid) {
      container.innerHTML = `<div class="chart-grid" id="cmpGridWrap"></div>${batchSectionHtml()}`;
      const wrap = document.getElementById("cmpGridWrap");
      for (const metric of metrics) {
        const panel = document.createElement("div");
        panel.className = "panel";
        panel.innerHTML = `<div class="chart-title">${esc(metric.label)} <span class="unit">${esc(metric.unit)}</span></div><div></div>`;
        wrap.appendChild(panel);
        const series = await compareSeriesFor(metric.key);
        Charts.lineChart(panel.lastElementChild, {
          series, logX: true, height: 230, seconds: metric.seconds,
          xLabel: "context", yLabel: metric.unit,
        });
      }
    } else {
      const metric = metrics.find(m => m.key === state.compare.metric) || metrics[0];
      if (!metric) { container.innerHTML = `<div class="empty">No comparable metrics.</div>`; return; }
      container.innerHTML = `<div class="panel">
          <div class="chart-title">${esc(metric.label)} across context size <span class="unit">${esc(metric.unit)}</span></div>
          <div id="cmpHero"></div>
        </div>
        <div class="panel"><div class="chart-title">Values</div><div class="tbl-wrap" id="cmpTable"></div></div>
        ${batchSectionHtml()}`;
      const series = await compareSeriesFor(metric.key);
      Charts.lineChart(document.getElementById("cmpHero"), {
        series, logX: true, height: 440, seconds: metric.seconds,
        xLabel: "context (tokens ×1000)", yLabel: metric.unit,
      });
      renderCompareTable(series, metric);
    }
    await renderBatchCharts();
  }

  function batchSectionHtml() {
    const hasBatch = [...state.compare.slots.keys()].some(folder => {
      const summary = state.results.find(r => r.folder === folder);
      return summary && summary.has_batch;
    });
    if (!hasBatch) return "";
    return `<div style="margin-top:14px">
      <p class="hint" style="margin:0 0 8px">Batch sweep = N parallel clients. »End-to-end« is generated
        tokens ÷ total wall time (prompt phase included) — »decode« is the pure generation rate the
        server reported, summed across clients.</p>
      <div class="chart-grid" id="cmpBatchWrap"></div>
    </div>`;
  }

  async function renderBatchCharts() {
    const wrap = document.getElementById("cmpBatchWrap");
    if (!wrap) return;
    const entries = await compareEntries();
    const defs = CB.report.batchChartDefs(entries);
    const batchEntries = entries.filter(e => e.detail.batch_data && e.detail.batch_data.length);
    wrap.innerHTML = "";
    for (const def of defs) {
      const unit = (def.title.match(/\[(.+)\]/) || [])[1] || "";
      const series = batchEntries.map(e => ({
        name: e.name,
        color: seriesColor(e.slot),
        points: e.detail.batch_data.map(b => [b.batch_size, b[def.field]]),
      })).filter(s => s.points.some(p => p[1] != null && isFinite(p[1])));
      if (!series.length) continue;
      const panel = document.createElement("div");
      panel.className = "panel";
      panel.innerHTML = `<div class="chart-title">${esc(def.title.replace(/ \[.+\]$/, ""))}
        <span class="unit">${esc(unit)}</span></div><div></div>`;
      wrap.appendChild(panel);
      Charts.lineChart(panel.lastElementChild, {
        series, logX: true, height: 240,
        xLabel: "batch size (parallel clients)", yLabel: unit,
        xTickFormat: v => String(v),
      });
    }
  }

  function renderCompareTable(series, metric) {
    const node = document.getElementById("cmpTable");
    if (!node) return;
    const xs = [...new Set(series.flatMap(s => s.points.map(p => p[0])))].sort((a, b) => a - b);
    node.innerHTML = `<table class="tbl">
      <thead><tr><th>context</th>${series.map(s =>
        `<th><span class="legend-swatch" style="background:${s.color};display:inline-block;vertical-align:middle;margin-right:5px"></span>${esc(s.name)}</th>`).join("")}</tr></thead>
      <tbody>${xs.map(x => `<tr><td>${esc(x)}k</td>${series.map(s => {
        const p = s.points.find(pt => pt[0] === x && pt[1] != null);
        return `<td>${p ? esc(fmt(p[1], { seconds: metric.seconds })) : "–"}</td>`;
      }).join("")}</tr>`).join("")}</tbody>
    </table>`;
  }

  CB.views.compare = renderCompareView;
  CB.onCompareResize = renderCompareCharts;
})();
