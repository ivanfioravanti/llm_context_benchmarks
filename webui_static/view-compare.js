/* Compare view: pick saved runs, interactive metric charts, values table,
   batch-sweep charts and the export group (ZIP / HTML / PDF). */

(function () {
  "use strict";

  const {
    state, esc, fmt, seriesColor, ctxNum, cachedSeriesPoints, qmarkHtml, fmtDate, pageHead,
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
          <label class="check" id="cmpBaselineWrap" style="display:none">Relative to
            <select id="cmpBaseline"></select></label>
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
    document.getElementById("cmpBaseline").addEventListener("change", e => {
      state.compare.baseline = e.target.value || null;
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

  function baselineFolder() {
    const folder = state.compare.baseline;
    return folder && state.compare.slots.has(folder) ? folder : null;
  }

  async function compareSeriesFor(metricKey) {
    const series = [];
    for (const [folder, slot] of state.compare.slots.entries()) {
      let detail;
      try { detail = await ensureDetail(folder); } catch (e) { continue; }
      const name = seriesLabel(detail.summary);
      series.push({
        name,
        color: seriesColor(slot),
        slot, folder,
        points: detail.results.map(r => [ctxNum(r.context_size), r[metricKey]]),
      });
      const cached = cachedSeriesPoints(detail, metricKey);
      if (cached) series.push({
        name: `${name} · cached`, color: seriesColor(slot), slot, folder,
        dash: true, isCached: true, points: cached,
      });
    }
    series.sort((a, b) => a.slot - b.slot || (a.isCached ? 1 : 0) - (b.isCached ? 1 : 0));

    // baseline mode: every point becomes a percentage of the baseline run's
    // (cold) value at the same context size — the baseline stays flat at 100
    const base = series.find(s => s.folder === baselineFolder() && !s.isCached);
    if (base) {
      const baseAt = new Map(base.points.filter(p => p[1] != null && isFinite(p[1]) && p[1] > 0));
      for (const s of series) {
        s.points = s.points.map(([x, y]) =>
          [x, y != null && isFinite(y) && baseAt.has(x) ? (y / baseAt.get(x)) * 100 : null]);
      }
    }
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
      `<button class="metric-tab ${m.key === state.compare.metric ? "active" : ""}" data-metric="${esc(m.key)}"
        title="${esc(m.desc || "")}">
        ${esc(m.label)} <span class="unit">${esc(m.unit)}</span></button>`).join("");
    tabs.querySelectorAll(".metric-tab").forEach(btn =>
      btn.addEventListener("click", () => {
        state.compare.metric = btn.dataset.metric;
        renderCompareCharts();
      }));

    syncBaselinePicker();
    const relative = !!baselineFolder();
    const unitFor = metric => (relative ? "% of baseline" : metric.unit);
    const secondsFor = metric => (relative ? false : metric.seconds);

    if (state.compare.grid) {
      container.innerHTML = `<div class="chart-grid" id="cmpGridWrap"></div>${batchSectionHtml()}`;
      const wrap = document.getElementById("cmpGridWrap");
      for (const metric of metrics) {
        const panel = document.createElement("div");
        panel.className = "panel";
        panel.innerHTML = `<div class="chart-title">${esc(metric.label)} <span class="unit">${esc(unitFor(metric))}</span>${qmarkHtml(metric.desc)}</div><div></div>`;
        wrap.appendChild(panel);
        const series = await compareSeriesFor(metric.key);
        Charts.lineChart(panel.lastElementChild, {
          series, logX: true, height: 230, seconds: secondsFor(metric),
          xLabel: "context", yLabel: unitFor(metric),
        });
      }
    } else {
      const metric = metrics.find(m => m.key === state.compare.metric) || metrics[0];
      if (!metric) { container.innerHTML = `<div class="empty">No comparable metrics.</div>`; return; }
      container.innerHTML = `<div class="panel">
          <div class="chart-title">${esc(metric.label)} across context size <span class="unit">${esc(unitFor(metric))}</span>${qmarkHtml(metric.desc)}</div>
          <div id="cmpHero"></div>
        </div>
        <div class="panel"><div class="chart-title">Values</div><div class="tbl-wrap" id="cmpTable"></div></div>
        ${batchSectionHtml()}`;
      const series = await compareSeriesFor(metric.key);
      Charts.lineChart(document.getElementById("cmpHero"), {
        series, logX: true, height: 440, seconds: secondsFor(metric),
        xLabel: "context (tokens ×1000)", yLabel: unitFor(metric),
      });
      renderCompareTable(series, { ...metric, seconds: secondsFor(metric) });
    }
    await renderBatchCharts();
  }

  // the »Relative to« picker lists the selected runs; it only shows for 2+
  function syncBaselinePicker() {
    const wrap = document.getElementById("cmpBaselineWrap");
    const select = document.getElementById("cmpBaseline");
    if (!wrap || !select) return;
    if (state.compare.baseline && !state.compare.slots.has(state.compare.baseline)) {
      state.compare.baseline = null;
    }
    const runs = [...state.compare.slots.entries()].sort((a, b) => a[1] - b[1]).map(([folder]) => {
      const summary = state.results.find(r => r.folder === folder);
      return { folder, name: summary ? seriesLabel(summary) : folder };
    });
    // .check sets display:flex, which would override the hidden attribute
    wrap.style.display = runs.length < 2 ? "none" : "";
    select.innerHTML = `<option value="">absolute values</option>` + runs.map(r =>
      `<option value="${esc(r.folder)}" ${r.folder === state.compare.baseline ? "selected" : ""}>${esc(r.name)}</option>`).join("");
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
        // engines write 0 when a batch metric is unavailable — not a data point
        points: e.detail.batch_data.map(b => [b.batch_size, b[def.field] > 0 ? b[def.field] : null]),
      })).filter(s => s.points.some(p => p[1] != null && isFinite(p[1])));
      if (!series.length) continue;
      const panel = document.createElement("div");
      panel.className = "panel";
      panel.innerHTML = `<div class="chart-title">${esc(def.title.replace(/ \[.+\]$/, ""))}
        <span class="unit">${esc(unit)}</span>${qmarkHtml(def.desc)}</div><div></div>`;
      wrap.appendChild(panel);
      Charts.lineChart(panel.lastElementChild, {
        series, logX: true, height: 240, seconds: def.seconds,
        xLabel: "batch size (parallel clients)", yLabel: unit,
        xTickFormat: v => String(v),
      });
    }
  }

  function renderCompareTable(series, metric) {
    const node = document.getElementById("cmpTable");
    if (!node) return;
    const xs = [...new Set(series.flatMap(s => s.points.map(p => p[0])))].sort((a, b) => a - b);
    // context degradation: value at the largest context as % of the smallest —
    // how much throughput a setup retains (or how much slower time metrics get)
    const firstX = xs[0];
    const lastX = xs[xs.length - 1];
    const deltaRow = xs.length < 2 ? "" : `<tr class="delta"
      title="Value at the largest context as a percentage of the smallest — retention for throughput metrics, slowdown for time metrics.">
      <td>${esc(lastX)}k / ${esc(firstX)}k</td>${series.map(s => {
        const at = x => s.points.find(pt => pt[0] === x && pt[1] != null && isFinite(pt[1]));
        const first = at(firstX);
        const last = at(lastX);
        return `<td>${first && last && first[1] > 0 ? ((last[1] / first[1]) * 100).toFixed(0) + "%" : "–"}</td>`;
      }).join("")}</tr>`;
    node.innerHTML = `<table class="tbl">
      <thead><tr><th>context</th>${series.map(s =>
        `<th><span class="legend-swatch" style="background:${s.color};display:inline-block;vertical-align:middle;margin-right:5px"></span>${esc(s.name)}</th>`).join("")}</tr></thead>
      <tbody>${xs.map(x => `<tr><td>${esc(x)}k</td>${series.map(s => {
        const p = s.points.find(pt => pt[0] === x && pt[1] != null);
        return `<td>${p ? esc(fmt(p[1], { seconds: metric.seconds })) : "–"}</td>`;
      }).join("")}</tr>`).join("")}${deltaRow}</tbody>
    </table>`;
  }

  CB.views.compare = renderCompareView;
  CB.onCompareResize = renderCompareCharts;
})();
