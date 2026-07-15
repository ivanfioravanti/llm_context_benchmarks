/* Report/export pipeline shared by Compare and the result detail dialog:
   renders full-size charts offscreen and hands them to CBExport as
   ZIP (one overview PNG + CSV + TXT), interactive HTML report or PDF. */

(function () {
  "use strict";

  const { esc, fmt, toast, seriesColor, ctxNum, cachedSeriesPoints, metricsForKeys } = CB;

  function metricsForEntries(entries) {
    return metricsForKeys(new Set(entries.flatMap(e => e.detail.summary.columns)));
  }

  // every chart the given runs could produce — the export picker offers these
  function chartChoices(entries) {
    return [
      ...metricsForEntries(entries).map(m => ({ key: m.key, label: `${m.label} (${m.unit})`, desc: m.desc })),
      ...batchChartDefs(entries).map(d => ({ key: d.key, label: d.title, desc: d.desc })),
    ];
  }

  // table companions to the cached chart overlays (see core.js
  // CACHED_METRIC_FIELDS): what a KV-cache-reuse run gets tabulated as
  const CACHED_TABLE_DEFS = [
    { key: "incremental_prompt_tps", title: "Cached re-prompt — incremental prompt", unit: "tok/s",
      desc: "Prefill speed on top of a stored KV cache, counting only the tokens added after the cached prefix." },
    { key: "generation_tps", title: "Cached re-prompt — generation", unit: "tok/s",
      desc: "Decode speed of the warm run that reused the stored KV cache." },
    { key: "time_to_first_token", title: "Cached re-prompt — TTFT", unit: "s", seconds: true,
      desc: "Time to first token when the context prefix is already cached. Lower is better." },
  ];

  // deselected chart keys survive across exports (stored as the off-list so
  // metrics that appear later default to on)
  const CHART_PREF_KEY = "cb-export-charts-off";

  function excludedCharts() {
    try { return new Set(JSON.parse(localStorage.getItem(CHART_PREF_KEY) || "[]")); }
    catch (e) { return new Set(); }
  }

  // Batch charts: prompt + end-to-end are always there; the rest only when
  // some run in the set actually recorded the field (engines report 0 or
  // omit it entirely when a metric is unavailable).
  function batchChartDefs(entries) {
    const withBatch = entries.filter(e => e.detail.batch_data && e.detail.batch_data.length);
    if (!withBatch.length) return [];
    const has = field => withBatch.some(e => e.detail.batch_data.some(b => b[field] != null && b[field] > 0));
    const defs = [
      { key: "batch_prompt", title: "Batch — prompt throughput", unit: "tok/s", field: "prompt_tps",
        desc: "Prompt (prefill) throughput summed across N parallel clients. Higher is better." },
      { key: "batch_e2e", title: "Batch — end-to-end gen throughput", unit: "tok/s", field: "generation_tps",
        desc: "End-to-end generation throughput: generated tokens ÷ total wall time including the prompt phase, summed across clients." },
    ];
    if (has("decode_tps_total")) {
      defs.push({ key: "batch_decode", title: "Batch — decode throughput", unit: "tok/s", field: "decode_tps_total",
        desc: "Pure generation rate the server reported, summed across clients — prompt phase excluded." });
    }
    if (has("time_to_first_token")) {
      defs.push({ key: "batch_ttft", title: "Batch — TTFT", unit: "s", field: "time_to_first_token", seconds: true,
        desc: "Time to first token at this concurrency — the wait each client sees under load. Lower is better." });
    }
    if (has("time_per_output_token")) {
      defs.push({ key: "batch_tpot", title: "Batch — TPOT", unit: "s", field: "time_per_output_token", seconds: true,
        desc: "Average gap between two generated tokens at this concurrency. Lower is better." });
    }
    if (has("peak_memory_gb")) {
      defs.push({ key: "batch_peak_mem", title: "Batch — peak memory", unit: "GB", field: "peak_memory_gb",
        desc: "Peak accelerator/unified memory while serving N clients in parallel." });
    }
    if (has("kv_cache_gb")) {
      defs.push({ key: "batch_kv", title: "Batch — KV cache", unit: "GB", field: "kv_cache_gb",
        desc: "Key-value-cache size with N concurrent sequences — grows with concurrency." });
    }
    if (has("host_memory_gb")) {
      defs.push({ key: "batch_host_mem", title: "Batch — host RAM", unit: "GB", field: "host_memory_gb",
        desc: "Resident memory of the server process while serving N clients in parallel." });
    }
    return defs;
  }

  // Renders all charts (metric sweeps + batch) for the given runs offscreen.
  // opts.theme: null (current UI theme) | "light" — forced for print exports.
  // opts.raw: return class-based SVG markup (themable + interactive in the
  //           HTML report) instead of rasterized canvases.
  async function renderExportCharts(entries, metrics, opts) {
    const {
      theme = null, width = 1160, height = 480, batchHeight = 420,
      legend = true, raw = false, only = null, pointLabels = false,
    } = opts || {};
    const stage = document.createElement("div");
    stage.style.cssText = `position:fixed;left:-12000px;top:0;width:${width}px`;
    if (theme === "light") stage.className = "force-light";
    document.body.appendChild(stage);
    const surface = () => getComputedStyle(stage).getPropertyValue("--surface").trim() || "#fff";
    const charts = [];

    const seriesFor = getPoints => entries
      .map(e => ({ name: e.name, color: seriesColor(e.slot || 0), points: getPoints(e) }))
      .filter(s => s.points.some(p => p[1] != null && isFinite(p[1])));

    const render = async (series, chartOpts, key, title, desc) => {
      if (!series.length) return;
      stage.innerHTML = "";
      Charts.lineChart(stage, {
        series, logX: true, width, legend: false, pointLabels,
        svgTitle: raw ? undefined : title,
        svgLegend: raw ? false : legend,
        ...chartOpts,
      });
      const svg = stage.querySelector("svg");
      if (!svg) return;
      charts.push(raw
        ? { key, title, desc, svg: svg.outerHTML }
        : { key, title, desc, canvas: await CBExport.svgToCanvas(svg, 2, surface()) });
    };

    try {
      for (const metric of metrics) {
        if (only && !only.has(metric.key)) continue;
        const series = seriesFor(e => e.detail.results.map(r => [ctxNum(r.context_size), r[metric.key]]));
        for (const e of entries) {
          const cached = cachedSeriesPoints(e.detail, metric.key);
          if (cached) series.push({
            name: `${e.name} · cached`, color: seriesColor(e.slot || 0), dash: true, points: cached,
          });
        }
        await render(
          series,
          // no x-axis label: the »0.5k / 1k / …« ticks say it all
          { height, seconds: metric.seconds, yLabel: metric.unit },
          metric.key,
          `${metric.label} across context size [${metric.unit}]`,
          metric.desc,
        );
      }
      for (const def of batchChartDefs(entries)) {
        if (only && !only.has(def.key)) continue;
        await render(
          // engines write 0 when a batch metric is unavailable — not a data point
          seriesFor(e => (e.detail.batch_data || []).map(b => [b.batch_size, b[def.field] > 0 ? b[def.field] : null])),
          { height: batchHeight, seconds: def.seconds, xLabel: "batch size (parallel clients)", yLabel: def.unit, xTickFormat: v => String(v) },
          def.key,
          `${def.title} [${def.unit}]`,
          def.desc,
        );
      }
    } finally {
      stage.remove();
    }
    return charts;
  }

  function themeTokens() {
    const styles = getComputedStyle(document.documentElement);
    const tok = name => styles.getPropertyValue(name).trim();
    return {
      page: tok("--page") || "#fff",
      surface: tok("--surface") || "#fff",
      ink: tok("--ink") || "#111",
      ink2: tok("--ink-2") || "#555",
      muted: tok("--muted") || "#888",
      hairline: tok("--hairline") || "#ddd",
    };
  }

  // format: "zip" | "html" | "pdf" — `only` (Set of chart keys) limits which
  // charts are rendered; tables and CSVs always keep the full data
  async function exportRuns(entries, format, baseName, title, only) {
    const metrics = metricsForEntries(entries);
    const tables = CBExport.buildTables(entries, metrics, fmt, batchChartDefs(entries), CACHED_TABLE_DEFS);
    const stamp = new Date().toISOString().slice(0, 16).replace(/[T:]/g, "-");

    if (format === "zip") {
      // one compact dashboard PNG instead of a folder of single charts;
      // value labels at the points because a PNG has no hover tooltip
      const charts = await renderExportCharts(entries, metrics,
        { width: 720, height: 330, batchHeight: 300, legend: false, only, pointLabels: true });
      const encoder = new TextEncoder();
      const files = [];
      if (charts.length) {
        const overview = CBExport.composeOverview({
          title,
          // entry names already carry the model, so the sub only adds context
          // no local machine info here: for endpoint runs the inference ran
          // elsewhere — the endpoint name is the machine that matters
          runs: entries.map(e => ({
            name: e.name,
            color: seriesColor(e.slot || 0),
            sub: [e.detail.summary.engine, e.detail.summary.endpoint].filter(Boolean).join(" · "),
          })),
          charts,
          tokens: themeTokens(),
        });
        files.push({ name: "overview.png", data: await CBExport.canvasToPngBytes(overview) });
      }
      files.push({ name: "results.csv", data: encoder.encode(CBExport.buildResultsCsv(entries)) });
      const batchCsv = CBExport.buildBatchCsv(entries);
      if (batchCsv) files.push({ name: "batch_results.csv", data: encoder.encode(batchCsv) });
      const cachedCsv = CBExport.buildCachedCsv(entries);
      if (cachedCsv) files.push({ name: "cached_results.csv", data: encoder.encode(cachedCsv) });
      files.push({ name: "comparison.txt", data: encoder.encode(CBExport.buildComparisonTxt(entries, tables, title)) });
      CBExport.download(CBExport.makeZip(files), `${baseName}_${stamp}.zip`);
      return files.length + " files zipped";
    }

    if (format === "html") {
      // interactive SVG charts: theme with the report's toggle, points show
      // their values on hover/click
      const charts = await renderExportCharts(entries, metrics,
        { raw: true, width: 1060, height: 420, batchHeight: 380, only });
      const legend = entries.map(e => ({ name: e.name, color: seriesColor(e.slot || 0) }));
      const html = CBExport.buildHtmlReport(title, entries, tables, charts, legend);
      CBExport.download(new Blob([html], { type: "text/html" }), `${baseName}_${stamp}.html`);
      return "HTML report saved";
    }

    // pdf — always the print-friendly light style; value labels because
    // print has no hover tooltip either
    const charts = await renderExportCharts(entries, metrics, { theme: "light", only, pointLabels: true });
    const chartData = [];
    for (const c of charts) chartData.push({ title: c.title, jpeg: await CBExport.canvasToJpeg(c.canvas) });
    const pdf = CBExport.buildPdfReport(title, entries, tables, chartData);
    CBExport.download(pdf, `${baseName}_${stamp}.pdf`);
    return "PDF report saved";
  }

  async function runExport(group, format, entries, baseName, getTitle, only) {
    const btn = group.querySelector(`[data-export="${format}"]`);
    group.querySelectorAll(".btn").forEach(b => { b.disabled = true; });
    const original = btn.textContent;
    btn.textContent = "…";
    try {
      const message = await exportRuns(entries, format, baseName, getTitle(entries), only);
      toast("Export ready — " + message + ".");
    } catch (e) {
      toast("Export failed: " + e.message, true);
    } finally {
      btn.textContent = original;
      group.querySelectorAll(".btn").forEach(b => { b.disabled = false; });
    }
  }

  // small popover under the clicked export button: pick which charts go in
  function openChartPicker(group, format, entries, baseName, getTitle) {
    const choices = chartChoices(entries);
    const off = excludedCharts();
    const pop = document.createElement("div");
    pop.className = "export-pop";
    pop.dataset.format = format;
    pop.innerHTML = `
      <div class="pop-title">Charts in the ${esc(format.toUpperCase())} export</div>
      <div class="pop-list">${choices.map(c => `
        <label class="check" title="${esc(c.desc || "")}"><input type="checkbox" value="${esc(c.key)}" ${off.has(c.key) ? "" : "checked"}>
          ${esc(c.label)}</label>`).join("")}
      </div>
      <div class="pop-actions">
        <button class="btn small" data-pick="all">All</button>
        <button class="btn small" data-pick="none">None</button>
        <span style="flex:1"></span>
        <button class="btn small primary" data-go>Export</button>
      </div>`;
    group.appendChild(pop);
    const boxes = () => [...pop.querySelectorAll("input[type=checkbox]")];
    pop.querySelector('[data-pick="all"]').addEventListener("click", () => boxes().forEach(b => { b.checked = true; }));
    pop.querySelector('[data-pick="none"]').addEventListener("click", () => boxes().forEach(b => { b.checked = false; }));
    const onOutside = e => { if (!pop.contains(e.target) && !group.contains(e.target)) pop.close(); };
    pop.close = () => { pop.remove(); document.removeEventListener("mousedown", onOutside); };
    document.addEventListener("mousedown", onOutside);
    pop.querySelector("[data-go]").addEventListener("click", async () => {
      const selected = new Set(boxes().filter(b => b.checked).map(b => b.value));
      // persist per chart key, keeping off-entries of charts not offered here
      choices.forEach(c => (selected.has(c.key) ? off.delete(c.key) : off.add(c.key)));
      localStorage.setItem(CHART_PREF_KEY, JSON.stringify([...off]));
      pop.close();
      await runExport(group, format, entries, baseName, getTitle, selected);
    });
  }

  function wireExportGroup(container, getEntries, baseName, getTitle) {
    container.querySelectorAll("[data-export]").forEach(btn => {
      btn.addEventListener("click", async () => {
        const group = btn.closest(".export-group");
        const open = group.querySelector(".export-pop");
        if (open) {
          const sameFormat = open.dataset.format === btn.dataset.export;
          open.close();
          if (sameFormat) return; // second click on the same button toggles the picker away
        }
        let entries;
        try { entries = await getEntries(); } catch (e) { toast(e.message, true); return; }
        if (!entries.length) { toast("Nothing selected to export.", true); return; }
        openChartPicker(group, btn.dataset.export, entries, baseName, getTitle);
      });
    });
  }

  function exportGroupHtml() {
    return `<span class="export-group">
      <span class="export-label">Export</span>
      <button class="btn small" data-export="zip" title="Overview PNG + CSV + plain text, zipped">ZIP</button>
      <button class="btn small" data-export="html" title="Self-contained HTML report (dark/light toggle)">HTML</button>
      <button class="btn small" data-export="pdf" title="PDF report">PDF</button>
    </span>`;
  }

  CB.report = { batchChartDefs, wireExportGroup, exportGroupHtml };
})();
