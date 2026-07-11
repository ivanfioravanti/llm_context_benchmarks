/* Report/export pipeline shared by Compare and the result detail dialog:
   renders full-size charts offscreen and hands them to CBExport as
   ZIP (one overview PNG + CSV + TXT), interactive HTML report or PDF. */

(function () {
  "use strict";

  const { fmt, toast, seriesColor, ctxNum, metricsForKeys } = CB;

  function metricsForEntries(entries) {
    return metricsForKeys(new Set(entries.flatMap(e => e.detail.summary.columns)));
  }

  // Batch charts: prompt + end-to-end are always there; the pure decode rate
  // only when the engine recorded it (decode_tps_total from server usage).
  function batchChartDefs(entries) {
    const withBatch = entries.filter(e => e.detail.batch_data && e.detail.batch_data.length);
    if (!withBatch.length) return [];
    const defs = [
      { key: "batch_prompt", title: "Batch — prompt throughput [tok/s]", field: "prompt_tps" },
      { key: "batch_e2e", title: "Batch — end-to-end gen throughput [tok/s]", field: "generation_tps" },
    ];
    if (withBatch.some(e => e.detail.batch_data.some(b => b.decode_tps_total != null))) {
      defs.push({ key: "batch_decode", title: "Batch — decode throughput [tok/s]", field: "decode_tps_total" });
    }
    if (withBatch.some(e => e.detail.batch_data.some(b => b.host_memory_gb != null))) {
      defs.push({ key: "batch_host_mem", title: "Batch — host RAM [GB]", field: "host_memory_gb" });
    }
    return defs;
  }

  // Renders all charts (metric sweeps + batch) for the given runs offscreen.
  // opts.theme: null (current UI theme) | "light" — forced for print exports.
  // opts.raw: return class-based SVG markup (themable + interactive in the
  //           HTML report) instead of rasterized canvases.
  async function renderExportCharts(entries, metrics, opts) {
    const { theme = null, width = 1160, height = 480, batchHeight = 420, legend = true, raw = false } = opts || {};
    const stage = document.createElement("div");
    stage.style.cssText = `position:fixed;left:-12000px;top:0;width:${width}px`;
    if (theme === "light") stage.className = "force-light";
    document.body.appendChild(stage);
    const surface = () => getComputedStyle(stage).getPropertyValue("--surface").trim() || "#fff";
    const charts = [];

    const seriesFor = getPoints => entries
      .map(e => ({ name: e.name, color: seriesColor(e.slot || 0), points: getPoints(e) }))
      .filter(s => s.points.some(p => p[1] != null && isFinite(p[1])));

    const render = async (series, chartOpts, key, title) => {
      if (!series.length) return;
      stage.innerHTML = "";
      Charts.lineChart(stage, {
        series, logX: true, width, legend: false,
        svgTitle: raw ? undefined : title,
        svgLegend: raw ? false : legend,
        ...chartOpts,
      });
      const svg = stage.querySelector("svg");
      if (!svg) return;
      charts.push(raw
        ? { key, title, svg: svg.outerHTML }
        : { key, title, canvas: await CBExport.svgToCanvas(svg, 2, surface()) });
    };

    try {
      for (const metric of metrics) {
        await render(
          seriesFor(e => e.detail.results.map(r => [ctxNum(r.context_size), r[metric.key]])),
          { height, seconds: metric.seconds, xLabel: "context (tokens ×1000)", yLabel: metric.unit },
          metric.key,
          `${metric.label} across context size [${metric.unit}]`,
        );
      }
      for (const def of batchChartDefs(entries)) {
        await render(
          seriesFor(e => (e.detail.batch_data || []).map(b => [b.batch_size, b[def.field]])),
          { height: batchHeight, xLabel: "batch size (parallel clients)", yLabel: def.title.match(/\[(.+)\]/)[1], xTickFormat: v => String(v) },
          def.key,
          def.title,
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

  // format: "zip" | "html" | "pdf"
  async function exportRuns(entries, format, baseName, title) {
    const metrics = metricsForEntries(entries);
    const tables = CBExport.buildTables(entries, metrics, fmt);
    const stamp = new Date().toISOString().slice(0, 16).replace(/[T:]/g, "-");

    if (format === "zip") {
      // one compact dashboard PNG instead of a folder of single charts
      const charts = await renderExportCharts(entries, metrics,
        { width: 720, height: 330, batchHeight: 300, legend: false });
      const overview = CBExport.composeOverview({
        title,
        // entry names already carry the model, so the sub only adds context
        runs: entries.map(e => ({
          name: e.name,
          color: seriesColor(e.slot || 0),
          sub: [e.detail.summary.engine, e.detail.summary.machine].filter(Boolean).join(" · "),
        })),
        charts,
        tokens: themeTokens(),
      });
      const encoder = new TextEncoder();
      const files = [{ name: "overview.png", data: await CBExport.canvasToPngBytes(overview) }];
      files.push({ name: "results.csv", data: encoder.encode(CBExport.buildResultsCsv(entries)) });
      const batchCsv = CBExport.buildBatchCsv(entries);
      if (batchCsv) files.push({ name: "batch_results.csv", data: encoder.encode(batchCsv) });
      files.push({ name: "comparison.txt", data: encoder.encode(CBExport.buildComparisonTxt(entries, tables, title)) });
      CBExport.download(CBExport.makeZip(files), `${baseName}_${stamp}.zip`);
      return files.length + " files zipped";
    }

    if (format === "html") {
      // interactive SVG charts: theme with the report's toggle, points show
      // their values on hover/click
      const charts = await renderExportCharts(entries, metrics,
        { raw: true, width: 1060, height: 420, batchHeight: 380 });
      const legend = entries.map(e => ({ name: e.name, color: seriesColor(e.slot || 0) }));
      const html = CBExport.buildHtmlReport(title, entries, tables, charts, legend);
      CBExport.download(new Blob([html], { type: "text/html" }), `${baseName}_${stamp}.html`);
      return "HTML report saved";
    }

    // pdf — always the print-friendly light style
    const charts = await renderExportCharts(entries, metrics, { theme: "light" });
    const chartData = [];
    for (const c of charts) chartData.push({ title: c.title, jpeg: await CBExport.canvasToJpeg(c.canvas) });
    const pdf = CBExport.buildPdfReport(title, entries, tables, chartData);
    CBExport.download(pdf, `${baseName}_${stamp}.pdf`);
    return "PDF report saved";
  }

  function wireExportGroup(container, getEntries, baseName, getTitle) {
    container.querySelectorAll("[data-export]").forEach(btn => {
      btn.addEventListener("click", async () => {
        const group = btn.closest(".export-group");
        group.querySelectorAll(".btn").forEach(b => { b.disabled = true; });
        const original = btn.textContent;
        btn.textContent = "…";
        try {
          const entries = await getEntries();
          if (!entries.length) { toast("Nothing selected to export.", true); return; }
          const message = await exportRuns(entries, btn.dataset.export, baseName, getTitle(entries));
          toast("Export ready — " + message + ".");
        } catch (e) {
          toast("Export failed: " + e.message, true);
        } finally {
          btn.textContent = original;
          group.querySelectorAll(".btn").forEach(b => { b.disabled = false; });
        }
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

  CB.report = { metricsForEntries, batchChartDefs, wireExportGroup, exportGroupHtml };
})();
