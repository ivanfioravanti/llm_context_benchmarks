/* Results view: browse saved runs from output/, select for comparison,
   rename/delete, and the per-run detail dialog with charts and export. */

(function () {
  "use strict";

  const {
    state, METRICS, esc, fmt, api, toast, seriesColor, ctxNum, cachedSeriesPoints, qmarkHtml, fmtDate,
    pageHead, resultName, resultSubtitle, seriesLabel, matchesFilter,
    ensureResults, ensureDetail, toggleCompare, openModal,
  } = CB;

  async function renderResultsView() {
    const root = document.getElementById("view-results");
    root.innerHTML = `${pageHead("Bench · Archive", "Results",
      `Every saved run from <span class="num">output/</span>. Select any set to compare.`)}
    <div class="results-toolbar">
      <input type="text" id="resFilter" placeholder="Filter by model, engine, label, machine…" value="${esc(state.resultsFilter)}">
      <button class="btn small" id="resRefresh">Refresh</button>
      <span style="flex:1"></span>
      <button class="btn primary" id="resCompare" disabled>Compare selected</button>
    </div>
    <div id="resList"><div class="empty">Loading…</div></div>`;

    document.getElementById("resFilter").addEventListener("input", e => {
      state.resultsFilter = e.target.value;
      renderResultRows();
    });
    document.getElementById("resRefresh").addEventListener("click", async () => {
      await ensureResults(true);
      renderResultRows();
    });
    document.getElementById("resCompare").addEventListener("click", () => {
      location.hash = "compare";
    });

    try { await ensureResults(); } catch (e) {
      document.getElementById("resList").innerHTML =
        `<div class="empty">Could not load results: ${esc(e.message)}</div>`;
      return;
    }
    renderResultRows();
  }

  function renderResultRows() {
    const list = document.getElementById("resList");
    if (!list) return;
    const rows = state.results.filter(r => !r.error && matchesFilter(r, state.resultsFilter));
    if (!state.results.length) {
      list.innerHTML = `<div class="empty"><strong>No saved results yet</strong>
        Run a benchmark — every run lands in output/ and shows up here.</div>`;
      return;
    }
    list.innerHTML = rows.map(r => {
      const checked = state.compare.slots.has(r.folder) ? "checked" : "";
      const ctxRange = r.contexts.length ? `${r.contexts[0]}–${r.contexts[r.contexts.length - 1]}` : "–";
      return `<div class="result-row" data-folder="${esc(r.folder)}">
        <input type="checkbox" data-act="select" ${checked} aria-label="Select for comparison">
        <div class="r-name">
          <div class="r-label">${esc(resultName(r))}</div>
          <div class="r-model">${esc(resultSubtitle(r))}</div>
        </div>
        <span class="tag">${esc(r.engine)}</span>
        <span class="r-machine">${esc(r.machine || "–")}</span>
        <span class="r-ctx">${esc(ctxRange)}</span>
        <div class="r-peak"><span class="v">${esc(fmt(r.peak_generation_tps))}</span> <span class="unit">t/s peak</span></div>
        <span class="r-spark">${Charts.sparkline(r.gen_series)}</span>
        <div class="r-actions">
          <span class="r-date">${esc(fmtDate(r.timestamp))}</span>
          <button class="btn small" data-act="detail">Details</button>
          <button class="btn small" data-act="rename" title="Rename label">✎</button>
          <button class="btn small danger" data-act="delete" title="Delete result">✕</button>
        </div>
      </div>`;
    }).join("") || `<div class="empty">No results match the filter.</div>`;

    list.querySelectorAll(".result-row").forEach(row => {
      const folder = row.dataset.folder;
      row.querySelector('[data-act="select"]').addEventListener("change", e => {
        toggleCompare(folder, e.target.checked);
        updateCompareButton();
      });
      row.querySelector('[data-act="detail"]').addEventListener("click", () => openDetail(folder));
      row.querySelector('[data-act="rename"]').addEventListener("click", () => renameResult(folder));
      row.querySelector('[data-act="delete"]').addEventListener("click", () => deleteResult(folder));
    });
    updateCompareButton();
  }

  function updateCompareButton() {
    const btn = document.getElementById("resCompare");
    if (!btn) return;
    const n = state.compare.slots.size;
    btn.disabled = n < 1;
    btn.textContent = n ? `Compare selected (${n})` : "Compare selected";
  }

  async function renameResult(folder) {
    const summary = state.results.find(r => r.folder === folder);
    const label = prompt("Label for this run (shown in charts):", summary ? summary.label : "");
    if (label === null) return;
    try {
      await api(`/api/results/${encodeURIComponent(folder)}`, { method: "PATCH", body: { label } });
      await ensureResults(true);
      renderResultRows();
    } catch (e) { toast(e.message, true); }
  }

  async function deleteResult(folder) {
    if (!confirm(`Delete result »${folder}«?\nThis removes the folder from output/ permanently.`)) return;
    try {
      await api(`/api/results/${encodeURIComponent(folder)}`, { method: "DELETE" });
      state.compare.slots.delete(folder);
      delete state.details[folder];
      await ensureResults(true);
      renderResultRows();
      toast("Result deleted.");
    } catch (e) { toast(e.message, true); }
  }

  // ------------------------------------------------------------- detail

  async function openDetail(folder) {
    let detail;
    try { detail = await ensureDetail(folder); } catch (e) { toast(e.message, true); return; }
    const summary = detail.summary;
    const rows = detail.results;
    const cols = ["context_size", "prompt_tokens", "prompt_tps", "generation_tokens", "generation_tps",
      "time_to_first_token", "time_per_output_token", "total_time",
      "peak_memory_gb", "host_memory_gb", "kv_cache_gb"]
      .filter(c => rows.some(r => r[c] != null));

    // charts: the four core panels plus memory panels when the run has them
    const chartKeys = ["generation_tps", "prompt_tps", "time_to_first_token", "total_time",
      "peak_memory_gb", "host_memory_gb", "kv_cache_gb"]
      .filter(key => rows.some(r => r[key] != null && isFinite(r[key])));
    const chartPanels = chartKeys.map(key => {
      const metric = METRICS.find(m => m.key === key) || { label: key, unit: "" };
      return `<div><div class="chart-title">${esc(metric.label)}
        ${metric.unit ? `<span class="unit">${esc(metric.unit)}</span>` : ""}${qmarkHtml(metric.desc)}</div>
        <div data-chart="${esc(key)}"></div></div>`;
    }).join("");

    const hw = detail.hardware_info || {};
    const modal = openModal(`
      <button class="btn small modal-close" data-close>Close</button>
      <span class="modal-close" style="margin-right:10px">${CB.report.exportGroupHtml()}</span>
      <div class="eyebrow">${esc(summary.engine)} · ${esc(fmtDate(summary.timestamp))}</div>
      <h2>${esc(resultName(summary))}</h2>
      <div class="page-sub">${esc(summary.model)}${summary.hardware ? " — " + esc(summary.hardware) : ""}</div>
      <div class="detail-charts">${chartPanels}</div>
      <div class="tbl-wrap"><table class="tbl">
        <thead><tr>${cols.map(c => `<th>${esc(c.replace(/_/g, " "))}</th>`).join("")}</tr></thead>
        <tbody>${rows.map(r => `<tr>${cols.map(c =>
          `<td>${c === "context_size" ? esc(r[c]) : esc(fmt(r[c], { seconds: /time|ttft|tpot/.test(c) }))}</td>`).join("")}</tr>`).join("")}
        </tbody></table></div>
      ${detail.cached_results && detail.cached_results.length ? (() => {
        const cachedRows = detail.cached_results;
        const cCols = [
          ["cached_tokens", "cached tokens", {}],
          ["delta_tokens", "new tokens", {}],
          ["incremental_prompt_tps", "incr. prompt t/s", {}],
          ["generation_tps", "gen t/s", {}],
          ["time_to_first_token", "ttft", { seconds: true }],
          ["kv_cache_gb", "kv cache GB", {}],
        ].filter(([key]) => cachedRows.some(r => r[key] != null));
        return `
        <h3 style="font-size:14px;margin:18px 0 4px">Cached re-prompt <span class="unit">KV-cache reuse</span></h3>
        <p class="hint" style="margin:0 0 8px">Warm runs on top of a stored KV cache — »incr. prompt«
          only counts the tokens added after the cached prefix.</p>
        <div class="tbl-wrap"><table class="tbl">
          <thead><tr><th>context</th>${cCols.map(([, label]) => `<th>${esc(label)}</th>`).join("")}</tr></thead>
          <tbody>${cachedRows.map(r => `<tr><td>${esc(r.context_size)}</td>
            ${cCols.map(([key, , o]) => `<td>${esc(fmt(r[key], o))}</td>`).join("")}
          </tr>`).join("")}</tbody>
        </table></div>`;
      })() : ""}
      ${detail.batch_data && detail.batch_data.length ? (() => {
        const batchRows = detail.batch_data;
        const bCols = [
          ["prompt_tps", "prompt t/s", {}],
          ["generation_tps", "e2e gen t/s", {}],
          ["decode_tps_total", "decode t/s", {}],
          ["decode_tps_per_client", "decode / client", {}],
          ["time_to_first_token", "ttft", { seconds: true }],
          ["time_per_output_token", "tpot", { seconds: true }],
          ["peak_memory_gb", "peak GB", {}],
          ["kv_cache_gb", "kv GB", {}],
          ["host_memory_gb", "host GB", {}],
        ].filter(([key]) => batchRows.some(b => b[key] != null && b[key] > 0));
        const hasDecode = bCols.some(([key]) => key === "decode_tps_total");
        return `
        <h3 style="font-size:14px;margin:18px 0 4px">Batch sweep <span class="unit">N parallel clients</span></h3>
        <p class="hint" style="margin:0 0 8px">»E2E gen« = generated tokens ÷ total wall time (prompt
          phase included)${hasDecode ? " — »decode« = pure generation rate, summed across clients" : ""}.</p>
        <div class="tbl-wrap"><table class="tbl">
          <thead><tr><th>batch</th>${bCols.map(([, label]) => `<th>${esc(label)}</th>`).join("")}</tr></thead>
          <tbody>${batchRows.map(b => `<tr><td>${esc(b.batch_size)}</td>
            ${bCols.map(([key, , o]) => `<td>${esc(fmt(b[key], o))}</td>`).join("")}
          </tr>`).join("")}</tbody>
        </table></div>`;
      })() : ""}
      ${detail.perplexity_data ? `<p style="font-size:12.5px;color:var(--ink-2)">
        Perplexity: <span class="num">${esc(fmt(detail.perplexity_data.perplexity))}</span>
        ± ${esc(fmt(detail.perplexity_data.std_error))} (${esc(detail.perplexity_data.dataset || "")})</p>` : ""}
      <details class="advanced"><summary>Hardware & files</summary><div>
        <dl class="kv-list">
          ${Object.entries(hw).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`).join("")}
        </dl>
        <p style="font-size:12px">${detail.files.map(f =>
          `<a href="/output/${encodeURIComponent(folder)}/${encodeURIComponent(f)}" target="_blank" style="color:var(--accent)">${esc(f)}</a>`
        ).join(" · ")}</p>
      </div></details>`);

    CB.report.wireExportGroup(modal,
      async () => [{ detail, slot: 0, name: seriesLabel(summary) }],
      "context-bench-run",
      () => seriesLabel(summary));

    const color = seriesColor(0);
    for (const key of chartKeys) {
      const node = modal.querySelector(`[data-chart="${key}"]`);
      const metric = METRICS.find(m => m.key === key) || {};
      const series = [{
        name: seriesLabel(summary), color,
        points: rows.map(r => [ctxNum(r.context_size), r[key]]),
      }];
      const cached = cachedSeriesPoints(detail, key);
      if (cached) series.push({ name: "cached (incremental)", color, dash: true, points: cached });
      Charts.lineChart(node, {
        series, logX: true, height: 190, seconds: metric.seconds,
        xLabel: "context", yLabel: metric.unit || "", legend: !!cached,
      });
    }
  }

  CB.views.results = renderResultsView;
})();
