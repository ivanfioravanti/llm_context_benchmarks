/* Context Bench core: shared state, API/UI helpers, router, modal, theme.
   Everything is exposed on window.CB; view modules register themselves in
   CB.views and are dispatched by CB.render(). Load order:
   charts.js → export.js → core.js → report.js → view-*.js → app.js */

(function () {
  "use strict";

  const esc = Charts.escapeHtml;
  const fmt = Charts.fmtNum;

  const state = {
    meta: null,
    endpoints: [],
    results: [],
    details: {},           // folder -> full detail payload
    runsPollTimer: null,
    runLogs: {},           // run id -> { offset, text, notified }
    compare: {
      slots: new Map(),    // folder -> color slot index (0..7)
      metric: "generation_tps",
      grid: false,
      baseline: null,      // folder shown as 100% — null = absolute values
    },
    resultsFilter: "",
    runFormEngine: null,
    runFormEndpoint: "",
    runModelPicker: null,
  };

  const METRICS = [
    { key: "generation_tps", label: "Generation", unit: "tok/s",
      desc: "Decode speed: generated tokens per second of pure generation time. Higher is better." },
    { key: "prompt_tps", label: "Prompt processing", unit: "tok/s",
      desc: "Prefill speed: prompt tokens processed per second before generation starts. Higher is better." },
    { key: "time_to_first_token", label: "TTFT", unit: "s", seconds: true,
      desc: "Time to first token: how long from sending the request until the first generated token arrives — dominated by prompt processing. Lower is better." },
    { key: "time_per_output_token", label: "TPOT", unit: "s", seconds: true,
      desc: "Time per output token: average gap between two generated tokens once generation is running. Lower is better." },
    { key: "total_time", label: "Total time", unit: "s", seconds: true,
      desc: "Wall-clock time of the whole request: prompt processing plus generation." },
    { key: "generation_utf8_bytes_per_sec", label: "Gen bytes/s", unit: "B/s",
      desc: "Tokenizer-independent generation throughput: UTF-8 bytes of generated text per second — comparable across models with different tokenizers." },
    { key: "prompt_utf8_bytes_per_sec", label: "Prompt bytes/s", unit: "B/s",
      desc: "Tokenizer-independent prefill throughput: UTF-8 bytes of prompt text processed per second." },
    { key: "generation_chars_per_sec", label: "Gen chars/s", unit: "chars/s",
      desc: "Generated Unicode characters per second — like bytes/s but counting characters, so multi-byte scripts read naturally." },
    { key: "prompt_chars_per_sec", label: "Prompt chars/s", unit: "chars/s",
      desc: "Prompt Unicode characters processed per second — tokenizer-independent prefill throughput in characters." },
    { key: "peak_memory_gb", label: "Peak memory", unit: "GB",
      desc: "Peak accelerator/unified memory used during the run (reported by the framework, e.g. MLX)." },
    { key: "host_memory_gb", label: "Host RAM", unit: "GB",
      desc: "Resident memory of the server process, sampled while the benchmark ran." },
    { key: "kv_cache_gb", label: "KV cache", unit: "GB",
      desc: "Size of the key-value cache after the context was processed — grows with context length." },
    { key: "kv_cache_usage_perc", label: "KV usage", unit: "%",
      desc: "Share of the server's KV-cache pool in use (as reported by e.g. vLLM or llama.cpp)." },
  ];

  function metricsForKeys(keys) {
    return METRICS.filter(m => keys.has(m.key));
  }

  // hoverable »?« that explains a metric; keyboard-reachable via tabindex
  function qmarkHtml(desc) {
    return desc ? `<span class="qmark" tabindex="0" data-tip="${esc(desc)}">?</span>` : "";
  }

  // ----------------------------------------------------------------- utils

  async function api(path, opts) {
    const res = await fetch(path, opts && opts.body ? {
      method: opts.method || "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts.body),
    } : opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
      throw new Error(detail);
    }
    return res.json();
  }

  function toast(message, isError) {
    const stack = document.getElementById("toastStack");
    const node = document.createElement("div");
    node.className = "toast" + (isError ? " err" : "");
    node.textContent = message;
    stack.appendChild(node);
    setTimeout(() => node.remove(), isError ? 7000 : 3500);
  }

  function seriesColor(slot) {
    const styles = getComputedStyle(document.documentElement);
    return styles.getPropertyValue(`--s${(slot % 8) + 1}`).trim();
  }

  function ctxNum(ctx) {
    const v = parseFloat(String(ctx).replace(/k$/i, ""));
    return isFinite(v) ? v : 0;
  }

  // KV-cache-reuse runs (MLX --cached) carry warm re-prompt measurements in
  // cached_results; these map onto the cold metrics as dashed overlays.
  const CACHED_METRIC_FIELDS = {
    prompt_tps: "incremental_prompt_tps",
    generation_tps: "generation_tps",
    time_to_first_token: "time_to_first_token",
  };

  function cachedSeriesPoints(detail, metricKey) {
    const field = CACHED_METRIC_FIELDS[metricKey];
    const rows = detail.cached_results;
    if (!field || !rows || !rows.length) return null;
    const points = rows
      .map(r => [ctxNum(r.context_size), r[field]])
      .filter(p => p[1] != null && isFinite(p[1]) && p[1] > 0)
      .sort((a, b) => a[0] - b[0]);
    return points.length ? points : null;
  }

  function fmtDate(iso) {
    if (!iso) return "–";
    return iso.replace("T", " ").slice(0, 16);
  }

  function fmtDuration(seconds) {
    if (seconds == null) return "–";
    const s = Math.floor(seconds);
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m " + (s % 60) + "s";
    return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
  }

  function pageHead(eyebrow, title, sub, actionsHtml) {
    return `<div class="page-head">
      <div>
        <div class="eyebrow">${esc(eyebrow)}</div>
        <h1 class="page-title">${esc(title)}</h1>
        ${sub ? `<div class="page-sub">${sub}</div>` : ""}
      </div>
      ${actionsHtml || ""}
    </div>`;
  }

  function resultName(summary) {
    if (summary.label) return summary.label;
    return `${summary.engine}: ${summary.model}`;
  }

  // chart/series label: always carries the model so runs against the same
  // endpoint stay distinguishable in legends, tooltips and exports
  function seriesLabel(summary) {
    if (!summary.label) return `${summary.engine}: ${summary.model}`;
    const labelHasModel = summary.model &&
      summary.label.toLowerCase().includes(String(summary.model).toLowerCase());
    return labelHasModel ? summary.label : `${summary.label} · ${summary.model}`;
  }

  function resultSubtitle(summary) {
    const parts = [`${summary.engine} · ${summary.model}`];
    if (summary.machine) parts.push(summary.machine);
    return parts.join(" · ");
  }

  function matchesFilter(summary, filter) {
    if (!filter) return true;
    const haystack = [summary.folder, summary.engine, summary.model, summary.label,
      summary.machine, summary.endpoint].join(" ").toLowerCase();
    return filter.toLowerCase().split(/\s+/).every(term => haystack.includes(term));
  }

  function engineById(id) {
    return state.meta.engines.find(e => e.id === id);
  }

  // human-readable connection target of an endpoint ("" for local engines)
  function endpointTarget(ep) {
    if (ep.base_url) return ep.base_url;
    if (ep.host) return ep.host + (ep.port ? ":" + ep.port : "");
    return "";
  }

  // ------------------------------------------------------------- data cache

  async function ensureResults(force) {
    if (!force && state.results.length) return state.results;
    state.results = await api("/api/results");
    return state.results;
  }

  async function ensureDetail(folder) {
    if (state.details[folder]) return state.details[folder];
    state.details[folder] = await api(`/api/results/${encodeURIComponent(folder)}`);
    return state.details[folder];
  }

  // stable color slots for the comparison selection
  function toggleCompare(folder, on) {
    const slots = state.compare.slots;
    if (on) {
      if (slots.has(folder)) return true;
      if (slots.size >= 8) {
        toast("Comparison is limited to 8 runs — the palette stays readable that way.", true);
        return false;
      }
      const used = new Set(slots.values());
      let slot = 0;
      while (used.has(slot)) slot++;
      slots.set(folder, slot);
    } else {
      slots.delete(folder);
    }
    return true;
  }

  // ----------------------------------------------------------------- modal

  function openModal(html, opts) {
    const backdrop = document.getElementById("modalBackdrop");
    const modal = document.getElementById("modal");
    modal.className = "modal" + (opts && opts.narrow ? " narrow" : "");
    modal.innerHTML = html;
    backdrop.hidden = false;
    modal.querySelectorAll("[data-close]").forEach(b => b.addEventListener("click", closeModal));
    return modal;
  }
  function closeModal() {
    document.getElementById("modalBackdrop").hidden = true;
    document.getElementById("modal").innerHTML = "";
  }
  document.getElementById("modalBackdrop").addEventListener("click", e => {
    if (e.target.id === "modalBackdrop") closeModal();
  });
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

  // ----------------------------------------------------------------- theme

  function initTheme() {
    const saved = localStorage.getItem("cb-theme");
    const theme = saved || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    applyTheme(theme);
    document.getElementById("themeToggle").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      localStorage.setItem("cb-theme", next);
      applyTheme(next);
      render();
    });
  }
  function applyTheme(theme) {
    if (theme === "light") document.documentElement.dataset.theme = "light";
    else delete document.documentElement.dataset.theme;
  }

  // ----------------------------------------------------------------- router

  const VIEW_NAMES = ["run", "results", "compare", "endpoints", "tools"];
  const views = {};   // name -> render function, registered by the view modules

  function currentView() {
    const hash = location.hash.replace("#", "");
    return VIEW_NAMES.includes(hash) ? hash : "run";
  }

  function render() {
    const view = currentView();
    document.querySelectorAll("#nav a").forEach(a =>
      a.classList.toggle("active", a.dataset.view === view));
    for (const name of VIEW_NAMES) {
      document.getElementById("view-" + name).hidden = name !== view;
    }
    if (views[view]) views[view]();
  }
  window.addEventListener("hashchange", render);

  // charts pick up their width at render time — re-render on resize
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (currentView() === "compare" && CB.onCompareResize) CB.onCompareResize();
    }, 250);
  });

  // ------------------------------------------------------------ model picker
  // A text input that upgrades to a dropdown once the target server's
  // model list was fetched (OpenAI /v1/models or the local Ollama daemon).

  function attachModelPicker({ select, input, button, hint, getConnection, localEngine }) {
    function show(asSelect) {
      select.hidden = !asSelect;
      input.hidden = asSelect;
    }
    async function load(auto) {
      const connection = getConnection();
      if (!connection && !localEngine) {
        show(false);
        if (!auto) toast("No connection configured — enter the model manually.", true);
        return;
      }
      button.disabled = true;
      const original = button.textContent;
      button.textContent = "…";
      try {
        const res = connection
          ? await api("/api/models", { body: connection })
          : await api(`/api/cached-models?engine=${encodeURIComponent(localEngine)}`);
        if (res.models && res.models.length) {
          const current = (select.hidden ? input.value : select.value).trim();
          const models = res.models.slice();
          if (current && current !== "__custom__" && !models.includes(current)) models.unshift(current);
          select.innerHTML = models.map(m =>
            `<option value="${esc(m)}" ${m === current ? "selected" : ""}>${esc(m)}</option>`).join("") +
            `<option value="__custom__">✎ enter manually…</option>`;
          show(true);
          if (hint) hint.textContent = `${res.models.length} model${res.models.length === 1 ? "" : "s"} ` +
            (connection ? "found on the server." : "in the local HF cache.");
        } else {
          show(false);
          if (hint) hint.textContent = connection
            ? "No model list from the server — enter manually."
            : "No cached models found in HF_HOME — enter a repo id manually.";
          if (!auto && res.detail) toast("Model discovery failed: " + res.detail, true);
        }
      } catch (e) {
        show(false);
        if (hint) hint.textContent = "Model discovery failed — enter manually.";
      } finally {
        button.disabled = false;
        button.textContent = original;
      }
    }
    select.addEventListener("change", () => {
      if (select.value === "__custom__") {
        input.value = "";
        show(false);
        input.focus();
      }
    });
    button.addEventListener("click", () => load(false));
    return {
      load,
      value: () => (select.hidden ? input.value : select.value === "__custom__" ? "" : select.value).trim(),
      set: value => {
        if (!select.hidden && [...select.options].some(o => o.value === value)) select.value = value;
        else { input.value = value; show(false); }
      },
    };
  }

  function modelPickerHtml(idPrefix, currentValue, placeholder) {
    return `<div class="model-row">
      <select id="${idPrefix}Sel" hidden></select>
      <input type="text" id="${idPrefix}" value="${esc(currentValue || "")}" placeholder="${esc(placeholder || "")}">
      <button type="button" class="btn small" id="${idPrefix}Load" title="Load model list from the server">↻</button>
    </div>`;
  }

  window.CB = {
    state, METRICS, metricsForKeys, qmarkHtml,
    esc, fmt, api, toast,
    seriesColor, ctxNum, cachedSeriesPoints, fmtDate, fmtDuration,
    pageHead, resultName, resultSubtitle, seriesLabel, matchesFilter, engineById, endpointTarget,
    ensureResults, ensureDetail, toggleCompare,
    openModal, closeModal, initTheme,
    views, currentView, render,
    attachModelPicker, modelPickerHtml,
    onCompareResize: null,
  };
})();
