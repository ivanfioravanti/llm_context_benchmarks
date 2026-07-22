/* Run view: endpoint-driven benchmark launcher plus the live run list
   (status LEDs, live readout, context/batch progress lanes, streaming log). */

(function () {
  "use strict";

  const {
    state, esc, fmt, api, toast, fmtDuration, pageHead, engineById, endpointTarget,
    attachModelPicker, modelPickerHtml, currentView,
  } = CB;

  // connection values for the run: from the endpoint, else the draft / DOM
  function readConnection(ep) {
    const draft = state.runForm;
    if (ep) {
      return {
        base_url: ep.base_url || "",
        api_key: ep.api_key || "",
        host: ep.host || "",
        port: ep.port || "",
      };
    }
    const field = id => document.getElementById(id);
    return {
      base_url: field("rfBaseUrl") ? field("rfBaseUrl").value.trim() : (draft.baseUrl || ""),
      api_key: field("rfApiKey") ? field("rfApiKey").value : (draft.apiKey || ""),
      host: field("rfHost") ? field("rfHost").value.trim() : (draft.host || "localhost"),
      port: field("rfPort") ? field("rfPort").value : (draft.port || "8080"),
    };
  }

  // Pull the live form into state.runForm before a re-render wipes the DOM.
  function captureRunForm() {
    if (!document.getElementById("rfMaxTokens")) return;
    const draft = state.runForm;
    draft.endpoint = document.getElementById("rfEndpoint").value;
    const engineSel = document.getElementById("rfEngine");
    if (engineSel) draft.engine = engineSel.value;
    draft.model = state.runModelPicker
      ? state.runModelPicker.value()
      : (document.getElementById("rfModel") || {}).value || "";
    draft.label = document.getElementById("rfLabel").value;
    draft.contexts = [...document.querySelectorAll("#rfContexts input:checked")].map(i => i.value);
    draft.maxTokens = document.getElementById("rfMaxTokens").value;
    draft.runs = document.getElementById("rfRuns").value;
    draft.timeout = document.getElementById("rfTimeout").value;
    draft.saveResponses = document.getElementById("rfSaveResponses").checked;
    const cold = document.getElementById("rfColdPrefill");
    if (cold) draft.coldPrefill = cold.checked;
    const options = {};
    document.querySelectorAll("#view-run [data-optkey]").forEach(node => {
      options[node.dataset.optkey] = node.type === "checkbox" ? node.checked : node.value;
    });
    draft.options = options;
    const extra = document.getElementById("rfExtraArgs");
    if (extra) draft.extraArgs = extra.value;
    const baseUrl = document.getElementById("rfBaseUrl");
    if (baseUrl) draft.baseUrl = baseUrl.value;
    const apiKey = document.getElementById("rfApiKey");
    if (apiKey) draft.apiKey = apiKey.value;
    const host = document.getElementById("rfHost");
    if (host) draft.host = host.value;
    const port = document.getElementById("rfPort");
    if (port) draft.port = port.value;
  }

  function optionValue(opt, draftOpts) {
    if (draftOpts && Object.prototype.hasOwnProperty.call(draftOpts, opt.key)) {
      return draftOpts[opt.key];
    }
    return opt.default;
  }

  function renderRunView(opts) {
    const keepDraft = opts && opts.keepDraft;
    if (!keepDraft) captureRunForm();

    const root = document.getElementById("view-run");
    const engines = state.meta.engines;
    const draft = state.runForm;

    // endpoint-first: the endpoint carries engine + connection + default model
    const ep = state.endpoints.find(x => x.id === draft.endpoint) || null;
    if (ep && ep.engine && engineById(ep.engine)) draft.engine = ep.engine;
    if (!draft.engine || !engineById(draft.engine)) {
      draft.engine = (engines.find(e => e.available) || engines[0]).id;
    }
    const engine = engineById(draft.engine);
    const engineLocked = !!(ep && ep.engine);
    const manual = !ep;
    const defaultCtx = engine.default_contexts.split(",").map(s => s.trim());
    const selectedCtx = draft.contexts || defaultCtx;
    const target = ep ? (endpointTarget(ep) || "local") : "";
    const modelValue = draft.model || (ep ? ep.model : "") || "";
    const labelValue = draft.label || (ep ? ep.name : "") || "";

    root.innerHTML = `
      ${pageHead("Bench · Launch", "Run sweep",
        "Prompt processing & generation speed across context sizes.")}
      <div class="grid-2">
        <div class="panel">
          <div class="form-row cols-2">
            <div class="field">
              <label for="rfEndpoint">Endpoint</label>
              <select id="rfEndpoint">
                <option value="">— manual / local —</option>
                ${state.endpoints.map(x => `<option value="${esc(x.id)}" ${ep && ep.id === x.id ? "selected" : ""}>${esc(x.name)}</option>`).join("")}
              </select>
              <span class="hint">${state.endpoints.length
                ? "Engine & connection come from the endpoint; its name labels the run."
                : `No endpoints yet — add them under <a href="#endpoints" style="color:var(--accent)">Endpoints</a>.`}</span>
            </div>
            ${engineLocked ? `
            <div class="field">
              <label>Target</label>
              <div class="target-chip">
                <span class="tag">${esc(engine.label)}</span>
                <span class="mono-url">${esc(target)}</span>
              </div>
              <span class="hint">${esc(engine.description)}</span>
            </div>` : `
            <div class="field">
              <label for="rfEngine">Engine</label>
              <select id="rfEngine">
                ${engines.map(e => `<option value="${esc(e.id)}" ${e.id === engine.id ? "selected" : ""}
                  ${e.available ? "" : "disabled"}>${esc(e.label)}${e.available ? "" : " — needs Apple Silicon"}</option>`).join("")}
              </select>
              <span class="hint">${esc(engine.description)}</span>
            </div>`}
          </div>
          <div class="form-row cols-2">
            <div class="field">
              <label for="rfModel">Model ${engine.model === "auto" ? "(optional, auto-detected)" : ""}</label>
              ${modelPickerHtml("rfModel", modelValue, engine.example)}
              <span class="hint" id="rfModelHint"></span>
            </div>
            <div class="field">
              <label for="rfLabel">Run label</label>
              <input type="text" id="rfLabel" value="${esc(labelValue)}" placeholder="e.g. M3 Ultra · speculative on">
              <span class="hint">Shown in results & comparison charts.</span>
            </div>
          </div>
          ${manual ? renderConnectionFields(engine, draft) : ""}
          ${!engine.available ? `<p class="hint" style="color:var(--critical)">
            This engine needs Apple Silicon (MLX) and is disabled on this machine.</p>` : ""}
          <div class="field" style="margin-bottom:12px">
            <label>Context sizes <span class="unit">tokens ×1000</span></label>
            <div class="ctx-chips" id="rfContexts">
              ${state.meta.context_files.map(c => {
                const size = c.name.replace(/k$/, "");
                return `<label class="ctx-chip"><input type="checkbox" value="${esc(size)}"
                  ${selectedCtx.includes(String(size)) ? "checked" : ""}>
                  <span>${esc(c.name)}</span></label>`;
              }).join("")}
            </div>
          </div>
          <div class="form-row cols-4">
            <div class="field"><label for="rfMaxTokens">Max tokens</label>
              <input type="number" id="rfMaxTokens" value="${esc(draft.maxTokens)}" min="1"></div>
            <div class="field"><label for="rfRuns">Runs / context</label>
              <input type="number" id="rfRuns" value="${esc(draft.runs)}" min="1"></div>
            <div class="field"><label for="rfTimeout">Timeout <span class="unit">s</span></label>
              <input type="number" id="rfTimeout" value="${esc(draft.timeout)}" min="10"></div>
            <div class="field field-check">
              <label class="check"><input type="checkbox" id="rfSaveResponses"
                ${draft.saveResponses ? "checked" : ""}> Save responses</label></div>
          </div>
          ${engine.cold_prefill ? `<label class="check" style="margin-bottom:4px">
            <input type="checkbox" id="rfColdPrefill" ${draft.coldPrefill ? "checked" : ""}>
            Cold prefill (clear cache between contexts)</label>` : ""}
          ${renderEngineOptions(engine, draft.options)}
          <details class="advanced">
            <summary>Extra CLI arguments</summary>
            <div class="field"><input type="text" id="rfExtraArgs" value="${esc(draft.extraArgs)}"
              placeholder="--some-flag value"></div>
          </details>
          <div class="btn-row" style="margin-top:16px">
            <button class="btn primary" id="rfStart" ${engine.available ? "" : "disabled"}>Start benchmark</button>
          </div>
        </div>
        <div>
          <div class="panel-head" style="margin-bottom:8px">
            <span class="eyebrow">Live · Active & recent runs</span>
          </div>
          <div id="runList" data-kind="benchmark"></div>
        </div>
      </div>`;

    document.getElementById("rfEndpoint").addEventListener("change", e => {
      captureRunForm();
      draft.endpoint = e.target.value;
      const next = state.endpoints.find(x => x.id === draft.endpoint) || null;
      if (next) {
        if (next.engine && engineById(next.engine)) draft.engine = next.engine;
        draft.model = next.model || "";
        draft.label = next.name || "";
      }
      renderRunView({ keepDraft: true });
    });
    const engineSelect = document.getElementById("rfEngine");
    if (engineSelect) engineSelect.addEventListener("change", e => {
      captureRunForm();
      draft.engine = e.target.value;
      // engine-specific defaults — keep shared knobs (max tokens, contexts, …)
      draft.options = null;
      renderRunView({ keepDraft: true });
    });

    // like readConnection, but with the engine defaults filled in so model
    // discovery can hit the server before the user typed anything
    const runConnection = () => {
      // Local engines do not have an HTTP endpoint.  readConnection() still
      // carries the form's default host/port, which would otherwise make the
      // model picker probe localhost:8080 instead of using local discovery.
      if (!engine.connection) return null;
      const connection = { engine: engine.id, ...readConnection(ep) };
      if (engine.connection === "base_url") {
        connection.base_url = connection.base_url || engine.default_base_url || "";
        connection.host = "";
        connection.port = "";
      } else if (engine.connection === "hostport") {
        connection.base_url = "";
        connection.host = connection.host || "localhost";
      }
      const ollama = engine.id === "ollama-api" || engine.id === "ollama-cli";
      if (!ollama && !connection.base_url && !connection.host) return null;
      return connection;
    };
    // MLX has no server to query — its models live in the local HF cache.
    const localEngine = engine.id === "mlx" ? "mlx" : null;
    state.runModelPicker = attachModelPicker({
      select: document.getElementById("rfModelSel"),
      input: document.getElementById("rfModel"),
      button: document.getElementById("rfModelLoad"),
      hint: document.getElementById("rfModelHint"),
      getConnection: runConnection,
      localEngine,
    });
    for (const id of ["rfBaseUrl", "rfHost", "rfPort", "rfApiKey"]) {
      const node = document.getElementById(id);
      if (node) node.addEventListener("change", () => state.runModelPicker.load(true));
    }
    if (runConnection() || localEngine) {
      state.runModelPicker.load(true).then(() => {
        if (draft.model) state.runModelPicker.set(draft.model);
      });
    }
    document.getElementById("rfStart").addEventListener("click", startRun);

    renderRunList();
    startRunsPolling();
  }

  function renderConnectionFields(engine, draft) {
    if (engine.connection === "base_url") {
      return `<div class="form-row cols-2">
        <div class="field"><label for="rfBaseUrl">Base URL</label>
          <input type="text" id="rfBaseUrl" value="${esc(draft.baseUrl || "")}"
            placeholder="${esc(engine.default_base_url || "http://host:port/v1")}"></div>
        <div class="field"><label for="rfApiKey">API key (optional)</label>
          <input type="password" id="rfApiKey" value="${esc(draft.apiKey || "")}" autocomplete="off"></div>
      </div>`;
    }
    if (engine.connection === "hostport") {
      return `<div class="form-row cols-2">
        <div class="field"><label for="rfHost">Host</label>
          <input type="text" id="rfHost" value="${esc(draft.host || "localhost")}"></div>
        <div class="field"><label for="rfPort">Port</label>
          <input type="number" id="rfPort" value="${esc(draft.port || "8080")}"></div>
      </div>`;
    }
    return "";
  }

  function renderEngineOptions(engine, draftOpts) {
    if (!engine.options.length) return "";
    const fields = engine.options.map(opt => {
      const id = "opt_" + opt.key;
      const saved = optionValue(opt, draftOpts);
      if (opt.type === "flag" || opt.type === "invflag" || opt.type === "optbool") {
        return `<label class="check"><input type="checkbox" id="${id}" data-optkey="${esc(opt.key)}"
          ${saved ? "checked" : ""}> ${esc(opt.label)}</label>`;
      }
      if (opt.type === "choice") {
        return `<div class="field"><label for="${id}">${esc(opt.label)}</label>
          <select id="${id}" data-optkey="${esc(opt.key)}">
            ${opt.choices.map(c => `<option value="${esc(c)}" ${c === saved ? "selected" : ""}>${esc(c || "default")}</option>`).join("")}
          </select></div>`;
      }
      const value = saved == null ? "" : saved;
      const type = (opt.type === "int" || opt.type === "float") ? "number" : "text";
      const step = opt.type === "float" ? ` step="any"` : "";
      return `<div class="field"><label for="${id}">${esc(opt.label)}${opt.required ? " *" : ""}</label>
        <input type="${type}"${step} id="${id}" data-optkey="${esc(opt.key)}" value="${esc(value)}"
          ${opt.help ? `title="${esc(opt.help)}"` : ""}></div>`;
    });
    return `<details class="advanced" ${engine.options.some(o => o.required) ? "open" : ""}>
      <summary>Engine options</summary>
      <div><div class="form-row cols-3">${fields.join("")}</div></div>
    </details>`;
  }

  async function startRun() {
    captureRunForm();
    const draft = state.runForm;
    const engine = engineById(draft.engine);
    const ep = state.endpoints.find(x => x.id === draft.endpoint) || null;
    const contexts = draft.contexts || [];
    if (!contexts.length) { toast("Select at least one context size.", true); return; }

    const options = {};
    const draftOpts = draft.options || {};
    for (const [key, value] of Object.entries(draftOpts)) {
      if (typeof value === "boolean") options[key] = value;
      else if (value !== "") options[key] = value;
    }

    const connection = readConnection(ep);
    const payload = {
      engine: engine.id,
      model: draft.model,
      label: draft.label,
      endpoint_id: ep ? ep.id : null,
      contexts: contexts.join(","),
      max_tokens: draft.maxTokens,
      runs: draft.runs,
      timeout: draft.timeout,
      save_responses: draft.saveResponses,
      cold_prefill: engine.cold_prefill ? draft.coldPrefill : undefined,
      connection,
      options,
      extra_args: draft.extraArgs,
    };
    try {
      const run = await api("/api/runs", { body: payload });
      toast(`Benchmark started: ${run.engine} · ${run.model}`);
      renderRunList();
      startRunsPolling();
    } catch (err) {
      toast("Start failed: " + err.message, true);
    }
  }

  // ------------------------------------------------- run list & polling

  async function renderRunList() {
    const container = document.querySelector(".view:not([hidden]) #runList");
    if (!container) return;
    let runs;
    try { runs = await api("/api/runs"); } catch (e) { return; }
    const kind = container.dataset.kind;
    if (kind) runs = runs.filter(r => r.kind === kind);

    if (!runs.length) {
      container.innerHTML = kind === "ctxgen"
        ? ""
        : `<div class="empty"><strong>No runs yet this session</strong>
            Configure a sweep on the left and hit »Start benchmark«.</div>`;
      return;
    }

    const anyActive = runs.some(r => r.status === "running" || r.status === "starting");
    // iterate oldest-first so prepend leaves the newest run on top
    for (const run of runs.slice().reverse()) {
      let card = container.querySelector(`[data-run="${run.id}"]`);
      if (!card) card = createRunCard(container, run);
      updateRunCard(card, run);
    }
    if (anyActive) startRunsPolling(); else stopRunsPolling();
  }

  function runProgressSummary(run, statusText) {
    const parts = [statusText];
    if (run.contexts && run.contexts.length) {
      const done = Math.min(run.contexts_done || 0, run.contexts.length);
      let bit = `context ${done}/${run.contexts.length}`;
      if (run.phase === "context" && run.current_context) bit += ` (${run.current_context})`;
      parts.push(bit);
    }
    if (run.batch_sizes && run.batch_sizes.length) {
      const done = Math.min(run.batch_sizes_done || 0, run.batch_sizes.length);
      let bit = `batch ${done}/${run.batch_sizes.length}`;
      if (run.phase === "batch" && run.current_batch_size != null) bit += ` (${run.current_batch_size}×)`;
      parts.push(bit);
    }
    if (run.error) parts.push(run.error);
    return parts.join(" · ");
  }

  function setLaneHtml(node, signature, html, hidden) {
    if (hidden) {
      node.hidden = true;
      if (node.dataset.sig) {
        node.dataset.sig = "";
        node.innerHTML = "";
      }
      return;
    }
    node.hidden = false;
    if (node.dataset.sig === signature) return;
    node.dataset.sig = signature;
    node.innerHTML = html;
  }

  function createRunCard(container, run) {
    const card = document.createElement("div");
    card.className = "run-card";
    card.dataset.run = run.id;
    container.prepend(card);
    card.innerHTML = `
      <div class="run-card-head">
        <span class="led"></span>
        <div>
          <div class="run-card-title"></div>
          <div class="run-card-meta"></div>
        </div>
        <div class="run-card-spacer"></div>
        <button class="btn small" data-act="toggle-log">Log</button>
        <button class="btn small danger" data-act="stop">Stop</button>
        <button class="btn small danger" data-act="delete" hidden>Delete</button>
      </div>
      <div class="readout">
        <div class="readout-cell"><span class="eyebrow" data-f="prompt-label">Prompt</span>
          <span class="readout-value" data-f="prompt">–<span class="unit"> t/s</span></span></div>
        <div class="readout-cell"><span class="eyebrow" data-f="gen-label">Generation</span>
          <span class="readout-value" data-f="gen">–<span class="unit"> t/s</span></span></div>
        <div class="readout-cell"><span class="eyebrow" data-f="ttft-label">TTFT</span>
          <span class="readout-value" data-f="ttft">–<span class="unit"> s</span></span></div>
        <div class="readout-cell"><span class="eyebrow">Elapsed</span>
          <span class="readout-value" data-f="elapsed">–</span></div>
      </div>
      <div class="ctx-lane" data-f="context-lane"></div>
      <div class="ctx-lane" data-f="batch-lane" hidden></div>
      <pre class="console" data-f="log" hidden></pre>`;
    card.querySelector('[data-act="stop"]').addEventListener("click", async () => {
      try { await api(`/api/runs/${run.id}/stop`, { method: "POST" }); } catch (e) { toast(e.message, true); }
    });
    card.querySelector('[data-act="delete"]').addEventListener("click", async () => {
      if (!confirm("Remove this run from the list?")) return;
      try {
        await api(`/api/runs/${run.id}`, { method: "DELETE" });
        card.remove();
      } catch (e) { toast(e.message, true); }
    });
    card.querySelector('[data-act="toggle-log"]').addEventListener("click", () => {
      const log = card.querySelector('[data-f="log"]');
      log.hidden = !log.hidden;
      if (!log.hidden) log.scrollTop = log.scrollHeight;
    });
    const cached = state.runLogs[run.id];
    if (cached && cached.text) card.querySelector('[data-f="log"]').textContent = cached.text;
    return card;
  }

  async function updateRunCard(card, run) {
    const dot = card.querySelector(".led");
    dot.className = "led " + run.status;
    const active = run.status === "running" || run.status === "starting";
    card.querySelector('[data-act="stop"]').hidden = !active;
    card.querySelector('[data-act="delete"]').hidden = active;
    card.querySelector(".run-card-title").textContent =
      (run.label ? run.label + " — " : "") + run.engine + " · " + run.model;
    const statusText = { starting: "starting", running: "running", done: "completed",
      failed: "failed (exit " + run.returncode + ")", stopped: "stopped" }[run.status] || run.status;
    const meta = card.querySelector(".run-card-meta");
    meta.textContent = runProgressSummary(run, statusText);
    meta.setAttribute("aria-live", "polite");
    card.querySelector('[data-f="elapsed"]').textContent = fmtDuration(run.elapsed);

    const liveSource = run.live.source || run.phase;
    const batchTag = liveSource === "batch" && (run.live.source_batch_size ?? run.current_batch_size) != null
      ? ` · ${(run.live.source_batch_size ?? run.current_batch_size)}×`
      : (liveSource === "batch" ? " · batch" : "");
    card.querySelector('[data-f="prompt-label"]').textContent = "Prompt" + batchTag;
    card.querySelector('[data-f="gen-label"]').textContent = "Generation" + batchTag;
    card.querySelector('[data-f="ttft-label"]').textContent = "TTFT" + batchTag;

    if (run.live.prompt_tps != null)
      card.querySelector('[data-f="prompt"]').innerHTML = `${esc(fmt(run.live.prompt_tps))}<span class="unit"> t/s</span>`;
    if (run.live.generation_tps != null)
      card.querySelector('[data-f="gen"]').innerHTML = `${esc(fmt(run.live.generation_tps))}<span class="unit"> t/s</span>`;
    if (run.live.ttft != null)
      card.querySelector('[data-f="ttft"]').innerHTML = `${esc(run.live.ttft.toFixed(2))}<span class="unit"> s</span>`;

    const contextLane = card.querySelector('[data-f="context-lane"]');
    if (run.contexts.length) {
      const ctxSig = [run.status, run.phase, run.contexts_done, run.current_context, run.contexts.join(",")].join("|");
      const ctxHtml = `<span class="progress-label">Context</span>` + run.contexts.map((ctx, i) => {
        const name = /k$/.test(ctx) ? ctx : ctx + "k";
        let cls = "";
        if (run.status === "done" || i < run.contexts_done) cls = "done";
        else if (run.phase === "context" && run.current_context && run.current_context.replace(/k$/, "") === String(ctx).replace(/k$/, "")) cls = "active";
        return `<span class="ctx-step ${cls}">${esc(name)}</span>`;
      }).join("");
      setLaneHtml(contextLane, ctxSig, ctxHtml, false);
    } else setLaneHtml(contextLane, "", "", true);

    const batchLane = card.querySelector('[data-f="batch-lane"]');
    if (run.batch_sizes && run.batch_sizes.length) {
      const skipped = new Set(run.batch_skipped || []);
      const activeBatchIndex = run.current_batch_index != null
        ? run.current_batch_index
        : (run.phase === "batch" && run.current_batch_size != null
          ? run.batch_sizes.findIndex((size, i) => i >= (run.batch_sizes_done || 0) && size === run.current_batch_size)
          : -1);
      const batchSig = [
        run.status, run.phase, run.batch_sizes_done, run.current_batch_index,
        run.current_batch_size, [...skipped].join(","), run.batch_sizes.join(","),
      ].join("|");
      const batchHtml = `<span class="progress-label">Batch sweep</span>` + run.batch_sizes.map((size, i) => {
        let cls = "";
        let title = "";
        if (skipped.has(i)) { cls = "skipped"; title = ' title="skipped"'; }
        else if (run.status === "done" || i < run.batch_sizes_done) cls = "done";
        else if (run.phase === "batch" && (i === activeBatchIndex || (activeBatchIndex < 0 && size === run.current_batch_size))) cls = "active";
        return `<span class="ctx-step ${cls}"${title}>${esc(size)}×</span>`;
      }).join("");
      setLaneHtml(batchLane, batchSig, batchHtml, false);
    } else {
      setLaneHtml(batchLane, "", "", true);
    }

    const stopBtn = card.querySelector('[data-act="stop"]');
    stopBtn.disabled = !(run.status === "running" || run.status === "starting");

    // incremental log fetch — the child runs unbuffered, so this streams live
    const logNode = card.querySelector('[data-f="log"]');
    const cache = state.runLogs[run.id] || (state.runLogs[run.id] = { offset: 0, text: "" });
    if (cache.offset < run.log_length) {
      try {
        const data = await api(`/api/runs/${run.id}?offset=${cache.offset}`);
        cache.offset = data.next_offset;
        cache.text += (cache.text ? "\n" : "") + data.log.join("\n");
        const atBottom = logNode.scrollHeight - logNode.scrollTop - logNode.clientHeight < 40;
        logNode.textContent = cache.text;
        if (atBottom) logNode.scrollTop = logNode.scrollHeight;
      } catch (e) { /* transient */ }
    }
    if ((run.status === "done") && !cache.notified) {
      cache.notified = true;
      state.results = [];   // invalidate results cache
      if (run.result_folders && run.result_folders.length) {
        toast("Run finished — results saved: " + run.result_folders.join(", "));
      }
      if (run.kind === "ctxgen") {
        // new {size}k.txt files may exist now — refresh chips
        try {
          state.meta = await api("/api/meta");
          if (currentView() === "tools" && CB.views.tools) CB.views.tools();
        } catch (e) { /* next reload picks it up */ }
      }
    }
  }

  function startRunsPolling() {
    if (state.runsPollTimer) return;
    state.runsPollTimer = setInterval(() => {
      if (currentView() === "run" || currentView() === "tools") renderRunList();
    }, 1200);
  }
  function stopRunsPolling() {
    clearInterval(state.runsPollTimer);
    state.runsPollTimer = null;
  }

  CB.views.run = renderRunView;
  CB.runs = { renderRunList, startRunsPolling };
})();
