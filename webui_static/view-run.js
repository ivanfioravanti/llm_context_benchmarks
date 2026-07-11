/* Run view: endpoint-driven benchmark launcher plus the live run list
   (status LEDs, live readout, context progress lane, streaming log). */

(function () {
  "use strict";

  const {
    state, esc, fmt, api, toast, fmtDuration, pageHead, engineById, endpointTarget,
    attachModelPicker, modelPickerHtml, currentView,
  } = CB;

  // connection values for the run: from the endpoint, else the manual fields
  function readConnection(ep) {
    const connection = {};
    if (ep) {
      connection.base_url = ep.base_url || "";
      connection.api_key = ep.api_key || "";
      connection.host = ep.host || "";
      connection.port = ep.port || "";
    } else {
      const field = id => document.getElementById(id);
      if (field("rfBaseUrl")) connection.base_url = field("rfBaseUrl").value.trim();
      if (field("rfApiKey")) connection.api_key = field("rfApiKey").value;
      if (field("rfHost")) connection.host = field("rfHost").value.trim();
      if (field("rfPort")) connection.port = field("rfPort").value;
    }
    return connection;
  }

  function renderRunView() {
    const root = document.getElementById("view-run");
    const engines = state.meta.engines;

    // endpoint-first: the endpoint carries engine + connection + default model
    const ep = state.endpoints.find(x => x.id === state.runFormEndpoint) || null;
    if (ep && ep.engine && engineById(ep.engine)) state.runFormEngine = ep.engine;
    if (!state.runFormEngine || !engineById(state.runFormEngine)) {
      state.runFormEngine = (engines.find(e => e.available) || engines[0]).id;
    }
    const engine = engineById(state.runFormEngine);
    const engineLocked = !!(ep && ep.engine);
    const manual = !ep;
    const defaultCtx = engine.default_contexts.split(",").map(s => s.trim());
    const target = ep ? (endpointTarget(ep) || "local") : "";

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
              ${modelPickerHtml("rfModel", ep ? ep.model : "", engine.example)}
              <span class="hint" id="rfModelHint"></span>
            </div>
            <div class="field">
              <label for="rfLabel">Run label</label>
              <input type="text" id="rfLabel" value="${esc(ep ? ep.name : "")}" placeholder="e.g. M3 Ultra · speculative on">
              <span class="hint">Shown in results & comparison charts.</span>
            </div>
          </div>
          ${manual ? renderConnectionFields(engine) : ""}
          ${!engine.available ? `<p class="hint" style="color:var(--critical)">
            This engine needs Apple Silicon (MLX) and is disabled on this machine.</p>` : ""}
          <div class="field" style="margin-bottom:12px">
            <label>Context sizes <span class="unit">tokens ×1000</span></label>
            <div class="ctx-chips" id="rfContexts">
              ${state.meta.context_files.map(c => `
                <label class="ctx-chip"><input type="checkbox" value="${esc(c.name.replace(/k$/, ""))}"
                  ${defaultCtx.includes(String(c.name.replace(/k$/, ""))) ? "checked" : ""}>
                  <span>${esc(c.name)}</span></label>`).join("")}
            </div>
          </div>
          <div class="form-row cols-4">
            <div class="field"><label for="rfMaxTokens">Max tokens</label>
              <input type="number" id="rfMaxTokens" value="128" min="1"></div>
            <div class="field"><label for="rfRuns">Runs / context</label>
              <input type="number" id="rfRuns" value="2" min="1"></div>
            <div class="field"><label for="rfTimeout">Timeout <span class="unit">s</span></label>
              <input type="number" id="rfTimeout" value="3600" min="10"></div>
            <div class="field"><label>&nbsp;</label>
              <label class="check"><input type="checkbox" id="rfSaveResponses"> Save responses</label></div>
          </div>
          ${engine.cold_prefill ? `<label class="check" style="margin-bottom:4px">
            <input type="checkbox" id="rfColdPrefill" checked> Cold prefill (clear cache between contexts)</label>` : ""}
          ${renderEngineOptions(engine)}
          <details class="advanced">
            <summary>Extra CLI arguments</summary>
            <div class="field"><input type="text" id="rfExtraArgs" placeholder="--some-flag value"></div>
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
      state.runFormEndpoint = e.target.value;
      renderRunView();
    });
    const engineSelect = document.getElementById("rfEngine");
    if (engineSelect) engineSelect.addEventListener("change", e => {
      state.runFormEngine = e.target.value;
      renderRunView();
    });

    // like readConnection, but with the engine defaults filled in so model
    // discovery can hit the server before the user typed anything
    const runConnection = () => {
      const connection = { engine: engine.id, ...readConnection(ep) };
      if (!ep) {
        if ("base_url" in connection) connection.base_url = connection.base_url || engine.default_base_url || "";
        if ("host" in connection) connection.host = connection.host || "localhost";
      }
      const ollama = engine.id === "ollama-api" || engine.id === "ollama-cli";
      if (!ollama && !connection.base_url && !connection.host) return null;
      return connection;
    };
    state.runModelPicker = attachModelPicker({
      select: document.getElementById("rfModelSel"),
      input: document.getElementById("rfModel"),
      button: document.getElementById("rfModelLoad"),
      hint: document.getElementById("rfModelHint"),
      getConnection: runConnection,
    });
    for (const id of ["rfBaseUrl", "rfHost", "rfPort", "rfApiKey"]) {
      const node = document.getElementById(id);
      if (node) node.addEventListener("change", () => state.runModelPicker.load(true));
    }
    if (runConnection()) {
      state.runModelPicker.load(true).then(() => {
        if (ep && ep.model) state.runModelPicker.set(ep.model);
      });
    }
    document.getElementById("rfStart").addEventListener("click", startRun);

    renderRunList();
    startRunsPolling();
  }

  function renderConnectionFields(engine) {
    if (engine.connection === "base_url") {
      return `<div class="form-row cols-2">
        <div class="field"><label for="rfBaseUrl">Base URL</label>
          <input type="text" id="rfBaseUrl" placeholder="${esc(engine.default_base_url || "http://host:port/v1")}"></div>
        <div class="field"><label for="rfApiKey">API key (optional)</label>
          <input type="password" id="rfApiKey" autocomplete="off"></div>
      </div>`;
    }
    if (engine.connection === "hostport") {
      return `<div class="form-row cols-2">
        <div class="field"><label for="rfHost">Host</label>
          <input type="text" id="rfHost" value="localhost"></div>
        <div class="field"><label for="rfPort">Port</label>
          <input type="number" id="rfPort" value="8080"></div>
      </div>`;
    }
    return "";
  }

  function renderEngineOptions(engine) {
    if (!engine.options.length) return "";
    const fields = engine.options.map(opt => {
      const id = "opt_" + opt.key;
      if (opt.type === "flag" || opt.type === "invflag" || opt.type === "optbool") {
        return `<label class="check"><input type="checkbox" id="${id}" data-optkey="${esc(opt.key)}"
          ${opt.default ? "checked" : ""}> ${esc(opt.label)}</label>`;
      }
      if (opt.type === "choice") {
        return `<div class="field"><label for="${id}">${esc(opt.label)}</label>
          <select id="${id}" data-optkey="${esc(opt.key)}">
            ${opt.choices.map(c => `<option value="${esc(c)}" ${c === opt.default ? "selected" : ""}>${esc(c || "default")}</option>`).join("")}
          </select></div>`;
      }
      const value = opt.default == null ? "" : opt.default;
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
    const engine = engineById(state.runFormEngine);
    const ep = state.endpoints.find(x => x.id === state.runFormEndpoint) || null;
    const contexts = [...document.querySelectorAll("#rfContexts input:checked")].map(i => i.value);
    if (!contexts.length) { toast("Select at least one context size.", true); return; }

    const options = {};
    document.querySelectorAll("#view-run [data-optkey]").forEach(node => {
      const key = node.dataset.optkey;
      if (node.type === "checkbox") options[key] = node.checked;
      else if (node.value !== "") options[key] = node.value;
    });

    const connection = readConnection(ep);
    const cold = document.getElementById("rfColdPrefill");
    const payload = {
      engine: engine.id,
      model: state.runModelPicker ? state.runModelPicker.value() : document.getElementById("rfModel").value,
      label: document.getElementById("rfLabel").value,
      endpoint_id: ep ? ep.id : null,
      contexts: contexts.join(","),
      max_tokens: document.getElementById("rfMaxTokens").value,
      runs: document.getElementById("rfRuns").value,
      timeout: document.getElementById("rfTimeout").value,
      save_responses: document.getElementById("rfSaveResponses").checked,
      cold_prefill: cold ? cold.checked : undefined,
      connection,
      options,
      extra_args: document.getElementById("rfExtraArgs").value,
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
      </div>
      <div class="readout">
        <div class="readout-cell"><span class="eyebrow">Prompt</span>
          <span class="readout-value" data-f="prompt">–<span class="unit"> t/s</span></span></div>
        <div class="readout-cell"><span class="eyebrow">Generation</span>
          <span class="readout-value" data-f="gen">–<span class="unit"> t/s</span></span></div>
        <div class="readout-cell"><span class="eyebrow">TTFT</span>
          <span class="readout-value" data-f="ttft">–<span class="unit"> s</span></span></div>
        <div class="readout-cell"><span class="eyebrow">Elapsed</span>
          <span class="readout-value" data-f="elapsed">–</span></div>
      </div>
      <div class="ctx-lane" data-f="lane"></div>
      <pre class="console" data-f="log" hidden></pre>`;
    card.querySelector('[data-act="stop"]').addEventListener("click", async () => {
      try { await api(`/api/runs/${run.id}/stop`, { method: "POST" }); } catch (e) { toast(e.message, true); }
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
    card.querySelector(".run-card-title").textContent =
      (run.label ? run.label + " — " : "") + run.engine + " · " + run.model;
    const statusText = { starting: "starting", running: "running", done: "completed",
      failed: "failed (exit " + run.returncode + ")", stopped: "stopped" }[run.status] || run.status;
    card.querySelector(".run-card-meta").textContent =
      statusText + (run.error ? " — " + run.error : "");
    card.querySelector('[data-f="elapsed"]').textContent = fmtDuration(run.elapsed);
    if (run.live.prompt_tps != null)
      card.querySelector('[data-f="prompt"]').innerHTML = `${esc(fmt(run.live.prompt_tps))}<span class="unit"> t/s</span>`;
    if (run.live.generation_tps != null)
      card.querySelector('[data-f="gen"]').innerHTML = `${esc(fmt(run.live.generation_tps))}<span class="unit"> t/s</span>`;
    if (run.live.ttft != null)
      card.querySelector('[data-f="ttft"]').innerHTML = `${esc(run.live.ttft.toFixed(2))}<span class="unit"> s</span>`;

    const lane = card.querySelector('[data-f="lane"]');
    if (run.contexts.length) {
      lane.innerHTML = run.contexts.map((ctx, i) => {
        const name = /k$/.test(ctx) ? ctx : ctx + "k";
        let cls = "";
        if (run.status === "done" || i < run.contexts_done) cls = "done";
        else if (run.current_context && run.current_context.replace(/k$/, "") === String(ctx).replace(/k$/, "")) cls = "active";
        return `<span class="ctx-step ${cls}">${esc(name)}</span>`;
      }).join("");
    } else lane.hidden = true;

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
