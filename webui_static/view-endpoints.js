/* Endpoints view: named inference targets as cards with reachability LED,
   run stats, and the sectioned create/edit dialog with model discovery. */

(function () {
  "use strict";

  const {
    state, esc, api, toast, fmtDate, pageHead, engineById, endpointTarget,
    ensureResults, openModal, closeModal, attachModelPicker, modelPickerHtml, render,
  } = CB;

  function endpointRunStats(ep) {
    const runs = state.results.filter(r => !r.error && r.endpoint === ep.name);
    if (!runs.length) return null;
    const last = runs.map(r => r.timestamp || "").sort().pop();
    return { count: runs.length, last };
  }

  async function renderEndpointsView() {
    const root = document.getElementById("view-endpoints");
    try { state.endpoints = await api("/api/endpoints"); } catch (e) { /* keep old */ }
    try { await ensureResults(); } catch (e) { /* stats become unavailable */ }
    root.innerHTML = `${pageHead("Bench · Targets", "Endpoints",
      "Named inference targets. A run started against one carries its name into results and charts.",
      `<button class="btn primary" id="epNew">New endpoint</button>`)}
    <div class="endpoint-grid" id="epGrid"></div>`;
    document.getElementById("epNew").addEventListener("click", () => endpointForm(null));

    const grid = document.getElementById("epGrid");
    if (!state.endpoints.length) {
      grid.innerHTML = `<div class="empty" style="grid-column:1/-1"><strong>No endpoints yet</strong>
        Add your servers here — e.g. »M3 Ultra · llama.cpp« or »DGX · vLLM« — and their names
        show up in results and comparison charts.</div>`;
      return;
    }
    grid.innerHTML = state.endpoints.map(ep => {
      const engine = ep.engine ? engineById(ep.engine) : null;
      const connection = endpointTarget(ep);
      const stats = endpointRunStats(ep);
      const pingable = !!connection;
      return `
      <div class="ep-card" data-id="${esc(ep.id)}">
        <div class="ep-head">
          <span class="led idle" data-f="led"></span>
          <span class="ep-name">${esc(ep.name)}</span>
          <span class="ep-ping" data-f="ping">${pingable ? "…" : "LOCAL"}</span>
        </div>
        <dl class="ep-specs">
          ${engine ? `<dt>Engine</dt><dd>${esc(engine.label)}</dd>` : `<dt>Engine</dt><dd>any</dd>`}
          ${connection ? `<dt>Target</dt><dd class="mono">${esc(connection)}</dd>` : ""}
          ${ep.model ? `<dt>Model</dt><dd class="mono">${esc(ep.model)}</dd>` : ""}
          ${ep.hardware ? `<dt>Hardware</dt><dd>${esc(ep.hardware)}</dd>` : ""}
          ${ep.api_key ? `<dt>Auth</dt><dd class="mono">API key set</dd>` : ""}
          ${stats ? `<dt>Runs</dt><dd>${stats.count} saved${stats.last ? ` · last ${esc(fmtDate(stats.last))}` : ""}</dd>` : ""}
        </dl>
        ${ep.notes ? `<div class="ep-notes">${esc(ep.notes)}</div>` : ""}
        <div class="ep-actions">
          <button class="btn small primary" data-act="bench">Benchmark</button>
          ${stats ? `<button class="btn small" data-act="results">Results</button>` : ""}
          <span class="spacer"></span>
          <button class="btn small ghost" data-act="edit">Edit</button>
          <button class="btn small ghost danger" data-act="delete" title="Delete endpoint">✕</button>
        </div>
      </div>`;
    }).join("");

    grid.querySelectorAll(".ep-card").forEach(card => {
      const ep = state.endpoints.find(x => x.id === card.dataset.id);
      card.querySelector('[data-act="bench"]').addEventListener("click", () => {
        state.runForm.endpoint = ep.id;
        if (ep.engine && ep.engine !== state.runForm.engine) {
          state.runForm.engine = ep.engine;
          state.runForm.options = null;
        }
        state.runForm.model = ep.model || "";
        state.runForm.label = ep.name || "";
        location.hash = "run";
        render();
      });
      const resultsBtn = card.querySelector('[data-act="results"]');
      if (resultsBtn) resultsBtn.addEventListener("click", () => {
        state.resultsFilter = ep.name;
        location.hash = "results";
      });
      card.querySelector('[data-act="edit"]').addEventListener("click", () => endpointForm(ep));
      card.querySelector('[data-act="delete"]').addEventListener("click", async () => {
        if (!confirm(`Delete endpoint »${ep.name}«?`)) return;
        try {
          await api(`/api/endpoints/${encodeURIComponent(ep.id)}`, { method: "DELETE" });
          renderEndpointsView();
        } catch (e) { toast(e.message, true); }
      });
      pingEndpoint(ep, card);
    });
  }

  async function pingEndpoint(ep, card) {
    if (!endpointTarget(ep)) return;
    const led = card.querySelector('[data-f="led"]');
    const label = card.querySelector('[data-f="ping"]');
    try {
      const res = await api(`/api/endpoints/${encodeURIComponent(ep.id)}/ping`, { method: "POST" });
      if (!card.isConnected) return;
      if (res.ok) {
        led.className = "led on";
        label.textContent = `ONLINE · ${res.latency_ms} ms`;
      } else if (res.ok === false) {
        led.className = "led off";
        label.textContent = "OFFLINE";
        label.title = res.detail || "";
      } else {
        led.className = "led idle";
        label.textContent = "LOCAL";
      }
    } catch (e) {
      if (!card.isConnected) return;
      led.className = "led idle";
      label.textContent = "–";
    }
  }

  function endpointForm(endpoint) {
    const ep = endpoint || {};
    const modal = openModal(`
      <button class="btn small modal-close" data-close>Close</button>
      <div class="eyebrow">Bench · Target</div>
      <h2>${endpoint ? "Edit endpoint" : "New endpoint"}</h2>

      <div class="section-label" style="margin-top:18px">Identity</div>
      <div class="form-row cols-2">
        <div class="field"><label for="epName">Name *</label>
          <input type="text" id="epName" value="${esc(ep.name || "")}" placeholder="M3 Ultra · llama.cpp">
          <span class="hint">Labels every run in results & comparison charts.</span></div>
        <div class="field"><label for="epHardware">Server hardware</label>
          <input type="text" id="epHardware" value="${esc(ep.hardware || "")}" placeholder="e.g. DGX Spark · 128GB unified">
          <span class="hint">The machine the numbers belong to — recorded with every run.</span></div>
      </div>
      <div class="form-row">
        <div class="field"><label for="epNotes">Notes</label>
          <input type="text" id="epNotes" value="${esc(ep.notes || "")}" placeholder="e.g. speculative decoding on"></div>
      </div>

      <div class="section-label">Target</div>
      <div class="form-row cols-2">
        <div class="field"><label for="epEngine">Engine</label>
          <select id="epEngine"><option value="">any (decide per run)</option>
            ${state.meta.engines.map(e => `<option value="${esc(e.id)}" ${e.id === ep.engine ? "selected" : ""}>${esc(e.label)}</option>`).join("")}
          </select>
          <span class="hint" id="epEngineHint"></span></div>
        <div class="field" id="epBaseUrlField"><label for="epBaseUrl">Base URL</label>
          <input type="text" id="epBaseUrl" value="${esc(ep.base_url || "")}" placeholder="http://192.168.1.20:8000/v1">
          <span class="hint" id="epBaseUrlHint"></span></div>
      </div>
      <div class="form-row cols-2" id="epHostRow">
        <div class="field"><label for="epHost">Host</label>
          <input type="text" id="epHost" value="${esc(ep.host || "")}" placeholder="192.168.1.20">
          <span class="hint" id="epHostHint"></span></div>
        <div class="field"><label for="epPort">Port</label>
          <input type="number" id="epPort" value="${esc(ep.port || "")}" placeholder="8080"></div>
      </div>

      <div class="section-label">Auth & defaults</div>
      <div class="form-row cols-2">
        <div class="field"><label for="epApiKey">API key</label>
          <input type="password" id="epApiKey" value="${esc(ep.api_key || "")}" autocomplete="off">
          <span class="hint">Stored locally in webui_endpoints.json (owner-only).</span></div>
        <div class="field"><label for="epModel">Default model</label>
          ${modelPickerHtml("epModel", ep.model, "filled into the run form")}
          <span class="hint" id="epModelHint">Loaded from the target server once it is reachable.</span></div>
      </div>

      <div class="btn-row" style="margin-top:18px">
        <button class="btn primary" id="epSave">${endpoint ? "Save changes" : "Add endpoint"}</button>
      </div>`, { narrow: true });

    // show only the connection fields the chosen engine actually uses
    const syncConnectionFields = () => {
      const engine = engineById(modal.querySelector("#epEngine").value);
      const kind = engine ? engine.connection : "base_url";
      modal.querySelector("#epBaseUrlField").style.display = kind === "hostport" || kind === null ? "none" : "";
      modal.querySelector("#epHostRow").style.display = kind === "hostport" ? "" : "none";
      const hint = modal.querySelector("#epEngineHint");
      if (engine && kind === null) hint.textContent = "Runs locally — no connection needed.";
      else if (engine && kind === "hostport") hint.textContent = "Connects via host + port.";
      else if (engine) {
        hint.textContent = "Connects via base URL.";
        const baseUrl = modal.querySelector("#epBaseUrl");
        if (!baseUrl.value && engine.default_base_url) baseUrl.placeholder = engine.default_base_url;
      } else hint.textContent = "";
    };
    modal.querySelector("#epEngine").addEventListener("change", syncConnectionFields);
    syncConnectionFields();

    // Inside Docker, "localhost" is the container itself — servers on the
    // machine running Docker are reachable via host.docker.internal instead.
    if (state.meta.in_container) {
      const LOOPBACK_RE = /localhost|127\.0\.0\.1|0\.0\.0\.0|\[?::1\]?/i;
      const dockerHint = (input, hint) => {
        const sync = () => {
          if (LOOPBACK_RE.test(input.value)) {
            hint.textContent = "This points at the container itself, not this machine — use host.docker.internal instead.";
            hint.classList.add("warn");
          } else {
            hint.textContent = "Running in Docker: for a server on this machine use host.docker.internal, not localhost.";
            hint.classList.remove("warn");
          }
        };
        input.addEventListener("input", sync);
        sync();
      };
      dockerHint(modal.querySelector("#epBaseUrl"), modal.querySelector("#epBaseUrlHint"));
      dockerHint(modal.querySelector("#epHost"), modal.querySelector("#epHostHint"));
    }

    const epConnection = () => {
      const engineId = modal.querySelector("#epEngine").value;
      const engine = engineById(engineId);
      const connection = {
        engine: engineId,
        base_url: modal.querySelector("#epBaseUrl").value.trim(),
        api_key: modal.querySelector("#epApiKey").value,
        host: modal.querySelector("#epHost").value.trim(),
        port: modal.querySelector("#epPort").value,
      };
      if (engine && engine.connection === "base_url") {
        connection.host = "";
        connection.port = "";
      } else if (engine && engine.connection === "hostport") {
        connection.base_url = "";
      }
      const ollama = engineId === "ollama-api" || engineId === "ollama-cli";
      if (!ollama && !connection.base_url && !connection.host) return null;
      return connection;
    };
    const epModelPicker = attachModelPicker({
      select: modal.querySelector("#epModelSel"),
      input: modal.querySelector("#epModel"),
      button: modal.querySelector("#epModelLoad"),
      hint: modal.querySelector("#epModelHint"),
      getConnection: epConnection,
    });
    for (const id of ["#epEngine", "#epBaseUrl", "#epHost", "#epPort", "#epApiKey"]) {
      modal.querySelector(id).addEventListener("change", () => epModelPicker.load(true));
    }
    if (epConnection()) epModelPicker.load(true);

    modal.querySelector("#epSave").addEventListener("click", async () => {
      const payload = {
        name: modal.querySelector("#epName").value,
        engine: modal.querySelector("#epEngine").value,
        base_url: modal.querySelector("#epBaseUrl").value,
        api_key: modal.querySelector("#epApiKey").value,
        host: modal.querySelector("#epHost").value,
        port: modal.querySelector("#epPort").value,
        model: epModelPicker.value(),
        hardware: modal.querySelector("#epHardware").value,
        notes: modal.querySelector("#epNotes").value,
      };
      try {
        const saved = endpoint
          ? await api(`/api/endpoints/${encodeURIComponent(endpoint.id)}`, { method: "PUT", body: payload })
          : await api("/api/endpoints", { body: payload });
        closeModal();
        renderEndpointsView();
        toast(saved.base_url_corrected
          ? `Endpoint saved — appended /v1 to the base URL (the server answers there): ${saved.base_url}`
          : "Endpoint saved.");
      } catch (e) { toast(e.message, true); }
    });
  }

  CB.views.endpoints = renderEndpointsView;
})();
