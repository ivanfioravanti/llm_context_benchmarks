/* Tools view: token-precise context-file generation, including uploading
   custom .txt source texts; generator progress shows inline below. */

(function () {
  "use strict";

  const { state, esc, api, toast, pageHead } = CB;

  function renderToolsView() {
    const root = document.getElementById("view-tools");
    root.innerHTML = `${pageHead("Bench · Utilities", "Tools",
      "Context-file generation and housekeeping.")}
    <div style="max-width:880px">
      <div class="panel">
        <div class="panel-title" style="margin-bottom:10px">Generate context files</div>
        <p style="font-size:12.5px;color:var(--ink-2);margin-top:0">Creates token-precise
          <span class="num">{size}k.txt</span> files from a source text (tiktoken).</p>
        <div class="form-row cols-2">
          <div class="field"><label>Source text</label>
            <div class="model-row">
              <select id="cgSource">${(state.meta.source_files || []).map(f =>
                `<option>${esc(f)}</option>`).join("")}</select>
              <button type="button" class="btn small" id="cgUpload" title="Upload your own .txt source">Upload…</button>
              <input type="file" id="cgUploadFile" accept=".txt,text/plain" hidden>
            </div>
            <span class="hint">Any .txt works — long books give the best token variety.</span></div>
          <div class="field"><label>Sizes <span class="unit">×1000 tokens</span></label>
            <input type="text" id="cgSizes" value="2,4,8,16,32,64,128"></div>
        </div>
        <button class="btn primary" id="cgRun">Generate</button>
        <div style="margin-top:16px">
          <div class="eyebrow" style="margin-bottom:6px">Available context files</div>
          <div class="ctx-chips">${state.meta.context_files.map(c =>
            `<span class="ctx-chip"><span>${esc(c.name)}</span></span>`).join("") || "none"}</div>
        </div>
      </div>
      <div id="runList" data-kind="ctxgen" style="margin-top:16px"></div>
    </div>`;

    document.getElementById("cgUpload").addEventListener("click", () =>
      document.getElementById("cgUploadFile").click());
    document.getElementById("cgUploadFile").addEventListener("change", async e => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        const body = await file.arrayBuffer();
        const res = await fetch(`/api/source-files?name=${encodeURIComponent(file.name)}`, {
          method: "POST", body,
        });
        if (!res.ok) {
          let detail = res.statusText;
          try { detail = (await res.json()).detail || detail; } catch (err) { /* ignore */ }
          throw new Error(detail);
        }
        const saved = await res.json();
        toast(`Uploaded ${saved.name}.`);
        state.meta = await api("/api/meta");
        renderToolsView();
        document.getElementById("cgSource").value = saved.name;
      } catch (err) {
        toast("Upload failed: " + err.message, true);
      }
    });

    document.getElementById("cgRun").addEventListener("click", async () => {
      try {
        await api("/api/context-files", {
          body: {
            source: document.getElementById("cgSource").value,
            sizes: document.getElementById("cgSizes").value,
          },
        });
        toast("Context file generation started.");
        CB.runs.renderRunList();
        CB.runs.startRunsPolling();
      } catch (e) { toast(e.message, true); }
    });
    CB.runs.renderRunList();
  }

  CB.views.tools = renderToolsView;
})();
