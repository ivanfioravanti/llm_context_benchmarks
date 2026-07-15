/* Export machinery: charts as PNG/JPEG, results as CSV / plain text,
   self-contained HTML report, hand-rolled PDF report, ZIP bundling.
   All client-side, no dependencies. */

(function () {
  "use strict";

  // ------------------------------------------------------------- ZIP (store)

  const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      table[n] = c >>> 0;
    }
    return table;
  })();

  function crc32(bytes) {
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
  }

  function makeZip(files) {
    const encoder = new TextEncoder();
    const chunks = [];
    const central = [];
    let offset = 0;
    const now = new Date();
    const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
    const day = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();

    for (const file of files) {
      const nameBytes = encoder.encode(file.name);
      const crc = crc32(file.data);
      const size = file.data.length;
      const local = new DataView(new ArrayBuffer(30));
      local.setUint32(0, 0x04034b50, true);
      local.setUint16(4, 20, true);
      local.setUint16(6, 0x0800, true);
      local.setUint16(8, 0, true);
      local.setUint16(10, time, true);
      local.setUint16(12, day, true);
      local.setUint32(14, crc, true);
      local.setUint32(18, size, true);
      local.setUint32(22, size, true);
      local.setUint16(26, nameBytes.length, true);
      local.setUint16(28, 0, true);
      chunks.push(new Uint8Array(local.buffer), nameBytes, file.data);

      const cd = new DataView(new ArrayBuffer(46));
      cd.setUint32(0, 0x02014b50, true);
      cd.setUint16(4, 20, true);
      cd.setUint16(6, 20, true);
      cd.setUint16(8, 0x0800, true);
      cd.setUint16(10, 0, true);
      cd.setUint16(12, time, true);
      cd.setUint16(14, day, true);
      cd.setUint32(16, crc, true);
      cd.setUint32(20, size, true);
      cd.setUint32(24, size, true);
      cd.setUint16(28, nameBytes.length, true);
      cd.setUint32(42, offset, true);
      central.push(new Uint8Array(cd.buffer), nameBytes);
      offset += 30 + nameBytes.length + size;
    }

    let cdSize = 0;
    for (const c of central) cdSize += c.length;
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, files.length, true);
    end.setUint16(10, files.length, true);
    end.setUint32(12, cdSize, true);
    end.setUint32(16, offset, true);
    return new Blob([...chunks, ...central, new Uint8Array(end.buffer)], { type: "application/zip" });
  }

  // --------------------------------------------------------- SVG -> canvas

  const INLINE_PROPS = ["fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin",
    "stroke-dasharray", "paint-order", "opacity", "font-family", "font-size", "font-weight",
    "letter-spacing", "visibility"];

  async function svgToCanvas(svgEl, scale, background) {
    const clone = svgEl.cloneNode(true);
    const walk = (orig, copy) => {
      if (orig.nodeType !== 1) return;
      const computed = getComputedStyle(orig);
      let style = "";
      for (const prop of INLINE_PROPS) {
        const value = computed.getPropertyValue(prop);
        if (value) style += `${prop}:${value};`;
      }
      copy.setAttribute("style", style);
      for (let i = 0; i < orig.children.length; i++) walk(orig.children[i], copy.children[i]);
    };
    walk(svgEl, clone);

    const width = parseFloat(svgEl.getAttribute("width"));
    const height = parseFloat(svgEl.getAttribute("height"));
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const xml = new XMLSerializer().serializeToString(clone);
    const url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(xml);

    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error("SVG rasterization failed"));
      img.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  async function canvasToPngBytes(canvas) {
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
    return new Uint8Array(await blob.arrayBuffer());
  }

  async function canvasToJpeg(canvas) {
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.92));
    return { bytes: new Uint8Array(await blob.arrayBuffer()), width: canvas.width, height: canvas.height };
  }

  // --------------------------------------------------------- text builders

  function csvEscape(value) {
    const s = value == null ? "" : String(value);
    return /[",\n;]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  function buildResultsCsv(entries) {
    const metaCols = ["run", "engine", "model", "endpoint", "folder"];
    const dataCols = [];
    for (const e of entries) {
      for (const row of e.detail.results) {
        for (const key of Object.keys(row)) {
          if (key !== "generated_text" && key !== "reasoning_text" && !dataCols.includes(key)) dataCols.push(key);
        }
      }
    }
    const lines = [metaCols.concat(dataCols).join(",")];
    for (const e of entries) {
      const s = e.detail.summary;
      for (const row of e.detail.results) {
        lines.push(metaCols.map(c => csvEscape({
          run: e.name, engine: s.engine, model: s.model, endpoint: s.endpoint, folder: s.folder,
        }[c])).concat(dataCols.map(c => csvEscape(row[c]))).join(","));
      }
    }
    return lines.join("\n") + "\n";
  }

  function buildBatchCsv(entries) {
    const withBatch = entries.filter(e => e.detail.batch_data && e.detail.batch_data.length);
    if (!withBatch.length) return null;
    const dataCols = [];
    for (const e of withBatch) {
      for (const row of e.detail.batch_data) {
        for (const key of Object.keys(row)) if (!dataCols.includes(key)) dataCols.push(key);
      }
    }
    const lines = [["run", "engine", "model"].concat(dataCols).join(",")];
    for (const e of withBatch) {
      const s = e.detail.summary;
      for (const row of e.detail.batch_data) {
        lines.push([csvEscape(e.name), csvEscape(s.engine), csvEscape(s.model)]
          .concat(dataCols.map(c => csvEscape(row[c]))).join(","));
      }
    }
    return lines.join("\n") + "\n";
  }

  function buildCachedCsv(entries) {
    const withCached = entries.filter(e => e.detail.cached_results && e.detail.cached_results.length);
    if (!withCached.length) return null;
    const dataCols = [];
    for (const e of withCached) {
      for (const row of e.detail.cached_results) {
        for (const key of Object.keys(row)) {
          if (key !== "generated_text" && key !== "reasoning_text" && !dataCols.includes(key)) dataCols.push(key);
        }
      }
    }
    const lines = [["run", "engine", "model"].concat(dataCols).join(",")];
    for (const e of withCached) {
      const s = e.detail.summary;
      for (const row of e.detail.cached_results) {
        lines.push([csvEscape(e.name), csvEscape(s.engine), csvEscape(s.model)]
          .concat(dataCols.map(c => csvEscape(row[c]))).join(","));
      }
    }
    return lines.join("\n") + "\n";
  }

  function pad(value, width) {
    const s = value == null ? "–" : String(value);
    return s.length >= width ? s : s + " ".repeat(width - s.length);
  }

  function wrapText(text, max) {
    const lines = [];
    let line = "";
    for (const word of String(text).split(/\s+/)) {
      if (line && (line + " " + word).length > max) { lines.push(line); line = word; }
      else line = line ? line + " " + word : word;
    }
    if (line) lines.push(line);
    return lines;
  }

  // shared tabular model for TXT / HTML / PDF: one table per metric, plus
  // cached re-prompt tables and one table per batch chart
  function buildTables(entries, metrics, fmt, batchDefs) {
    const contexts = [...new Set(entries.flatMap(e => e.detail.results.map(r => r.context_size)))]
      .sort((a, b) => parseFloat(a) - parseFloat(b));
    const tables = [];
    for (const metric of metrics) {
      const rows = contexts.map(ctx => [ctx].concat(entries.map(e => {
        const row = e.detail.results.find(r => r.context_size === ctx);
        return row && row[metric.key] != null && isFinite(row[metric.key])
          ? fmt(row[metric.key], { seconds: metric.seconds }) : null;
      })));
      if (!rows.some(r => r.slice(1).some(v => v != null))) continue;
      tables.push({
        title: `${metric.label} [${metric.unit}]`,
        desc: metric.desc,
        headers: ["context"].concat(entries.map(e => e.name)),
        rows,
      });
    }

    const withCached = entries.filter(e => e.detail.cached_results && e.detail.cached_results.length);
    if (withCached.length) {
      const cachedDefs = [
        { key: "incremental_prompt_tps", title: "Cached re-prompt — incremental prompt [tok/s]",
          desc: "Prefill speed on top of a stored KV cache, counting only the tokens added after the cached prefix." },
        { key: "generation_tps", title: "Cached re-prompt — generation [tok/s]",
          desc: "Decode speed of the warm run that reused the stored KV cache." },
        { key: "time_to_first_token", title: "Cached re-prompt — TTFT [s]", seconds: true,
          desc: "Time to first token when the context prefix is already cached. Lower is better." },
      ];
      const cachedContexts = [...new Set(withCached.flatMap(e => e.detail.cached_results.map(r => r.context_size)))]
        .sort((a, b) => parseFloat(a) - parseFloat(b));
      for (const def of cachedDefs) {
        const rows = cachedContexts.map(ctx => [ctx].concat(entries.map(e => {
          const row = (e.detail.cached_results || []).find(r => r.context_size === ctx);
          return row && row[def.key] != null && isFinite(row[def.key]) && row[def.key] > 0
            ? fmt(row[def.key], { seconds: def.seconds }) : null;
        })));
        if (!rows.some(r => r.slice(1).some(v => v != null))) continue;
        tables.push({
          title: def.title,
          desc: def.desc,
          headers: ["context"].concat(entries.map(e => e.name)),
          rows,
        });
      }
    }

    const withBatch = entries.filter(e => e.detail.batch_data && e.detail.batch_data.length);
    if (withBatch.length && batchDefs) {
      const batchSizes = [...new Set(withBatch.flatMap(e => e.detail.batch_data.map(b => b.batch_size)))]
        .sort((a, b) => a - b);
      for (const def of batchDefs) {
        const rows = batchSizes.map(bs => [bs].concat(entries.map(e => {
          const row = (e.detail.batch_data || []).find(b => b.batch_size === bs);
          return row && row[def.field] > 0 ? fmt(row[def.field], { seconds: def.seconds }) : null;
        })));
        if (!rows.some(r => r.slice(1).some(v => v != null))) continue;
        tables.push({
          title: def.title,
          desc: def.desc,
          headers: ["batch"].concat(entries.map(e => e.name)),
          rows,
        });
      }
    }
    return tables;
  }

  function buildComparisonTxt(entries, tables, title) {
    const lines = [];
    lines.push((title || "CONTEXT BENCH — COMPARISON EXPORT").toUpperCase());
    lines.push("Generated: " + new Date().toISOString().slice(0, 19).replace("T", " "));
    lines.push("");
    lines.push("Runs:");
    entries.forEach((e, i) => {
      const s = e.detail.summary;
      lines.push(`  [${i + 1}] ${e.name}`);
      // deliberately no local hardware info: for endpoint runs the inference
      // ran elsewhere — the endpoint name is the machine that matters
      lines.push(`      ${s.engine} · ${s.model}${s.endpoint ? " · " + s.endpoint : ""}`);
      lines.push(`      ${s.folder}`);
    });
    lines.push("");
    for (const table of tables) {
      lines.push(`== ${table.title} ==`);
      if (table.desc) lines.push(`   ${table.desc}`);
      const widths = table.headers.map((h, i) =>
        Math.max(h.length, ...table.rows.map(r => (r[i] == null ? 1 : String(r[i]).length))) + 2);
      lines.push(table.headers.map((h, i) => pad(h, widths[i])).join(""));
      for (const row of table.rows) lines.push(row.map((v, i) => pad(v, widths[i])).join(""));
      lines.push("");
    }
    return lines.join("\n");
  }

  // --------------------------------------------------------- HTML report

  const htmlEscape = Charts.escapeHtml;

  // Self-contained HTML report with a dark/light toggle (dark is default).
  // charts: [{title, svg}] — raw class-based SVG markup, themed by the
  // report CSS; dots are interactive (hover/click shows the value).
  function buildHtmlReport(title, entries, tables, charts, legend) {
    const h = htmlEscape;
    const generated = new Date().toISOString().slice(0, 19).replace("T", " ");
    const runsHtml = entries.map((e, i) => {
      const s = e.detail.summary;
      return `<tr><td class="idx">${i + 1}</td><td><strong>${h(e.name)}</strong></td>
        <td>${h(s.engine)}</td><td class="mono">${h(s.model)}</td>
        <td>${h(s.endpoint || "–")}</td><td class="mono small">${h(s.folder)}</td></tr>`;
    }).join("");
    const legendHtml = (legend || []).length < 2 ? "" :
      `<div class="legend">${legend.map(l =>
        `<span class="legend-item"><span class="legend-swatch" style="background:${l.color}"></span>${h(l.name)}</span>`).join("")}</div>`;
    const qmark = desc => (desc ? `<span class="qmark" tabindex="0" data-tip="${h(desc)}">?</span>` : "");
    const chartsHtml = charts.map(c => `<figure>
      <figcaption>${h(c.title)}${qmark(c.desc)}</figcaption>
      <div class="viz">${c.svg}</div>
      ${legendHtml}
    </figure>`).join("\n");
    const tablesHtml = tables.map(t => `
      <h2>${h(t.title)}${qmark(t.desc)}</h2>
      <table><thead><tr>${t.headers.map((x, i) =>
        `<th${i ? "" : ' class="left"'}>${h(x)}</th>`).join("")}</tr></thead>
      <tbody>${t.rows.map(r => `<tr>${r.map((v, i) =>
        `<td${i ? "" : ' class="left mono"'}>${v == null ? "–" : h(v)}</td>`).join("")}</tr>`).join("")}</tbody>
      </table>`).join("\n");

    return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${h(title)}</title>
<style>
  :root {
    color-scheme: dark;
    --page: #0c0c0b; --surface: #1a1a19; --raised: #232322;
    --ink: #f4f3ee; --ink-2: #c3c2b7; --muted: #8b897f;
    --hairline: #2c2c2a; --stripe: #1f1f1e; --head: #161615;
    --grid: #262624; --axis: #383835;
  }
  :root[data-theme="light"] {
    color-scheme: light;
    --page: #f4f3ee; --surface: #fcfcfb; --raised: #f4f3ee;
    --ink: #171614; --ink-2: #52514e; --muted: #8b897f;
    --hairline: #e0dfd7; --stripe: #faf9f6; --head: #f9f8f4;
    --grid: #e5e4dc; --axis: #c6c5bb;
  }
  body { margin: 0; background: var(--page); color: var(--ink);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1060px; margin: 0 auto; padding: 40px 28px 80px; }
  header { border-bottom: 2px solid var(--ink); padding-bottom: 18px; margin-bottom: 26px;
    position: relative; }
  .eyebrow { font: 600 10px/1 ui-monospace, Menlo, monospace; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
  h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.02em; }
  .meta { color: var(--ink-2); font-size: 12.5px; }
  .theme-toggle { position: absolute; top: 0; right: 0;
    background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px;
    color: var(--ink-2); font: 11px ui-monospace, Menlo, monospace; letter-spacing: 0.06em;
    padding: 6px 12px; cursor: pointer; }
  .theme-toggle:hover { color: var(--ink); }
  h2 { font-size: 15px; margin: 34px 0 10px; letter-spacing: -0.01em; }
  table { border-collapse: collapse; width: 100%; background: var(--surface);
    border: 1px solid var(--hairline); border-radius: 10px; overflow: hidden; font-size: 12.5px; }
  th { text-align: right; font: 500 10px/1.4 ui-monospace, Menlo, monospace;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
    padding: 9px 12px; border-bottom: 1px solid var(--hairline); background: var(--head); }
  td { padding: 7px 12px; border-bottom: 1px solid var(--hairline); text-align: right;
    font-variant-numeric: tabular-nums; }
  tr:last-child td { border-bottom: none; }
  tr:nth-child(even) td { background: var(--stripe); }
  th.left, td.left { text-align: left; }
  td.idx { color: var(--muted); width: 24px; }
  .mono { font-family: ui-monospace, Menlo, monospace; font-size: 11.5px; }
  .small { font-size: 10.5px; color: var(--muted); }
  figure { margin: 22px 0; background: var(--surface); border: 1px solid var(--hairline);
    border-radius: 12px; padding: 16px 18px; }
  figcaption { font-size: 13.5px; font-weight: 600; margin-bottom: 10px; }
  .viz svg { display: block; width: 100%; height: auto; }
  .viz .gridline { stroke: var(--grid); stroke-width: 1; }
  .viz .axisline { stroke: var(--axis); stroke-width: 1; }
  .viz .tick-label { fill: var(--muted); font-family: ui-monospace, Menlo, monospace; font-size: 10px; }
  .viz .axis-title { fill: var(--muted); font-family: ui-monospace, Menlo, monospace;
    font-size: 9.5px; letter-spacing: 0.12em; text-transform: uppercase; }
  .viz .series-line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .viz .series-dot { stroke: var(--surface); stroke-width: 2; cursor: pointer; transition: r 100ms ease; }
  .viz .series-dot:hover { r: 6.5; }
  .viz .crosshair { display: none; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 16px; padding-top: 10px; }
  .legend-item { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--ink-2); }
  .legend-swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
  .qmark { display: inline-flex; align-items: center; justify-content: center;
    width: 15px; height: 15px; margin-left: 7px; vertical-align: 1px;
    border: 1px solid var(--axis); border-radius: 50%;
    color: var(--muted); font: 600 9.5px/1 ui-monospace, Menlo, monospace;
    cursor: help; position: relative; }
  .qmark:hover { color: var(--ink); border-color: var(--muted); }
  .qmark::after { content: attr(data-tip); position: absolute; left: -20px; top: calc(100% + 8px);
    width: 300px; background: var(--raised); border: 1px solid var(--hairline); border-radius: 8px;
    padding: 8px 11px; font: 400 12px/1.5 system-ui, sans-serif; color: var(--ink-2);
    text-align: left; text-transform: none; letter-spacing: normal; white-space: normal;
    box-shadow: 0 8px 30px rgba(0,0,0,.35); z-index: 20;
    opacity: 0; visibility: hidden; transition: opacity 120ms ease; pointer-events: none; }
  .qmark:hover::after, .qmark:focus::after { opacity: 1; visibility: visible; }
  .pt-tip { position: fixed; z-index: 10; pointer-events: none;
    background: var(--raised); border: 1px solid var(--hairline); border-radius: 8px;
    padding: 7px 10px; font-size: 12px; box-shadow: 0 8px 30px rgba(0,0,0,.35); max-width: 320px; }
  .pt-tip .tip-name { font-weight: 600; display: block; margin-bottom: 2px; }
  .pt-tip .tip-val { font-family: ui-monospace, Menlo, monospace; color: var(--ink-2); }
  footer { margin-top: 44px; color: var(--muted); font-size: 11px;
    font-family: ui-monospace, Menlo, monospace; }
  @media print {
    body { background: #fff; }
    figure { break-inside: avoid; }
    table { break-inside: avoid; }
  }
</style></head><body><div class="wrap">
<header>
  <div class="eyebrow">Context Bench · Report</div>
  <h1>${h(title)}</h1>
  <div class="meta">Generated ${h(generated)} · ${entries.length} run${entries.length === 1 ? "" : "s"}</div>
  <button class="theme-toggle" onclick="var r=document.documentElement;r.dataset.theme=r.dataset.theme==='light'?'':'light'">◐ dark / light</button>
</header>
<h2>Runs</h2>
<table><thead><tr><th class="left"></th><th class="left">Run</th><th class="left">Engine</th>
<th class="left">Model</th><th class="left">Endpoint</th><th class="left">Folder</th></tr></thead>
<tbody>${runsHtml}</tbody></table>
${chartsHtml}
${tablesHtml}
<footer>context bench — llm context benchmarks</footer>
</div>
<script>
(function () {
  var tip = document.createElement("div");
  tip.className = "pt-tip"; tip.hidden = true;
  var name = document.createElement("span"); name.className = "tip-name";
  var val = document.createElement("span"); val.className = "tip-val";
  tip.appendChild(name); tip.appendChild(val);
  document.body.appendChild(tip);
  function show(e) {
    var d = e.target.dataset;
    name.textContent = d.s;                        // textContent: no HTML injection
    val.textContent = d.x + "  ·  " + d.v;
    tip.hidden = false; move(e);
  }
  function move(e) {
    var x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + "px";
    tip.style.top = (e.clientY + 14) + "px";
  }
  function hide() { tip.hidden = true; }
  document.querySelectorAll(".series-dot").forEach(function (dot) {
    dot.addEventListener("mouseenter", show);
    dot.addEventListener("mousemove", move);
    dot.addEventListener("mouseleave", hide);
    dot.addEventListener("click", function (e) { show(e); e.stopPropagation(); });
  });
  document.addEventListener("click", hide);
})();
</script>
</body></html>`;
  }

  // ------------------------------------------------- overview composite PNG

  // Compose all chart canvases into one compact dashboard image.
  // opts: { title, runs: [{name, color, sub}], charts: [{canvas}], tokens }
  function composeOverview({ title, runs, charts, tokens }) {
    const S = 2;                       // everything below in CSS px, drawn at 2x
    const margin = 26;
    const gap = 14;
    const cols = 2;
    const cellW = charts.length ? charts[0].canvas.width / S : 560;
    const width = margin * 2 + cols * cellW + (cols - 1) * gap;

    // header layout
    const legendRows = [];
    let row = [], used = 0;
    for (const r of runs) {
      const w = 24 + (r.name.length + (r.sub ? r.sub.length + 3 : 0)) * 6.2;
      if (used + w > width - 2 * margin && row.length) { legendRows.push(row); row = []; used = 0; }
      row.push({ r, x: used });
      used += w;
    }
    if (row.length) legendRows.push(row);
    const headerH = 64 + legendRows.length * 20;

    const rows = Math.ceil(charts.length / cols);
    const rowHeights = [];
    for (let i = 0; i < rows; i++) {
      const inRow = charts.slice(i * cols, i * cols + cols);
      rowHeights.push(Math.max(...inRow.map(c => c.canvas.height / S)));
    }
    const height = headerH + margin +
      rowHeights.reduce((a, b) => a + b, 0) + gap * Math.max(0, rows - 1) + margin;

    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * S);
    canvas.height = Math.round(height * S);
    const ctx = canvas.getContext("2d");
    ctx.scale(S, S);
    ctx.fillStyle = tokens.page;
    ctx.fillRect(0, 0, width, height);

    // header
    ctx.fillStyle = tokens.muted;
    ctx.font = "600 9px ui-monospace, Menlo, monospace";
    ctx.fillText("CONTEXT BENCH · OVERVIEW", margin, 24);
    ctx.fillStyle = tokens.ink;
    ctx.font = "650 20px system-ui, sans-serif";
    ctx.fillText(title, margin, 46);
    ctx.fillStyle = tokens.muted;
    ctx.font = "11px system-ui, sans-serif";
    const generated = new Date().toISOString().slice(0, 16).replace("T", " ");
    ctx.fillText(generated, width - margin - ctx.measureText(generated).width, 46);
    legendRows.forEach((items, ri) => {
      const y = 64 + ri * 20;
      for (const item of items) {
        const x = margin + item.x;
        ctx.fillStyle = item.r.color;
        ctx.fillRect(x, y - 8, 12, 3.5);
        ctx.fillStyle = tokens.ink2;
        ctx.font = "11px system-ui, sans-serif";
        const label = item.r.sub ? `${item.r.name} — ${item.r.sub}` : item.r.name;
        ctx.fillText(label, x + 18, y - 2);
      }
    });
    ctx.strokeStyle = tokens.hairline;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin, headerH);
    ctx.lineTo(width - margin, headerH);
    ctx.stroke();

    // grid of charts
    let y = headerH + margin;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const chart = charts[i * cols + j];
        if (!chart) break;
        const x = margin + j * (cellW + gap);
        const w = chart.canvas.width / S;
        const hgt = chart.canvas.height / S;
        ctx.drawImage(chart.canvas, x, y, w, hgt);
        ctx.strokeStyle = tokens.hairline;
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, hgt - 1);
      }
      y += rowHeights[i] + gap;
    }
    return canvas;
  }


  // --------------------------------------------------------- PDF report

  function latin1(str) {
    let out = "";
    for (const ch of String(str)) {
      const code = ch.codePointAt(0);
      if (code <= 255) out += ch;
      else if (ch === "–" || ch === "—") out += "-";
      else if (ch === "’" || ch === "‘") out += "'";
      else if (ch === "“" || ch === "”") out += '"';
      else if (ch === "≥") out += ">=";
      else out += "?";
    }
    return out.replace(/[\\()]/g, c => "\\" + c);
  }

  function bytesOf(str) {
    return Uint8Array.from(str, c => c.charCodeAt(0) & 0xff);
  }

  // Minimal PDF writer: Helvetica/Courier text, lines, DCT(JPEG) images.
  function createPdf() {
    const W = 842, H = 595; // A4 landscape, points
    const pages = [];
    let page = null;

    function addPage() {
      page = { ops: [], images: [] };
      pages.push(page);
    }
    // y is given from the TOP of the page for sanity
    function text(x, y, str, opts) {
      const o = opts || {};
      const font = o.mono ? "/F3" : o.bold ? "/F2" : "/F1";
      const size = o.size || 10;
      const color = o.color || [0.09, 0.09, 0.08];
      page.ops.push(`BT ${font} ${size} Tf ${color.join(" ")} rg 1 0 0 1 ${x.toFixed(1)} ${(H - y).toFixed(1)} Tm (${latin1(str)}) Tj ET`);
    }
    function line(x1, y1, x2, y2, gray) {
      page.ops.push(`${gray ?? 0.75} G 0.7 w ${x1.toFixed(1)} ${(H - y1).toFixed(1)} m ${x2.toFixed(1)} ${(H - y2).toFixed(1)} l S`);
    }
    function image(jpeg, x, y, w, h) {
      const name = `Im${pages.length}_${page.images.length}`;
      page.images.push({ name, jpeg });
      page.ops.push(`q ${w.toFixed(1)} 0 0 ${h.toFixed(1)} ${x.toFixed(1)} ${(H - y - h).toFixed(1)} cm /${name} Do Q`);
    }

    function build() {
      const parts = [];
      let offset = 0;
      const offsets = [0];
      const push = bytes => { parts.push(bytes); offset += bytes.length; };
      const addObj = body => {
        offsets.push(offset);
        const n = offsets.length - 1;
        push(bytesOf(`${n} 0 obj\n`));
        if (body instanceof Uint8Array) push(body); else push(bytesOf(body));
        push(bytesOf(`\nendobj\n`));
        return n;
      };

      push(bytesOf("%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"));
      const fontF1 = addObj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
      const fontF2 = addObj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>");
      const fontF3 = addObj("<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>");

      const pageObjNums = [];
      const kidsPlaceholder = [];
      for (const p of pages) {
        const imageObjNums = [];
        for (const img of p.images) {
          const num = addObj(concat([
            bytesOf(`<< /Type /XObject /Subtype /Image /Width ${img.jpeg.width} /Height ${img.jpeg.height} ` +
              `/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${img.jpeg.bytes.length} >>\nstream\n`),
            img.jpeg.bytes,
            bytesOf("\nendstream"),
          ]));
          imageObjNums.push({ name: img.name, num });
        }
        const content = bytesOf(p.ops.join("\n"));
        const contentNum = addObj(concat([
          bytesOf(`<< /Length ${content.length} >>\nstream\n`), content, bytesOf("\nendstream"),
        ]));
        const xobjects = imageObjNums.map(i => `/${i.name} ${i.num} 0 R`).join(" ");
        kidsPlaceholder.push({ contentNum, xobjects });
        pageObjNums.push(null); // filled after pages object number known
      }

      // pages tree object comes after all content, then page objects reference it
      const pagesNumGuess = offsets.length + pages.length; // pages obj is created after page objects
      const realPageNums = [];
      for (const k of kidsPlaceholder) {
        const num = addObj(`<< /Type /Page /Parent ${pagesNumGuess} 0 R /MediaBox [0 0 ${W} ${H}] ` +
          `/Resources << /Font << /F1 ${fontF1} 0 R /F2 ${fontF2} 0 R /F3 ${fontF3} 0 R >> ` +
          `/XObject << ${k.xobjects} >> >> /Contents ${k.contentNum} 0 R >>`);
        realPageNums.push(num);
      }
      const pagesNum = addObj(`<< /Type /Pages /Kids [${realPageNums.map(n => `${n} 0 R`).join(" ")}] /Count ${realPageNums.length} >>`);
      const catalogNum = addObj(`<< /Type /Catalog /Pages ${pagesNum} 0 R >>`);

      const xrefStart = offset;
      let xref = `xref\n0 ${offsets.length}\n0000000000 65535 f \n`;
      for (let i = 1; i < offsets.length; i++) xref += String(offsets[i]).padStart(10, "0") + " 00000 n \n";
      push(bytesOf(xref));
      push(bytesOf(`trailer\n<< /Size ${offsets.length} /Root ${catalogNum} 0 R >>\nstartxref\n${xrefStart}\n%%EOF\n`));
      return new Blob(parts, { type: "application/pdf" });

      function concat(arrays) {
        let total = 0;
        for (const a of arrays) total += a.length;
        const out = new Uint8Array(total);
        let pos = 0;
        for (const a of arrays) { out.set(a, pos); pos += a.length; }
        return out;
      }
    }

    return { W, H, addPage, text, line, image, build };
  }

  // charts: [{title, jpeg:{bytes,width,height}}]
  function buildPdfReport(title, entries, tables, charts) {
    const pdf = createPdf();
    const M = 46;
    const generated = new Date().toISOString().slice(0, 19).replace("T", " ");

    // --- cover ---
    pdf.addPage();
    pdf.text(M, 56, "CONTEXT BENCH · REPORT", { mono: true, size: 9, color: [0.55, 0.53, 0.5] });
    pdf.text(M, 86, title, { bold: true, size: 24 });
    pdf.text(M, 106, `Generated ${generated}  ·  ${entries.length} run${entries.length === 1 ? "" : "s"}`,
      { size: 10.5, color: [0.32, 0.32, 0.3] });
    pdf.line(M, 122, pdf.W - M, 122, 0.2);
    let y = 150;
    pdf.text(M, y, "Runs", { bold: true, size: 13 });
    y += 22;
    entries.forEach((e, i) => {
      const s = e.detail.summary;
      pdf.text(M, y, `${i + 1}.  ${e.name}`, { bold: true, size: 11 });
      y += 15;
      // deliberately no local hardware info: for endpoint runs the inference
      // ran elsewhere — the endpoint name is the machine that matters
      pdf.text(M + 16, y, `${s.engine} · ${s.model}${s.endpoint ? " · " + s.endpoint : ""}`, { size: 9.5, color: [0.32, 0.32, 0.3] });
      y += 13;
      pdf.text(M + 16, y, s.folder, { mono: true, size: 8, color: [0.55, 0.53, 0.5] });
      y += 20;
    });

    // --- chart pages ---
    for (const chart of charts) {
      pdf.addPage();
      pdf.text(M, 44, chart.title, { bold: true, size: 13 });
      pdf.line(M, 56, pdf.W - M, 56, 0.2);
      let iy = 72;
      if (chart.desc) {
        for (const line of wrapText(chart.desc, 100)) {
          pdf.text(M, iy, line, { size: 9, color: [0.45, 0.44, 0.4] });
          iy += 12;
        }
        iy += 6;
      }
      const maxW = pdf.W - 2 * M;
      const maxH = pdf.H - iy - 40;
      const ratio = Math.min(maxW / chart.jpeg.width, maxH / chart.jpeg.height);
      const w = chart.jpeg.width * ratio;
      const h = chart.jpeg.height * ratio;
      pdf.image(chart.jpeg, M + (maxW - w) / 2, iy, w, h);
    }

    // --- data tables (monospace) ---
    const lineHeight = 12;
    let ty = Infinity;
    const newDataPage = () => {
      pdf.addPage();
      pdf.text(M, 44, "Data", { bold: true, size: 13 });
      pdf.line(M, 56, pdf.W - M, 56, 0.2);
      ty = 78;
    };
    const room = () => pdf.H - 40 - ty;
    for (const table of tables) {
      const widths = table.headers.map((h, i) =>
        Math.max(String(h).length, ...table.rows.map(r => (r[i] == null ? 1 : String(r[i]).length))) + 2);
      const headerLine = table.headers.map((h, i) => pad(h, widths[i])).join("");
      const neededRows = table.rows.length + 3;
      if (ty === Infinity || room() < Math.min(neededRows, 6) * lineHeight) newDataPage();
      pdf.text(M, ty, table.title, { bold: true, size: 10.5 });
      ty += 16;
      if (table.desc) {
        for (const line of wrapText(table.desc, 110)) {
          pdf.text(M, ty, line, { size: 8, color: [0.45, 0.44, 0.4] });
          ty += 11;
        }
        ty += 2;
      }
      pdf.text(M, ty, headerLine, { mono: true, size: 8.5, color: [0.45, 0.44, 0.4] });
      ty += 4;
      pdf.line(M, ty, pdf.W - M, ty, 0.6);
      ty += lineHeight - 2;
      for (const row of table.rows) {
        if (room() < lineHeight) {
          newDataPage();
          pdf.text(M, ty, headerLine, { mono: true, size: 8.5, color: [0.45, 0.44, 0.4] });
          ty += lineHeight;
        }
        pdf.text(M, ty, row.map((v, i) => pad(v, widths[i])).join(""), { mono: true, size: 8.5 });
        ty += lineHeight;
      }
      ty += 14;
    }

    return pdf.build();
  }

  function download(blob, filename) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
  }

  window.CBExport = {
    makeZip, svgToCanvas, canvasToPngBytes, canvasToJpeg,
    buildResultsCsv, buildBatchCsv, buildCachedCsv, buildTables, buildComparisonTxt,
    buildHtmlReport, buildPdfReport, composeOverview, download,
  };
})();
