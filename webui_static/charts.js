/* Minimal SVG chart library for Context Bench.
   Line charts over context size (log2 x-axis) or batch size, with a
   crosshair tooltip, surface-ringed markers and an HTML legend. */

(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, parent) {
    const node = document.createElementNS(SVG_NS, name);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function fmtNum(value, opts) {
    if (value == null || !isFinite(value)) return "–";
    if (value === 0) return "0";
    const abs = Math.abs(value);
    if (opts && opts.seconds) {
      if (abs < 0.995) return (value * 1000).toFixed(0) + " ms";
      if (abs < 10) return value.toFixed(2) + " s";
      return value.toFixed(1) + " s";
    }
    if (abs >= 1e6) return (value / 1e6).toFixed(1) + "M";
    if (abs >= 1e4) return (value / 1e3).toFixed(1) + "K";
    if (abs >= 100) return value.toFixed(0);
    if (abs >= 10) return value.toFixed(1);
    if (abs >= 1) return value.toFixed(2);
    return value.toFixed(3);
  }

  function trimZeros(s) {
    return s.indexOf(".") >= 0 ? s.replace(/\.?0+$/, "") : s;
  }

  function fmtTick(t) {
    if (t === 0) return "0";
    const abs = Math.abs(t);
    if (abs >= 1e6) return trimZeros((t / 1e6).toFixed(1)) + "M";
    if (abs >= 1e4) return trimZeros((t / 1e3).toFixed(1)) + "K";
    const s = abs >= 100 ? t.toFixed(0) : abs >= 10 ? t.toFixed(1) : abs >= 1 ? t.toFixed(2) : t.toFixed(3);
    return trimZeros(s);
  }

  function niceTicks(min, max, count) {
    if (!(max > min)) max = min + 1;
    const span = max - min;
    const step0 = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    let step = mag;
    for (const m of [1, 2, 2.5, 5, 10]) {
      if (step0 <= m * mag) { step = m * mag; break; }
    }
    const ticks = [];
    const start = Math.ceil(min / step) * step;
    for (let v = start; v <= max + step * 1e-9; v += step) ticks.push(Math.round(v * 1e9) / 1e9);
    return ticks;
  }

  function ctxLabel(v) {
    return (v >= 1 ? (Number.isInteger(v) ? v : v) : v) + "k";
  }

  // ---------------------------------------------------------------
  // Line chart
  // opts: { series: [{name, color, points: [[x, y], ...]}],
  //         logX, xLabel, yLabel, height, seconds, xTickFormat }
  // ---------------------------------------------------------------
  function lineChart(container, opts) {
    container.classList.add("viz");
    container.innerHTML = "";
    const series = (opts.series || []).map(s => ({
      ...s,
      points: (s.points || [])
        .filter(p => p[1] != null && isFinite(p[1]))
        .slice()
        .sort((a, b) => a[0] - b[0]),
    })).filter(s => s.points.length);

    if (!series.length) {
      const empty = document.createElement("div");
      empty.className = "chart-empty";
      empty.textContent = "No data for this metric.";
      container.appendChild(empty);
      return { destroy() {} };
    }

    const width = Math.max(320, opts.width || container.clientWidth || 720);
    const margin = { top: 14, right: 16, bottom: 40, left: 56 };

    // optional in-SVG title & legend (used for standalone PNG export);
    // tokens resolve against the container so a .force-light wrapper works
    const tokens = getComputedStyle(container);
    const tok = name => tokens.getPropertyValue(name).trim();
    if (opts.svgTitle) margin.top += 30;
    let legendRows = [];
    if (opts.svgLegend && series.length >= 2) {
      const iw0 = width - margin.left - margin.right;
      let row = [], used = 0;
      for (const s of series) {
        const w = 30 + s.name.length * 6.8;
        if (used + w > iw0 && row.length) { legendRows.push(row); row = []; used = 0; }
        row.push({ s, x: used });
        used += w;
      }
      if (row.length) legendRows.push(row);
    }
    const legendHeight = legendRows.length ? legendRows.length * 20 + 10 : 0;
    const height = (opts.height || 300) + (opts.svgTitle ? 30 : 0) + legendHeight;

    const xt = v => (opts.logX ? Math.log2(v) : v);
    const xsAll = [...new Set(series.flatMap(s => s.points.map(p => p[0])))].sort((a, b) => a - b);
    const xMin = xt(xsAll[0]);
    const xMax = xt(xsAll[xsAll.length - 1]);
    const ysAll = series.flatMap(s => s.points.map(p => p[1]));
    let yMin = 0;
    let yMax = Math.max(...ysAll);
    if (yMax <= 0) yMax = 1;
    yMax *= 1.06;

    const iw = width - margin.left - margin.right;
    const ih = height - margin.top - margin.bottom - legendHeight;
    const px = v => margin.left + (xMax === xMin ? iw / 2 : ((xt(v) - xMin) / (xMax - xMin)) * iw);
    const py = v => margin.top + ih - ((v - yMin) / (yMax - yMin)) * ih;

    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img" }, null);
    container.appendChild(svg);

    if (opts.svgTitle) {
      const t = el("text", {
        x: margin.left, y: 22, fill: tok("--ink") || "#111",
        "font-family": "system-ui, sans-serif", "font-size": 13.5, "font-weight": 600,
      }, svg);
      t.textContent = opts.svgTitle;
    }
    for (let r = 0; r < legendRows.length; r++) {
      const y = height - legendHeight + 14 + r * 20;
      for (const item of legendRows[r]) {
        const x = margin.left + item.x;
        el("line", { x1: x, x2: x + 14, y1: y - 4, y2: y - 4, stroke: item.s.color, "stroke-width": 3 }, svg);
        const t = el("text", {
          x: x + 20, y, fill: tok("--ink-2") || "#555",
          "font-family": "system-ui, sans-serif", "font-size": 11,
        }, svg);
        t.textContent = item.s.name;
      }
    }

    // gridlines + y ticks — one consistent unit per axis
    const msMode = !!opts.seconds && yMax < 0.995;
    let yTitle = opts.yLabel || "";
    if (msMode && yTitle === "s") yTitle = "ms";
    const yTicks = niceTicks(yMin, msMode ? yMax * 1000 : yMax, 5).map(t => (msMode ? t / 1000 : t));
    for (const tick of yTicks) {
      const y = py(tick);
      el("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "gridline" }, svg);
      const label = el("text", { x: margin.left - 8, y: y + 3.5, "text-anchor": "end", class: "tick-label" }, svg);
      label.textContent = fmtTick(msMode ? tick * 1000 : tick);
    }
    // baseline + x ticks at data positions
    el("line", {
      x1: margin.left, x2: width - margin.right,
      y1: margin.top + ih, y2: margin.top + ih, class: "axisline",
    }, svg);
    const maxXTicks = Math.floor(iw / 46);
    const tickEvery = Math.max(1, Math.ceil(xsAll.length / Math.max(2, maxXTicks)));
    xsAll.forEach((v, i) => {
      if (i % tickEvery !== 0 && i !== xsAll.length - 1) return;
      const x = px(v);
      el("line", { x1: x, x2: x, y1: margin.top + ih, y2: margin.top + ih + 4, class: "axisline" }, svg);
      const label = el("text", { x, y: margin.top + ih + 16, "text-anchor": "middle", class: "tick-label" }, svg);
      label.textContent = opts.xTickFormat ? opts.xTickFormat(v) : ctxLabel(v);
    });
    if (opts.xLabel) {
      const t = el("text", { x: margin.left + iw / 2, y: margin.top + ih + 34, "text-anchor": "middle", class: "axis-title" }, svg);
      t.textContent = opts.xLabel;
    }
    if (yTitle) {
      const t = el("text", {
        x: 12, y: margin.top + ih / 2, class: "axis-title",
        transform: `rotate(-90 12 ${margin.top + ih / 2})`, "text-anchor": "middle",
      }, svg);
      t.textContent = yTitle;
    }

    // series — dots carry their values as data attributes so exported
    // (standalone) SVGs can offer point tooltips without re-plumbing data
    for (const s of series) {
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join("");
      el("path", { d, class: "series-line", stroke: s.color }, svg);
      for (const p of s.points) {
        el("circle", {
          cx: px(p[0]), cy: py(p[1]), r: 4, fill: s.color, class: "series-dot",
          "data-s": s.name,
          "data-x": opts.xTickFormat ? opts.xTickFormat(p[0]) : ctxLabel(p[0]),
          "data-v": fmtNum(p[1], opts),
        }, svg);
      }
    }

    // crosshair + tooltip
    const crosshair = el("line", {
      y1: margin.top, y2: margin.top + ih, x1: 0, x2: 0,
      class: "crosshair", visibility: "hidden",
    }, svg);
    const tooltip = document.createElement("div");
    tooltip.className = "viz-tooltip";
    tooltip.hidden = true;
    container.appendChild(tooltip);

    function onMove(event) {
      const rect = svg.getBoundingClientRect();
      const mx = ((event.clientX - rect.left) / rect.width) * width;
      if (mx < margin.left - 10 || mx > width - margin.right + 10) { onLeave(); return; }
      let best = xsAll[0];
      let bestDist = Infinity;
      for (const v of xsAll) {
        const dist = Math.abs(px(v) - mx);
        if (dist < bestDist) { bestDist = dist; best = v; }
      }
      const x = px(best);
      crosshair.setAttribute("x1", x);
      crosshair.setAttribute("x2", x);
      crosshair.setAttribute("visibility", "visible");

      const rows = series
        .map(s => {
          const p = s.points.find(pt => pt[0] === best);
          return p ? { name: s.name, color: s.color, value: p[1] } : null;
        })
        .filter(Boolean)
        .sort((a, b) => b.value - a.value);
      tooltip.innerHTML =
        `<div class="tt-title">${opts.xTickFormat ? opts.xTickFormat(best) : ctxLabel(best)}</div>` +
        rows.map(r =>
          `<div class="tt-row"><span class="tt-swatch" style="background:${r.color}"></span>` +
          `<span class="tt-name">${escapeHtml(r.name)}</span>` +
          `<span class="tt-val">${fmtNum(r.value, opts)}</span></div>`).join("");
      tooltip.hidden = false;
      const cw = container.clientWidth;
      const ttw = tooltip.offsetWidth;
      const left = (x / width) * rect.width;
      tooltip.style.left = Math.min(Math.max(4, left + 14), cw - ttw - 4) + "px";
      tooltip.style.top = "8px";
    }
    function onLeave() {
      crosshair.setAttribute("visibility", "hidden");
      tooltip.hidden = true;
    }
    svg.addEventListener("mousemove", onMove);
    svg.addEventListener("mouseleave", onLeave);

    // legend (only for >= 2 series)
    if (series.length >= 2 && opts.legend !== false) {
      const legend = document.createElement("div");
      legend.className = "legend";
      for (const s of series) {
        const item = document.createElement("div");
        item.className = "legend-item";
        item.innerHTML = `<span class="legend-swatch" style="background:${s.color}"></span>${escapeHtml(s.name)}`;
        legend.appendChild(item);
      }
      container.appendChild(legend);
    }
    return { destroy() { container.innerHTML = ""; } };
  }

  function sparkline(points, width, height, color) {
    width = width || 110;
    height = height || 26;
    const pts = (points || []).filter(p => p[1] != null && isFinite(p[1]));
    if (pts.length < 2) return "";
    const xs = pts.map(p => Math.log2(p[0]));
    const ys = pts.map(p => p[1]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = 0, yMax = Math.max(...ys) || 1;
    const px = v => 2 + ((v - xMin) / (xMax - xMin || 1)) * (width - 4);
    const py = v => height - 2 - ((v - yMin) / (yMax - yMin || 1)) * (height - 4);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${px(Math.log2(p[0])).toFixed(1)},${py(p[1]).toFixed(1)}`).join("");
    return `<svg class="spark" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" aria-hidden="true">` +
      `<path d="${d}"${color ? ` style="stroke:${color}"` : ""}></path></svg>`;
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  window.Charts = { lineChart, sparkline, fmtNum, escapeHtml };
})();
