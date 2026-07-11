/* Context Bench bootstrap: loads meta + endpoints, fills the sidebar
   readouts and hands control to the router. Views live in view-*.js. */

(function () {
  "use strict";

  const { state, esc, api, initTheme, render } = CB;

  async function boot() {
    initTheme();
    try {
      state.meta = await api("/api/meta");
      state.endpoints = await api("/api/endpoints");
    } catch (e) {
      document.getElementById("main").innerHTML =
        `<div class="empty" style="margin-top:40px"><strong>Backend unreachable</strong>${esc(e.message)}</div>`;
      return;
    }
    const hw = document.getElementById("hwReadout");
    hw.textContent = state.meta.hardware_string || "unknown hardware";
    hw.title = state.meta.hardware_string;
    document.getElementById("mlxBadge").innerHTML = state.meta.mlx_available
      ? `<span class="on">● MLX available</span>`
      : `<span class="off">○ MLX off — no Apple Silicon</span>`;
    render();
  }

  boot();
})();
