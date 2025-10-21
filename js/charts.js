// js/charts.js
let histChart = null;
let scatterChart = null;

export function renderMetrics({ rmse, mae, r2, acc }) {
  setText('rmse', isFinite(rmse) ? rmse.toFixed(6) : '—');
  setText('mae',  isFinite(mae)  ? mae.toFixed(6)  : '—');
  setText('r2',   isFinite(r2)   ? r2.toFixed(6)   : '—');
  setText('acc',  (acc!==undefined && acc!==null) ? acc.toFixed(4) : '—');
}

export function renderHistogram(residuals) {
  const ctx = getCtx('hist');
  const bins = 30;
  const min = Math.min(...residuals), max = Math.max(...residuals);
  const step = (max - min) / bins || 1;
  const counts = new Array(bins).fill(0);
  for (const r of residuals) {
    let b = Math.floor((r - min)/step);
    if (b < 0) b = 0; if (b >= bins) b = bins-1;
    counts[b]++;
  }
  const labels = [...Array(bins).keys()].map(i => (min + i*step).toFixed(2));

  histChart?.destroy();
  histChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Residuals count', data: counts }] },
    options: {
      responsive: true,
      scales: { x: { ticks: { maxRotation: 0 } } },
      plugins: { legend: { display: false } }
    }
  });
}

export function renderScatter(yTrue, yPred) {
  const ctx = getCtx('scatter');
  const pts = yTrue.map((v,i) => ({x:v, y:yPred[i]}));

  scatterChart?.destroy();
  scatterChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [{ label: 'y_true vs y_pred', data: pts, pointRadius: 2 }] },
    options: {
      responsive: true,
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: 'y_true' } },
        y: { min: 0, max: 1, title: { display: true, text: 'y_pred' } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function getCtx(id){ return document.getElementById(id).getContext('2d'); }
function setText(id, text){ const el = document.getElementById(id); if (el) el.textContent = text; }
