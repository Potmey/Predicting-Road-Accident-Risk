// js/app.js
import { DataLoader } from './data-loader.js';
import { buildModel, fitModel, predict, saveIndexedDB, loadIndexedDB, exportDownloads } from './model.js';
import { renderMetrics, renderHistogram, renderScatter } from './charts.js';

const $ = (id) => document.getElementById(id);
const log = (msg) => {
  const t = new Date().toLocaleTimeString();
  const el = $('log');
  el.textContent = (el.textContent === '—' ? '' : el.textContent + '\n') + `[${t}] ${msg}`;
  el.scrollTop = el.scrollHeight;
};
const setStatus = (s) => $('status').textContent = `Status: ${s}`;
function setTfVer(){ $('tfver').textContent = `TF.js: ${tf?.version_core||'unknown'}`; }

let LOADER = null;
let MODEL = null;
let featNames = null;

function readControls() {
  return {
    path: $('dataSource').value,
    useSubset: $('useSubset').checked,
    subsetSize: parseInt($('subsetSize').value,10),
    scaler: $('scaler').value,
    evalMode: $('evalMode').value,
    thr: parseFloat($('thr').value),
    arch: $('arch').value,
    epochs: parseInt($('epochs').value,10),
    batch: parseInt($('batch').value,10),
    lr: parseFloat($('lr').value),
    valSplit: parseFloat($('valSplit').value),
  };
}

function syncLabels() {
  $('subsetSizeVal').textContent = $('subsetSize').value;
  $('epochsVal').textContent = $('epochs').value;
  $('batchVal').textContent = $('batch').value;
  $('lrVal').textContent = Number($('lr').value).toFixed(4);
  $('valSplitVal').textContent = $('valSplit').value;
  $('thrVal').textContent = $('thr').value;
}

function readFeatureFilters(schema){
  const filters = {};
  for (const [k,f] of Object.entries(schema.features)) {
    if (f.type === 'numeric') {
      const minEl = $(`min_${k}`), maxEl = $(`max_${k}`);
      const min = parseFloat(minEl?.value ?? '-Infinity');
      const max = parseFloat(maxEl?.value ?? 'Infinity');
      filters[k] = { min: Number.isFinite(min)?min:-Infinity, max: Number.isFinite(max)?max:Infinity };
    } else if (f.type === 'boolean') {
      const sel = $(`sel_${k}`);
      filters[k] = { mode: sel?.value || 'any' };
    } else if (f.type === 'categorical') {
      const sel = $(`sel_${k}`);
      const selected = new Set([...(sel?.selectedOptions||[])].map(o => o.value));
      filters[k] = { set: selected };
    }
  }
  return filters;
}

async function onLoadData() {
  try {
    setButtonsDisabled(true);
    setStatus('loading data…');
    log('Loading data…');
    if (!LOADER) LOADER = new DataLoader(log, setStatus);

    const { path, useSubset, subsetSize } = readControls();
    await LOADER.loadCSV(path, useSubset, subsetSize);

    // Build dynamic filter controls from schema
    LOADER.buildFilterControls($('featureGrid'));

    // Encode/scale and split
    const { scaler } = readControls();
    const meta = LOADER.prepareMatrices(scaler);
    featNames = meta.featNames;

    setStatus('data loaded');
    log(`Data ready. Features=${featNames.length}, target=accident_risk. Use “Build Model”.`);
  } catch(e) {
    console.error(e);
    log('Load error: ' + e.message);
    setStatus('error');
  } finally {
    setButtonsDisabled(false);
  }
}

function onBuildModel() {
  try {
    setButtonsDisabled(true);
    setStatus('building model…');
    if (!LOADER || !LOADER.X) { log('Please load data first.'); setStatus('ready'); return; }

    const { arch, lr } = readControls();
    const inputLength = featNames.length;

    MODEL?.dispose?.();
    MODEL = buildModel(arch, inputLength, lr);

    setStatus('model built');
    log(`Model built [${arch}] — params: ${MODEL.countParams().toLocaleString()}`);
  } catch(e) {
    console.error(e);
    log('Build error: ' + e.message);
    setStatus('error');
  } finally {
    setButtonsDisabled(false);
  }
}

async function onTrain() {
  try {
    setButtonsDisabled(true);
    if (!LOADER || !MODEL) { log('Need data and model.'); return; }

    const { epochs, batch, valSplit } = readControls();
    setStatus('training…');
    log(`Training: epochs=${epochs}, batch=${batch}, valSplit=${valSplit}`);

    const tr = LOADER.getTensors('train');
    const va = LOADER.getTensors('val');
    await fitModel(MODEL, tr.X, tr.y, va.X, va.y, {epochs, batchSize: batch}, log);

    setStatus('training complete');
    log('Training complete. Use Evaluate.');
  } catch(e){
    console.error(e);
    log('Train error: ' + e.message);
    setStatus('error');
  } finally {
    setButtonsDisabled(false);
  }
}

function computeMetrics(yTrue, yPred, mode='reg', thr=0.5){
  const n = yTrue.length;
  const resid = [];
  let sse=0, sae=0, mean=0;
  for (let i=0;i<n;i++){ mean += yTrue[i]; }
  mean /= Math.max(1,n);
  let sst=0;
  let correct=0;
  for (let i=0;i<n;i++){
    const e = yPred[i]-yTrue[i];
    resid.push(e);
    sse += e*e;
    sae += Math.abs(e);
    sst += (yTrue[i]-mean)*(yTrue[i]-mean);
    if (mode==='bin'){
      const p = yPred[i] >= thr ? 1 : 0;
      const t = yTrue[i] >= thr ? 1 : 0;
      if (p===t) correct++;
    }
  }
  const rmse = Math.sqrt(sse/Math.max(1,n));
  const mae  = sae/Math.max(1,n);
  const r2   = 1 - (sse/Math.max(1,sst));
  const acc  = (mode==='bin') ? (correct/Math.max(1,n)) : null;
  return { rmse, mae, r2, acc, resid };
}

function flatten1(arr2d){ return arr2d.map(x=>x[0]); }

function evaluateOn(X, y, mode, thr){
  const yPred2d = predict(MODEL, X);
  const yPred = flatten1(yPred2d);
  const yTrue = flatten1(y);
  const { rmse, mae, r2, acc, resid } = computeMetrics(yTrue, yPred, mode, thr);

  renderMetrics({ rmse, mae, r2, acc });
  renderHistogram(resid);
  renderScatter(yTrue, yPred);

  log(`Eval → RMSE=${rmse.toFixed(6)}  MAE=${mae.toFixed(6)}  R²=${r2.toFixed(6)}${mode==='bin' ? `  Acc@${thr}=${acc.toFixed(4)}`:''}`);
}

function onEvaluateFiltered(){
  if (!LOADER || !MODEL) { log('Need data and model.'); return; }
  setStatus('evaluating…');
  const { evalMode, thr } = readControls();
  const filtered = LOADER.getFilteredTest(readFeatureFilters);
  if (filtered.n === 0) {
    log('Filtered test is empty. Relax filters.');
    setStatus('ready');
    return;
  }
  evaluateOn(filtered.X, filtered.y, evalMode, thr);
  setStatus('evaluation done');
}

function onEvaluateFull(){
  if (!LOADER || !MODEL) { log('Need data and model.'); return; }
  setStatus('evaluating…');
  const { evalMode, thr } = readControls();
  const te = LOADER.getTensors('test');
  evaluateOn(te.X, te.y, evalMode, thr);
  setStatus('evaluation done');
}

async function onSaveIdx(){
  if (!MODEL) { log('Build a model first.'); return; }
  await saveIndexedDB(MODEL, 'accident-risk-model');
  log('Saved to IndexedDB.');
}
async function onLoadIdx(){
  try {
    MODEL?.dispose?.();
    MODEL = await loadIndexedDB('accident-risk-model');
    log('Loaded from IndexedDB.');
    setStatus('model loaded');
  } catch(e){
    log('Load error: ' + e.message);
  }
}
async function onExport(){
  if (!MODEL) { log('Build a model first.'); return; }
  await exportDownloads(MODEL, 'accident-risk');
  log('Downloaded model.json and weights.bin');
}

function setButtonsDisabled(b){
  const ids = ['btnLoad','btnBuild','btnTrain','btnEvalFiltered','btnEvalFull','btnSaveIdx','btnLoadIdx','btnExport'];
  ids.forEach(id => { const el = $(id); if (el) el.disabled = b; });
}

function wireUI(){
  setTfVer(); setStatus('ready');
  $('btnLoad').onclick = onLoadData;
  $('btnBuild').onclick = onBuildModel;
  $('btnTrain').onclick = onTrain;
  $('btnEvalFiltered').onclick = onEvaluateFiltered;
  $('btnEvalFull').onclick = onEvaluateFull;
  $('btnSaveIdx').onclick = onSaveIdx;
  $('btnLoadIdx').onclick = onLoadIdx;
  $('btnExport').onclick = onExport;

  ['subsetSize','epochs','batch','lr','valSplit','thr'].forEach(id => $(id).addEventListener('input', syncLabels));
  syncLabels();

  log('Order: “Load Data” → “Build Model” → “Train” → “Evaluate (Filtered/Full)”.');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', wireUI);
} else {
  wireUI();
}
