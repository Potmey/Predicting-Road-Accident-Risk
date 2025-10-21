// js/app.js
import { DataLoader } from './data-loader.js';
import { buildModel, fitModel, predictOne, evaluateAccuracy } from './model.js';

const $ = (id)=>document.getElementById(id);
const log = (msg)=>{ const t=new Date().toLocaleTimeString(); const el=$('log'); el.textContent = (el.textContent==='—'?'':el.textContent+'\n')+`[${t}] ${msg}`; el.scrollTop=el.scrollHeight; };
const setStatus = (s)=> $('status').textContent = `Status: ${s}`;
function setTfVer(){ $('tfver').textContent = `TF.js: ${tf?.version_core||'unknown'}`; }

let LOADER=null, MODEL=null, READY=false;

async function onTrain(){
  try{
    disable(true);
    setStatus('loading data…'); log('Loading ./data/train.csv');
    if (!LOADER) LOADER = new DataLoader(log, setStatus);
    await LOADER.loadCSV('./data/train.csv');

    setStatus('preparing data…'); log('Encoding & scaling + split 80/20');
    const { featNames } = LOADER.prepareMatrices();
    log(`Features after encoding: ${featNames.length}`);

    const arch = $('arch').value;
    setStatus('building model…'); log(`Building model: ${arch}`);
    MODEL?.dispose?.(); MODEL = buildModel(arch, featNames.length, 0.001);
    log(`Params: ${MODEL.countParams().toLocaleString()}`);

    setStatus('training…'); log('Start fit (epochs=10, batch=256, valSplit=0.1)');
    await fitModel(MODEL, LOADER.getTrain(), LOADER.getTrainY(), 10, 256, log);

    setStatus('testing…'); log('Evaluate accuracy on full test (thr=0.5)');
    const acc = evaluateAccuracy(MODEL, LOADER.getTest(), LOADER.getTestY(), 0.5);
    $('testAcc').textContent = `${(acc*100).toFixed(2)}%`;

    // build simulation form and enable it
    LOADER.buildSimulationForm($('simGrid'));
    $('simFs').disabled = false; $('simCard').style.opacity='1';
    READY = true;
    setStatus('done'); log('Training finished. Simulation enabled.');
  } catch(e){
    console.error(e); log('Error: ' + e.message); setStatus('error');
  } finally{
    disable(false);
  }
}

function riskColor(v){
  if (v < 0.33) return 'green';
  if (v < 0.66) return 'yellow';
  return 'red';
}

function onPredict(){
  if (!READY || !MODEL) { log('Train first.'); return; }
  try{
    setStatus('predicting…');
    const x = LOADER.encodeSimulationInput();
    const r = predictOne(MODEL, x);
    const out = $('riskOut');
    out.textContent = r.toFixed(4);
    out.className = 'risk ' + riskColor(r);
    log(`Predicted risk = ${r.toFixed(6)}`);
    setStatus('ready');
  } catch(e){
    console.error(e); log('Predict error: ' + e.message); setStatus('error');
  }
}

function disable(b){
  $('btnTrain').disabled = b;
  $('arch').disabled = b;
  $('btnPredict').disabled = b || !READY;
}

function main(){
  setTfVer(); setStatus('ready');
  $('btnTrain').onclick = onTrain;
  $('btnPredict').onclick = onPredict;
  disable(false);
}

if (document.readyState==='loading'){ document.addEventListener('DOMContentLoaded', main); } else { main(); }
