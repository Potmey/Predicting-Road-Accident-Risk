// js/data-loader.js
export class DataLoader {
  constructor(logFn, statusFn) {
    this.log = logFn || console.log;
    this.setStatus = statusFn || (()=>{});
    this.raw = null;           // parsed rows (objects)
    this.schema = null;        // { features: {...}, target: 'accident_risk' }
    this.encoders = {};        // one-hot maps for categoricals
    this.scalers = {};         // {numericKey: {type, min, max, mean, std}}
    this.X = null; this.y = null;
    this.split = { idxTrain: [], idxVal: [], idxTest: [] };
  }

  async loadCSV(path, useSubset=false, subsetSize=50000) {
    this.setStatus('loading data…');
    this.log(`Loading CSV: ${path}`);
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
    const text = await res.text();
    this.raw = this.#parseCSV(text);
    if (useSubset && this.raw.length > subsetSize) {
      this.log(`Sampling subset: ${subsetSize}/${this.raw.length}`);
      this.raw = this.#sampleArray(this.raw, subsetSize, 2024);
    }
    this.log(`CSV loaded: rows=${this.raw.length}`);
    this.#inferSchema();
    this.setStatus('data loaded');
  }

  // Basic CSV parser (expects header row, comma separated, no quotes with commas inside for simplicity)
  #parseCSV(text) {
    const [headerLine, ...lines] = text.trim().split(/\r?\n/);
    const headers = headerLine.split(',').map(h => h.trim());
    return lines.map(line => {
      const cells = line.split(',').map(v => v.trim());
      const obj = {};
      headers.forEach((h,i) => obj[h] = cells[i] === undefined ? '' : cells[i]);
      return obj;
    });
  }

  #toNumberMaybe(v) {
    if (v === '' || v === null || v === undefined) return NaN;
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }

  #inferSchema() {
    // Target is fixed
    const target = 'accident_risk';
    if (!this.raw.length || !(target in this.raw[0])) {
      throw new Error(`Target "${target}" not found in CSV.`);
    }

    const knownCategoricals = new Set([
      'road_type','lighting','weather','time_of_day',
      'road_signs_present','public_road','holiday','school_season'
    ]);
    const knownNumeric = new Set([
      'num_lanes','curvature','speed_limit','num_reported_accidents'
    ]);

    const cols = Object.keys(this.raw[0]).filter(k => k !== target);
    const features = {};
    for (const col of cols) {
      // Try infer by known lists; otherwise infer by value patterns
      let type = 'numeric';
      if (knownCategoricals.has(col)) type = 'categorical';
      if (knownNumeric.has(col)) type = 'numeric';

      if (!knownCategoricals.has(col) && !knownNumeric.has(col)) {
        // quick sniff: if many unique small set → categorical
        const sampleVals = this.raw.slice(0, Math.min(5000, this.raw.length)).map(r => r[col]);
        const numericRatio = sampleVals.filter(v => Number.isFinite(Number(v))).length / sampleVals.length;
        const uniq = new Set(sampleVals.map(v => String(v)));
        if (numericRatio > 0.9) type = 'numeric';
        if (uniq.size <= 10) type = 'categorical';
      }

      features[col] = { name: col, type };
    }

    // Compute stats for numeric; collect uniques for categoricals/booleans
    for (const [k, f] of Object.entries(features)) {
      if (f.type === 'numeric') {
        const arr = this.raw.map(r => this.#toNumberMaybe(r[k])).filter(Number.isFinite);
        const min = Math.min(...arr), max = Math.max(...arr);
        const mean = arr.reduce((a,b)=>a+b,0)/arr.length || 0;
        const std = Math.sqrt(arr.reduce((s,v)=>s+(v-mean)*(v-mean),0)/Math.max(1,(arr.length-1)));
        f.stats = { min, max, mean, std, count: arr.length };
      } else {
        const uniq = new Set(this.raw.map(r => String(r[k])));
        f.values = [...uniq].filter(v => v !== '' && v !== 'undefined');
        // normalize booleans if applicable
        const lower = f.values.map(v => v.toLowerCase());
        if (lower.every(v => v === 'true' || v === 'false')) {
          f.type = 'boolean';
          f.values = ['True','False'];
        }
      }
    }

    this.schema = { features, target };
    this.log(`Schema inferred. Features: ${Object.keys(features).length}, target: ${target}`);
  }

  buildFilterControls(containerEl) {
    containerEl.innerHTML = '';
    const make = (html) => {
      const div = document.createElement('div'); div.innerHTML = html; return div.firstElementChild;
    };
    for (const [k, f] of Object.entries(this.schema.features)) {
      if (f.type === 'numeric') {
        const min = Number.isFinite(f.stats?.min) ? f.stats.min : 0;
        const max = Number.isFinite(f.stats?.max) ? f.stats.max : 1;
        const block = make(`
          <div>
            <label>${k} (range)</label>
            <input type="number" id="min_${k}" placeholder="min" value="${min}">
            <input type="number" id="max_${k}" placeholder="max" value="${max}">
          </div>`);
        containerEl.appendChild(block);
      } else if (f.type === 'boolean') {
        const block = make(`
          <div>
            <label>${k}</label>
            <select id="sel_${k}">
              <option value="any" selected>Any</option>
              <option value="True">True</option>
              <option value="False">False</option>
            </select>
          </div>`);
        containerEl.appendChild(block);
      } else if (f.type === 'categorical') {
        const opts = (f.values||[]).map(v => `<option value="${v}">${v}</option>`).join('');
        const block = make(`
          <div>
            <label>${k} (multi)</label>
            <select id="sel_${k}" multiple size="${Math.min(6,(f.values||[]).length||3)}">${opts}</select>
          </div>`);
        containerEl.appendChild(block);
      }
    }
  }

  // Return array of row indices that pass UI filters
  indicesByFilters(readFilterFn) {
    const filters = readFilterFn(this.schema);
    const idx = [];
    for (let i=0;i<this.raw.length;i++){
      const r = this.raw[i];
      let ok = true;
      for (const [k,f] of Object.entries(this.schema.features)) {
        const v = r[k];
        const flt = filters[k];
        if (!flt) continue;
        if (f.type === 'numeric') {
          const n = this.#toNumberMaybe(v);
          if (!Number.isFinite(n)) { ok=false; break; }
          if (n < flt.min || n > flt.max) { ok=false; break; }
        } else if (f.type === 'boolean') {
          if (flt.mode !== 'any' && String(v) !== flt.mode) { ok=false; break; }
        } else if (f.type === 'categorical') {
          if (flt.set && flt.set.size>0 && !flt.set.has(String(v))) { ok=false; break; }
        }
      }
      if (ok) idx.push(i);
    }
    return idx;
  }

  // Prepare encoded & scaled matrices and split into train/val/test
  prepareMatrices(scalerType='minmax', seed=2025) {
    // Build encoders for categoricals/booleans
    this.encoders = {};
    const featNames = [];
    for (const [k,f] of Object.entries(this.schema.features)) {
      if (f.type === 'numeric') {
        featNames.push(k);
      } else if (f.type === 'boolean') {
        featNames.push(k); // map True→1, False→0
      } else if (f.type === 'categorical') {
        this.encoders[k] = (f.values||[]).map(v => String(v));
        for (const _ of this.encoders[k]) featNames.push(`${k}__${_}`);
      }
    }

    // Build X/y
    const X = []; const y = [];
    for (const r of this.raw) {
      const row = [];
      for (const [k,f] of Object.entries(this.schema.features)) {
        if (f.type === 'numeric') {
          row.push(this.#toNumberMaybe(r[k]));
        } else if (f.type === 'boolean') {
          row.push(String(r[k]) === 'True' ? 1 : 0);
        } else if (f.type === 'categorical') {
          const cats = this.encoders[k] || [];
          const one = cats.map(v => (String(r[k])===v ? 1 : 0));
          row.push(...one);
        }
      }
      X.push(row);
      y.push([Number(r[this.schema.target])]);
    }

    // Fit scalers on numeric columns only (in their feature positions)
    this.scalers = { type: scalerType, stats: {} };
    const Xmat = X;
    const nRows = Xmat.length, nCols = (Xmat[0]||[]).length;
    // Determine which columns are numeric original
    const numericColIdx = [];
    let colPointer = 0;
    for (const [k,f] of Object.entries(this.schema.features)) {
      if (f.type === 'numeric' || f.type === 'boolean') {
        numericColIdx.push(colPointer);
        colPointer += 1;
      } else if (f.type === 'categorical') {
        colPointer += (this.encoders[k]||[]).length;
      }
    }

    // collect stats
    for (const c of numericColIdx) {
      const col = Xmat.map(r => r[c]).filter(Number.isFinite);
      const min = Math.min(...col), max = Math.max(...col);
      const mean = col.reduce((a,b)=>a+b,0)/Math.max(1,col.length);
      const std = Math.sqrt(col.reduce((s,v)=>s+(v-mean)*(v-mean),0)/Math.max(1,(col.length-1)));
      this.scalers.stats[c] = { min, max, mean, std };
    }
    // apply scaling
    for (let i=0;i<nRows;i++){
      for (const c of numericColIdx) {
        const v = Xmat[i][c];
        const st = this.scalers.stats[c];
        if (!Number.isFinite(v)) { Xmat[i][c] = 0; continue; }
        if (scalerType === 'minmax') {
          const d = (st.max - st.min) || 1;
          Xmat[i][c] = (v - st.min) / d;
        } else {
          const d = st.std || 1;
          Xmat[i][c] = (v - st.mean) / d;
        }
      }
    }

    this.X = Xmat; this.y = y;
    // random split (no time in CSV)
    const idx = [...Array(nRows).keys()];
    this.#shuffle(idx, seed);
    const nTrain = Math.floor(nRows*0.7);
    const nVal = Math.floor(nRows*0.15);
    this.split.idxTrain = idx.slice(0, nTrain);
    this.split.idxVal = idx.slice(nTrain, nTrain+nVal);
    this.split.idxTest = idx.slice(nTrain+nVal);
    this.log(`Prepared matrices: X=[${nRows}×${nCols}], y=[${nRows}×1]. Split: train=${this.split.idxTrain.length}, val=${this.split.idxVal.length}, test=${this.split.idxTest.length}`);
    return { featNames };
  }

  getTensors(part='train') {
    const idxs = part==='train' ? this.split.idxTrain : part==='val' ? this.split.idxVal : this.split.idxTest;
    const X = idxs.map(i => this.X[i]);
    const y = idxs.map(i => this.y[i]);
    return { X, y };
  }

  // Slice test by filters (for Evaluate (Filtered))
  getFilteredTest(readFilterFn) {
    const idxAll = this.split.idxTest;
    const filters = readFilterFn(this.schema);
    const pass = [];
    for (const i of idxAll) {
      const r = this.raw[i];
      let ok = true;
      for (const [k,f] of Object.entries(this.schema.features)) {
        const v = r[k];
        const flt = filters[k];
        if (!flt) continue;
        if (f.type === 'numeric') {
          const n = this.#toNumberMaybe(v);
          if (!Number.isFinite(n) || n < flt.min || n > flt.max) { ok=false; break; }
        } else if (f.type === 'boolean') {
          if (flt.mode !== 'any' && String(v) !== flt.mode) { ok=false; break; }
        } else if (f.type === 'categorical') {
          if (flt.set && flt.set.size>0 && !flt.set.has(String(v))) { ok=false; break; }
        }
      }
      if (ok) pass.push(i);
    }
    const X = pass.map(i => this.X[i]);
    const y = pass.map(i => this.y[i]);
    return { X, y, n: pass.length };
  }

  // utils
  #shuffle(a, seed=123) {
    let s = seed;
    const rnd = () => (s = (s*16807)%2147483647) / 2147483647;
    for (let i=a.length-1;i>0;i--){
      const j = Math.floor(rnd()*(i+1));
      [a[i],a[j]]=[a[j],a[i]];
    }
  }
  #sampleArray(arr, n, seed=123) {
    const idx = [...Array(arr.length).keys()];
    this.#shuffle(idx, seed);
    const chosen = idx.slice(0, Math.min(n, arr.length));
    return chosen.map(i => arr[i]);
  }
}
