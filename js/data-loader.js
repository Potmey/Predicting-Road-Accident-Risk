// js/data-loader.js
export class DataLoader {
  constructor(logFn, statusFn) {
    this.log = logFn || console.log;
    this.setStatus = statusFn || (()=>{});
    this.raw = null;           // [{col:val,...}]
    this.schema = null;        // { features: {name,type,values?,stats?}, target:'accident_risk' }
    this.encoders = {};        // {catKey: [values...]}
    this.scaler = { type:'minmax', stats:{} }; // by feature index
    this.X = null; this.y = null;
    this.idx = { train: [], test: [] };
    this.featNames = [];
  }

  async loadCSV(path='./data/train.csv') {
    this.setStatus('loading data…'); this.log(`Fetching ${path}`);
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
    const text = await res.text();
    this.raw = this.#parseCSV(text);
    if (!this.raw.length) throw new Error('CSV is empty.');
    this.#inferSchema();
    this.setStatus('data loaded'); this.log(`Loaded rows=${this.raw.length}`);
  }

  #parseCSV(text) {
    const [h, ...lines] = text.trim().split(/\r?\n/);
    const headers = h.split(',').map(s=>s.trim());
    return lines.map(line=>{
      const cells = line.split(',').map(v => v.trim());
      const o = {}; headers.forEach((k,i)=>o[k]=cells[i]??'');
      return o;
    });
  }

  #num(v){ const n=Number(v); return Number.isFinite(n)?n:NaN; }

  #inferSchema() {
    const target = 'accident_risk';
    if (!(target in this.raw[0])) throw new Error(`Target "${target}" not found`);
    const knownCat = new Set(['road_type','lighting','weather','time_of_day','road_signs_present','public_road','holiday','school_season']);
    const knownNum = new Set(['num_lanes','curvature','speed_limit','num_reported_accidents']);
    const features = {};
    const cols = Object.keys(this.raw[0]).filter(c=>c!==target);
    for (const c of cols){
      let type='numeric';
      if (knownCat.has(c)) type='categorical';
      if (knownNum.has(c)) type='numeric';
      if (!knownCat.has(c) && !knownNum.has(c)) {
        const sample = this.raw.slice(0,Math.min(5000,this.raw.length)).map(r=>r[c]);
        const uniq = new Set(sample.map(String));
        const numRatio = sample.filter(v=>Number.isFinite(Number(v))).length / Math.max(1,sample.length);
        if (uniq.size<=10) type='categorical';
        else if (numRatio>0.9) type='numeric';
      }
      features[c]={name:c,type};
    }
    // stats / values
    for (const [k,f] of Object.entries(features)){
      if (f.type==='numeric'){
        const arr = this.raw.map(r=>this.#num(r[k])).filter(Number.isFinite);
        const min = Math.min(...arr), max=Math.max(...arr);
        const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
        const std = Math.sqrt(arr.reduce((s,v)=>s+(v-mean)*(v-mean),0)/Math.max(1,(arr.length-1)));
        f.stats={min,max,mean,std};
      } else {
        let vals = [...new Set(this.raw.map(r=>String(r[k])).filter(v=>v!=='').map(String))];
        const lower = vals.map(v=>v.toLowerCase());
        if (lower.every(v=>v==='true'||v==='false')) { f.type='boolean'; vals=['True','False']; }
        f.values=vals.slice(0,50);
      }
    }
    this.schema = { features, target };
  }

  prepareMatrices() {
    // build encoders + featNames
    this.encoders={}; this.featNames=[];
    for (const [k,f] of Object.entries(this.schema.features)){
      if (f.type==='numeric' || f.type==='boolean'){ this.featNames.push(k); }
      else if (f.type==='categorical'){
        this.encoders[k] = f.values||[];
        for (const v of this.encoders[k]) this.featNames.push(`${k}__${v}`);
      }
    }
    const X=[]; const y=[];
    for (const r of this.raw){
      const row=[];
      for (const [k,f] of Object.entries(this.schema.features)){
        if (f.type==='numeric'){ row.push(this.#num(r[k])); }
        else if (f.type==='boolean'){ row.push(String(r[k])==='True'?1:0); }
        else { // categorical → one-hot
          const cats=this.encoders[k]||[];
          for (const v of cats) row.push(String(r[k])===v?1:0);
        }
      }
      X.push(row); y.push([Number(r[this.schema.target])]);
    }
    // fit MinMax scaler on numeric/boolean positions
    this.scaler={type:'minmax', stats:{}};
    const numericIdx=[];
    let col=0;
    for (const [k,f] of Object.entries(this.schema.features)){
      if (f.type==='numeric' || f.type==='boolean'){ numericIdx.push(col); col+=1; }
      else { col += (this.encoders[k]||[]).length; }
    }
    for (const c of numericIdx){
      const colVals = X.map(r=>r[c]).filter(Number.isFinite);
      const min=Math.min(...colVals), max=Math.max(...colVals);
      const mean = colVals.reduce((a,b)=>a+b,0)/Math.max(1,colVals.length);
      const std = Math.sqrt(colVals.reduce((s,v)=>s+(v-mean)*(v-mean),0)/Math.max(1,(colVals.length-1)));
      this.scaler.stats[c]={min,max,mean,std};
    }
    // apply scaling
    for (let i=0;i<X.length;i++){
      for (const c of numericIdx){
        const st=this.scaler.stats[c]; const v=X[i][c];
        if (!Number.isFinite(v)) { X[i][c]=0; continue; }
        const d=(st.max-st.min)||1; X[i][c]=(v-st.min)/d;
      }
    }
    this.X=X; this.y=y;
    // split 80/20 with seed
    const idx=[...Array(X.length).keys()]; this.#shuffle(idx,2025);
    const nTr=Math.floor(idx.length*0.8); this.idx.train=idx.slice(0,nTr); this.idx.test=idx.slice(nTr);
    return { featNames: this.featNames };
  }

  getTrain(){ return this.idx.train.map(i=>this.X[i]); }
  getTrainY(){ return this.idx.train.map(i=>this.y[i]); }
  getTest(){ return this.idx.test.map(i=>this.X[i]); }
  getTestY(){ return this.idx.test.map(i=>this.y[i]); }

  // === Simulation UI ===
  buildSimulationForm(container) {
    container.innerHTML='';
    const add = (html)=>{ const d=document.createElement('div'); d.innerHTML=html.trim(); return container.appendChild(d.firstElementChild); };
    // categorical helpers
    const cat = (id, label, values)=> add(`
      <div><label>${label}</label>
        <select id="${id}">
          ${values.map(v=>`<option value="${v}">${v}</option>`).join('')}
        </select>
      </div>`);
    const bool = (id,label)=> add(`
      <div><label>${label}</label>
        <select id="${id}">
          <option value="True">True</option>
          <option value="False">False</option>
        </select>
      </div>`);
    const num = (id,label,step='1',val='0')=> add(`
      <div><label>${label}</label>
        <input id="${id}" type="number" step="${step}" value="${val}" />
      </div>`);

    // fixed 14 fields (take values/stats from schema when available)
    const F=this.schema.features;
    cat('sim_road_type','road_type', F.road_type?.values||['urban','rural','highway']);
    num('sim_num_lanes','num_lanes','1', String(F.num_lanes?.stats?.mean??2|0));
    num('sim_curvature','curvature','0.01', String(F.curvature?.stats?.mean??0.5));
    num('sim_speed_limit','speed_limit','1', String(F.speed_limit?.stats?.mean??60));
    cat('sim_lighting','lighting', F.lighting?.values||['daylight','night','dim']);
    cat('sim_weather','weather', F.weather?.values||['clear','rainy','foggy']);
    bool('sim_road_signs_present','road_signs_present');
    bool('sim_public_road','public_road');
    cat('sim_time_of_day','time_of_day', F.time_of_day?.values||['morning','afternoon','evening']);
    bool('sim_holiday','holiday');
    bool('sim_school_season','school_season');
    num('sim_num_reported_accidents','num_reported_accidents','1', String(F.num_reported_accidents?.stats?.mean??0|0));
  }

  // read form and encode → scaled vector ready for model
  encodeSimulationInput() {
    const o = (id)=>document.getElementById(id).value;
    const sample = {
      road_type: o('sim_road_type'),
      num_lanes: Number(o('sim_num_lanes')),
      curvature: Number(o('sim_curvature')),
      speed_limit: Number(o('sim_speed_limit')),
      lighting: o('sim_lighting'),
      weather: o('sim_weather'),
      road_signs_present: o('sim_road_signs_present'),
      public_road: o('sim_public_road'),
      time_of_day: o('sim_time_of_day'),
      holiday: o('sim_holiday'),
      school_season: o('sim_school_season'),
      num_reported_accidents: Number(o('sim_num_reported_accidents'))
    };
    // encode like training
    const row=[];
    let col=0;
    for (const [k,f] of Object.entries(this.schema.features)){
      if (f.type==='numeric'){
        let v = Number(sample[k]); // may be NaN
        if (!Number.isFinite(v)) v = this.schema.features[k]?.stats?.mean ?? 0;
        const st = this.scaler.stats[col];
        const d=(st.max-st.min)||1; row.push((v - st.min)/d);
        col+=1;
      } else if (f.type==='boolean'){
        const v = String(sample[k])==='True'?1:0;
        const st = this.scaler.stats[col];
        const d=(st.max-st.min)||1; row.push((v - st.min)/d);
        col+=1;
      } else { // categorical
        const cats = this.encoders[k]||[];
        for (const v of cats) row.push(String(sample[k])===v?1:0);
        col += cats.length;
      }
    }
    return row;
  }

  #shuffle(a, seed=123){
    let s=seed; const rnd=()=> (s=(s*16807)%2147483647)/2147483647;
    for (let i=a.length-1;i>0;i--){ const j=Math.floor(rnd()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; }
  }
}
