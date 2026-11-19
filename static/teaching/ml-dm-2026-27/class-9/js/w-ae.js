/* Autoencoder lab: trains a real 64 -> 32 -> k -> 32 -> 64 autoencoder on 8x8 handwritten digits, in the browser.
   Pure JS (manual backprop + Adam). Shows reconstructions, the 2-D latent space, decoding of arbitrary latent points,
   and reconstruction error on a drawing of your own.                                                              */
(function () {
  'use strict';
  const root = document.getElementById('w-ae'); if (!root) return;

  const H = 32, NTRAIN = 800, KS = [2, 4, 8, 16], MAXEP = 400, LR = 0.01, BS = 32;
  const PAL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
  const D = U.digits();
  const Xtr = D.X.subarray(0, NTRAIN * 64), Xte = D.X.subarray(NTRAIN * 64), NTE = D.n - NTRAIN;

  /* ---------------- tiny linear algebra ---------------- */
  function mm(A, B, n, m, p) { const C = new Float64Array(n * p); for (let i = 0; i < n; i++) for (let t = 0; t < m; t++) { const a = A[i * m + t]; if (a === 0) continue; const bo = t * p, co = i * p; for (let j = 0; j < p; j++) C[co + j] += a * B[bo + j]; } return C; }
  function mmTA(A, B, n, m, p) { const C = new Float64Array(m * p); for (let i = 0; i < n; i++) for (let t = 0; t < m; t++) { const a = A[i * m + t]; if (a === 0) continue; const co = t * p, bo = i * p; for (let j = 0; j < p; j++) C[co + j] += a * B[bo + j]; } return C; }
  function mmTB(A, B, n, p, m) { const C = new Float64Array(n * m); for (let i = 0; i < n; i++) for (let t = 0; t < m; t++) { let s = 0; const ao = i * p, bo = t * p; for (let j = 0; j < p; j++) s += A[ao + j] * B[bo + j]; C[i * m + t] = s; } return C; }
  function addB(A, b, n, m) { for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) A[i * m + j] += b[j]; }
  function colsum(A, n, m) { const s = new Float64Array(m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) s[j] += A[i * m + j]; return s; }

  /* ---------------- the autoencoder ---------------- */
  class AE {
    constructor(k) {
      this.k = k; this.epoch = 0; this.t = 0; this.hist = [];
      const r = U.mulberry32(1000 + k), init = (i, o) => { const a = new Float64Array(i * o), s = Math.sqrt(2 / (i + o)); for (let q = 0; q < a.length; q++) a[q] = U.gauss(r) * s; return a; };
      this.P = [init(64, H), new Float64Array(H), init(H, k), new Float64Array(k), init(k, H), new Float64Array(H), init(H, 64), new Float64Array(64)];
      this.M = this.P.map((p) => new Float64Array(p.length)); this.V = this.P.map((p) => new Float64Array(p.length));
      this.rnd = U.mulberry32(7 + k);
    }
    encode(X, B) { const P = this.P, h1 = mm(X, P[0], B, 64, H); addB(h1, P[1], B, H); for (let i = 0; i < h1.length; i++) h1[i] = Math.tanh(h1[i]); const z = mm(h1, P[2], B, H, this.k); addB(z, P[3], B, this.k); return { h1, z }; }
    decode(Z, B) { const P = this.P, h2 = mm(Z, P[4], B, this.k, H); addB(h2, P[5], B, H); for (let i = 0; i < h2.length; i++) h2[i] = Math.tanh(h2[i]); const o = mm(h2, P[6], B, H, 64); addB(o, P[7], B, 64); for (let i = 0; i < o.length; i++) o[i] = 1 / (1 + Math.exp(-o[i])); return { h2, o }; }
    mse(X, B) { const { z } = this.encode(X, B), { o } = this.decode(z, B); let s = 0; for (let i = 0; i < o.length; i++) { const d = o[i] - X[i]; s += d * d; } return s / o.length; }
    step(X, B) {
      const P = this.P, k = this.k;
      const { h1, z } = this.encode(X, B), { h2, o } = this.decode(z, B);
      let loss = 0; const dz4 = new Float64Array(o.length), sc = 2 / (B * 64);
      for (let i = 0; i < o.length; i++) { const d = o[i] - X[i]; loss += d * d; dz4[i] = d * sc * o[i] * (1 - o[i]); }
      const g = new Array(8);
      g[6] = mmTA(h2, dz4, B, H, 64); g[7] = colsum(dz4, B, 64);
      const dh2 = mmTB(dz4, P[6], B, 64, H), dz3 = new Float64Array(dh2.length); for (let i = 0; i < dz3.length; i++) dz3[i] = dh2[i] * (1 - h2[i] * h2[i]);
      g[4] = mmTA(z, dz3, B, k, H); g[5] = colsum(dz3, B, H);
      const dz = mmTB(dz3, P[4], B, H, k);
      g[2] = mmTA(h1, dz, B, H, k); g[3] = colsum(dz, B, k);
      const dh1 = mmTB(dz, P[2], B, k, H), dz1 = new Float64Array(dh1.length); for (let i = 0; i < dz1.length; i++) dz1[i] = dh1[i] * (1 - h1[i] * h1[i]);
      g[0] = mmTA(X, dz1, B, 64, H); g[1] = colsum(dz1, B, H);
      this.t++; const b1 = 1 - Math.pow(0.9, this.t), b2 = 1 - Math.pow(0.999, this.t);
      for (let q = 0; q < 8; q++) { const p = P[q], m = this.M[q], v = this.V[q], gg = g[q]; for (let i = 0; i < p.length; i++) { m[i] = 0.9 * m[i] + 0.1 * gg[i]; v[i] = 0.999 * v[i] + 0.001 * gg[i] * gg[i]; p[i] -= LR * (m[i] / b1) / (Math.sqrt(v[i] / b2) + 1e-8); } }
      return loss / o.length;
    }
    trainEpoch() {
      const idx = Array.from({ length: NTRAIN }, (_, i) => i);
      for (let i = NTRAIN - 1; i > 0; i--) { const j = Math.floor(this.rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      let tot = 0, nb = 0;
      for (let s = 0; s + BS <= NTRAIN; s += BS) { const X = new Float64Array(BS * 64); for (let b = 0; b < BS; b++) X.set(Xtr.subarray(idx[s + b] * 64, idx[s + b] * 64 + 64), b * 64); tot += this.step(X, BS); nb++; }
      this.epoch++; this.hist.push({ tr: tot / nb, te: this.mse(Float64Array.from(Xte), NTE) });
    }
    get testErr() { return this.hist.length ? this.hist[this.hist.length - 1].te : null; }
  }

  /* baselines */
  const mean = new Float64Array(64); for (let i = 0; i < NTRAIN; i++) for (let j = 0; j < 64; j++) mean[j] += Xtr[i * 64 + j] / NTRAIN;
  let BASE = 0; for (let i = 0; i < NTE; i++) for (let j = 0; j < 64; j++) { const d = mean[j] - Xte[i * 64 + j]; BASE += d * d / (NTE * 64); }

  /* ---------------- UI ---------------- */
  root.innerHTML = `
    <div class="grid g2" style="align-items:start">
      <div class="card flat">
        <div class="row" style="gap:.6rem">
          <label class="ctl" style="margin:0"><span>Latent vector size <i>k</i> (numbers in it)</span>
            <div class="seg" id="ae-k">${KS.map((k, i) => `<button data-k="${k}" class="${i === 0 ? 'on' : ''}">${k}</button>`).join('')}</div></label>
          <span class="spacer"></span>
          <button class="btn primary" id="ae-train">▶ Train</button>
          <button class="btn small" id="ae-step">+1 epoch</button>
          <button class="btn small" id="ae-reset">Reset</button>
        </div>
        <div class="grid g3 mt" style="gap:.6rem">
          <div class="stat"><div class="n" id="ae-ep">0</div><div class="l">epochs (passes over 800 digits)</div></div>
          <div class="stat cool"><div class="n" id="ae-te">–</div><div class="l">test error (MSE, unseen digits)</div></div>
          <div class="stat"><div class="n" id="ae-base">${BASE.toFixed(3)}</div><div class="l">error if we always output the <b>average</b> digit</div></div>
        </div>
        <div class="mt"><canvas id="ae-curve" class="cv"></canvas><div class="tiny" style="margin-top:.15rem"><span style="color:#7a8aa3;font-weight:800">━</span> training error &nbsp;<span style="color:var(--gen);font-weight:800">━</span> test error &nbsp;<span style="color:#999">┅</span> average-digit baseline</div></div>
        <div class="mt">
          <canvas id="ae-rec" class="cv"></canvas>
          <div class="row" style="margin-top:.2rem"><span class="tiny" style="flex:1">Top: original (never trained on). Bottom: what the autoencoder rebuilds from its <b id="ae-kk">2</b>-number latent vector.</span><button class="btn small" id="ae-new">New examples</button></div>
        </div>
      </div>

      <div class="card flat">
        <div class="tabs" id="ae-tabs"><button data-t="lat" class="on">Latent space</button><button data-t="draw">Draw your own</button><button data-t="bars">Error vs. bottleneck</button></div>
        <div id="ae-lat">
          <div class="row top" style="gap:.9rem;flex-wrap:nowrap">
            <div><canvas id="ae-sc" class="cv" style="cursor:crosshair"></canvas><div class="legend" id="ae-leg"></div></div>
            <div style="min-width:7.5rem">
              <div class="cap" style="text-align:left;margin:0 0 .2rem">DECODED POINT</div>
              <canvas id="ae-dec" class="cv"></canvas>
              <div class="tiny" id="ae-xy" style="margin-top:.2rem"></div>
            </div>
          </div>
          <div class="verdict mt" id="ae-verd">Hover over the map: each dot is one digit, squeezed into the latent vector. The decoder turns any point back into an image.</div>
          <div class="tiny mt" id="ae-note"></div>
        </div>
        <div id="ae-draw" style="display:none">
          <div class="row top" style="gap:1.2rem">
            <div><canvas id="ae-in" class="cv" style="cursor:crosshair"></canvas><div class="cap">Your drawing</div></div>
            <div style="align-self:center;font-size:1.4rem;color:var(--muted)">→</div>
            <div><canvas id="ae-out" class="cv"></canvas><div class="cap">Reconstruction</div></div>
          </div>
          <div class="row mt" style="gap:.35rem"><button class="btn small" id="ae-dr-digit">Load a test digit</button><button class="btn small" id="ae-dr-noise">Add noise</button><button class="btn small" id="ae-dr-cross">Draw a “X”</button><button class="btn small" id="ae-dr-clear">Clear</button><span class="tiny">draw = click/drag · erase = shift</span></div>
          <div class="verdict mt" id="ae-dr-verd"></div>
        </div>
        <div id="ae-bars" style="display:none">
          <div class="small" style="margin-bottom:.8rem">Train each bottleneck size for a while, then compare. <b>Smaller latent vector → more compression → higher error.</b></div>
          <div class="mini-bars" id="ae-barsrow"></div><div class="mini-bars-x">${KS.map((k) => `<div>k = ${k}</div>`).join('')}</div>
          <div class="tiny mt">Reference: the “average digit” baseline is ${BASE.toFixed(3)}. Each bar is the current test error of that model.</div>
        </div>
      </div>
    </div>`;

  const $ = (s) => root.querySelector(s);
  const models = {}; KS.forEach((k) => (models[k] = new AE(k)));
  let k = 2, training = false, raf = null, tab = 'lat', ex = [], viewBox = null, hoverPt = null, drawImg = new Float64Array(64), drawing = false;
  const exRnd = U.mulberry32(3);
  function pickExamples() { ex = []; const used = new Set(); while (ex.length < 10) { const i = Math.floor(exRnd() * NTE); if (!used.has(i)) { used.add(i); ex.push(i); } } }
  pickExamples();
  const M = () => models[k];

  /* ---------- latent codes & 2-D projection ---------- */
  let codes = null, proj = null;
  function computeLatent() {
    const m = M(), n = D.n; codes = m.encode(Float64Array.from(D.X), n).z;
    if (k === 2) { proj = { mu: [0, 0], v1: [1, 0], v2: [0, 1], pts: null }; }
    else {
      const mu = new Float64Array(k); for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) mu[j] += codes[i * k + j] / n;
      const C = new Float64Array(k * k); for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) C[a * k + b] += (codes[i * k + a] - mu[a]) * (codes[i * k + b] - mu[b]) / n;
      const vecs = [], Cc = Float64Array.from(C);
      for (let c = 0; c < 2; c++) {
        let v = new Float64Array(k).map((_, j) => Math.sin(j + 1 + c)); // deterministic start
        for (let it = 0; it < 60; it++) { const w = new Float64Array(k); for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) w[a] += Cc[a * k + b] * v[b]; const nr = Math.hypot(...w) || 1; v = w.map((x) => x / nr); }
        let big = 0; v.forEach((x, j) => { if (Math.abs(x) > Math.abs(v[big])) big = j; }); if (v[big] < 0) v = v.map((x) => -x);
        let lam = 0; { const w = new Float64Array(k); for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) w[a] += Cc[a * k + b] * v[b]; for (let a = 0; a < k; a++) lam += w[a] * v[a]; }
        for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) Cc[a * k + b] -= lam * v[a] * v[b];
        vecs.push(v);
      }
      proj = { mu, v1: vecs[0], v2: vecs[1] };
    }
    const P = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) { let a = 0, b = 0; for (let j = 0; j < k; j++) { const d = codes[i * k + j] - proj.mu[j]; a += d * proj.v1[j]; b += d * proj.v2[j]; } P[i * 2] = a; P[i * 2 + 1] = b; }
    proj.P = P;
    // median nearest-neighbour distance in the 2-D view (subsample) -> "is this a hole?"
    // (NN distance in 2-D scales with 1/sqrt(density), so a subsample of size `sub` is rescaled by sqrt(sub/n))
    const sub = 300, nn = []; for (let a = 0; a < sub; a++) { let best = 1e9; for (let b = 0; b < sub; b++) if (a !== b) best = Math.min(best, Math.hypot(P[a * 2] - P[b * 2], P[a * 2 + 1] - P[b * 2 + 1])); nn.push(best); }
    nn.sort((x, y) => x - y); proj.nn = nn[Math.floor(nn.length / 2)] * Math.sqrt(sub / n);
    // view box
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; for (let i = 0; i < n; i++) { x0 = Math.min(x0, P[i * 2]); x1 = Math.max(x1, P[i * 2]); y0 = Math.min(y0, P[i * 2 + 1]); y1 = Math.max(y1, P[i * 2 + 1]); }
    const pad = 0.12 * Math.max(x1 - x0, y1 - y0, 1e-3), t = { x0: x0 - pad, x1: x1 + pad, y0: y0 - pad, y1: y1 + pad };
    const span = Math.max(t.x1 - t.x0, t.y1 - t.y0), cx = (t.x0 + t.x1) / 2, cy = (t.y0 + t.y1) / 2; // square view
    const tgt = { x0: cx - span / 2, x1: cx + span / 2, y0: cy - span / 2, y1: cy + span / 2 };
    viewBox = viewBox && M().epoch > 1 ? { x0: U.lerp(viewBox.x0, tgt.x0, .35), x1: U.lerp(viewBox.x1, tgt.x1, .35), y0: U.lerp(viewBox.y0, tgt.y0, .35), y1: U.lerp(viewBox.y1, tgt.y1, .35) } : tgt;
  }
  function decodeAt(px, py) {
    const z = new Float64Array(k); for (let j = 0; j < k; j++) z[j] = proj.mu[j] + px * proj.v1[j] + py * proj.v2[j];
    return M().decode(z, 1).o;
  }

  /* ---------- drawing routines ---------- */
  function drawCurve() {
    const cv = $('#ae-curve'), W = Math.min(root.querySelector('.card').clientWidth - 36, 560), Hh = 96, ctx = U.fit(cv, W, Hh), h = M().hist;
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, Hh);
    const xmax = Math.max(60, h.length), ymax = BASE * 1.15, X = (e) => 6 + (e / xmax) * (W - 12), Y = (v) => Hh - 6 - (U.clamp(v, 0, ymax) / ymax) * (Hh - 12);
    ctx.setLineDash([4, 3]); ctx.strokeStyle = '#999'; ctx.beginPath(); ctx.moveTo(6, Y(BASE)); ctx.lineTo(W - 6, Y(BASE)); ctx.stroke(); ctx.setLineDash([]);
    [['tr', '#7a8aa3'], ['te', '#e8712d']].forEach(([key, col]) => { if (!h.length) return; ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); h.forEach((p, i) => { const x = X(i + 1), y = Y(p[key]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); });
    ctx.fillStyle = '#667180'; ctx.font = '600 10px system-ui'; ctx.textAlign = 'right'; ctx.fillText('epochs →', W - 6, Hh - 8);
    if (!h.length) { ctx.textAlign = 'center'; ctx.fillStyle = '#9aa3b0'; ctx.font = '600 12px system-ui'; ctx.fillText('Press “Train” and watch the error fall', W / 2, Hh / 2); }
  }
  function drawRec() {
    const cv = $('#ae-rec'), px = 6, gap = 6, w = 10 * (8 * px + gap) - gap, ctx = U.fit(cv, w, 2 * 8 * px + 8);
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, 2 * 8 * px + 8);
    const X = new Float64Array(10 * 64); ex.forEach((e, i) => X.set(Xte.subarray(e * 64, e * 64 + 64), i * 64));
    const { z } = M().encode(X, 10), { o } = M().decode(z, 10);
    for (let i = 0; i < 10; i++) {
      U.drawGrid(ctx, X.subarray(i * 64, i * 64 + 64), 8, i * (8 * px + gap), 0, px);
      U.drawGrid(ctx, o.subarray(i * 64, i * 64 + 64), 8, i * (8 * px + gap), 8 * px + 8, px);
    }
    $('#ae-kk').textContent = k;
  }
  function drawScatter() {
    const cv = $('#ae-sc'), S = Math.max(220, Math.min(340, Math.floor(($('.card:last-child').clientWidth - 40) * 0.62))), ctx = U.fit(cv, S, S);
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, S, S);
    const v = viewBox, X = (a) => ((a - v.x0) / (v.x1 - v.x0)) * S, Y = (b) => S - ((b - v.y0) / (v.y1 - v.y0)) * S;
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(S, Y(0)); ctx.moveTo(X(0), 0); ctx.lineTo(X(0), S); ctx.stroke();
    for (let i = 0; i < D.n; i++) { ctx.fillStyle = PAL[D.y[i]]; ctx.globalAlpha = .72; ctx.beginPath(); ctx.arc(X(proj.P[i * 2]), Y(proj.P[i * 2 + 1]), 3.1, 0, 6.3); ctx.fill(); }
    ctx.globalAlpha = 1;
    if (hoverPt) { const x = X(hoverPt.x), y = Y(hoverPt.y); ctx.strokeStyle = '#17202c'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 7, 0, 6.3); ctx.moveTo(x - 11, y); ctx.lineTo(x + 11, y); ctx.moveTo(x, y - 11); ctx.lineTo(x, y + 11); ctx.stroke(); }
    ctx.fillStyle = '#667180'; ctx.font = '700 10px system-ui'; ctx.textAlign = 'left'; ctx.fillText(k === 2 ? 'latent dimension 2 ↑' : '2-D view of a ' + k + '-D latent vector ↑', 6, 12); ctx.textAlign = 'right'; ctx.fillText(k === 2 ? 'latent dimension 1 →' : 'main direction →', S - 6, S - 6);
    cv._S = S;
    const dc = $('#ae-dec'), dctx = U.fit(dc, 96, 96);
    if (hoverPt) { const o = decodeAt(hoverPt.x, hoverPt.y); U.drawGrid(dctx, o, 8, 0, 0, 12); }
    else { dctx.fillStyle = '#f3f1ea'; dctx.fillRect(0, 0, 96, 96); dctx.fillStyle = '#9aa3b0'; dctx.font = '600 11px system-ui'; dctx.textAlign = 'center'; dctx.fillText('hover the map', 48, 52); }
  }
  function updateVerdict() {
    const el = $('#ae-verd');
    if (!M().epoch) { el.className = 'verdict'; el.textContent = 'Untrained: the latent vector is random. Train for a few epochs, then explore.'; return; }
    if (!hoverPt) { el.className = 'verdict'; el.textContent = 'Hover over the map: each dot is one digit squeezed into the latent vector (colour = true digit, which the model never saw). The decoder turns any point back into an image.'; $('#ae-xy').textContent = ''; return; }
    let best = 1e9; for (let i = 0; i < D.n; i++) best = Math.min(best, Math.hypot(proj.P[i * 2] - hoverPt.x, proj.P[i * 2 + 1] - hoverPt.y));
    $('#ae-xy').textContent = '(' + hoverPt.x.toFixed(2) + ', ' + hoverPt.y.toFixed(2) + ')';
    if (best > 2.6 * proj.nn) { el.className = 'verdict warn'; el.innerHTML = '⚠ <b>A hole.</b> No real digit was encoded anywhere near here, so the decoder is guessing. <b>Rebuilding ≠ generating</b>: nothing guarantees that arbitrary codes give realistic images.'; }
    else { el.className = 'verdict ok'; el.innerHTML = '✓ Close to real digits: the decoder produces something digit-like. Nearby points = similar-looking digits (that is what makes a code <i>useful</i>).'; }
  }
  function drawBars() {
    const row = $('#ae-barsrow'), vals = KS.map((kk) => models[kk].testErr);
    row.innerHTML = KS.map((kk, i) => { const v = vals[i], hgt = v == null ? 2 : Math.max(4, v / (BASE * 1.1) * 100); return `<div class="bar ${v != null ? 'done' : ''}" style="height:${hgt}%"><span>${v == null ? '–' : v.toFixed(4)}</span></div>`; }).join('');
  }
  function drawDraw() {
    const ci = $('#ae-in'), co = $('#ae-out'), cs = 24, a = U.fit(ci, 8 * cs, 8 * cs), b = U.fit(co, 8 * cs, 8 * cs);
    U.drawGrid(a, drawImg, 8, 0, 0, cs, { gap: 1 });
    const { z } = M().encode(drawImg, 1), { o } = M().decode(z, 1);
    U.drawGrid(b, o, 8, 0, 0, cs, { gap: 1 });
    let err = 0; for (let i = 0; i < 64; i++) err += (o[i] - drawImg[i]) ** 2; err /= 64;
    const ref = M().testErr, el = $('#ae-dr-verd');
    if (!ref) { el.className = 'verdict'; el.textContent = 'Train the model first (left), then see how well it reconstructs your drawing.'; return; }
    const ratio = err / ref;
    el.className = 'verdict ' + (ratio > 2 ? 'warn' : 'ok');
    el.innerHTML = 'Your error: <b>' + err.toFixed(3) + '</b> · typical test digit: ' + ref.toFixed(3) + ' (' + ratio.toFixed(1) + '×) — ' + (ratio > 2 ? '<b>unusual!</b> It does not look like the digits this model knows. A high reconstruction error can be used as an <b>anomaly detector</b>.' : 'it looks like something the model knows how to rebuild.');
  }
  function refresh(full) {
    const m = M();
    $('#ae-ep').textContent = m.epoch; $('#ae-te').textContent = m.testErr == null ? '–' : m.testErr.toFixed(4);
    drawCurve(); drawRec(); computeLatent(); drawScatter(); updateVerdict(); drawBars(); if (tab === 'draw') drawDraw();
    $('#ae-train').textContent = training ? '❚❚ Pause' : (m.epoch >= MAXEP ? 'Max epochs' : '▶ Train'); $('#ae-train').disabled = m.epoch >= MAXEP && !training;
    $('#ae-note').textContent = k > 2 ? 'With k = ' + k + ' the latent vector has ' + k + ' numbers; the map shows its two most informative directions, so points can look more overlapped than they really are.' : 'With k = 2 the whole latent vector fits on a plane — that is why we can draw it directly.';
  }

  /* ---------- training loop ---------- */
  // Time-based pacing (same speed on any machine): slow enough to watch the first epochs, then faster.
  let lastT = 0;
  function loop(now) {
    if (!training) return;
    const m = M(), rate = m.epoch < 20 ? 10 : (m.epoch < 80 ? 25 : 80);   // epochs per second
    if (!lastT) lastT = now;
    let n = Math.floor((now - lastT) * rate / 1000);
    if (n >= 1) {
      lastT += n * 1000 / rate; n = Math.min(n, 30);          // catch up if frames are slow (battery saver, busy laptop)
      const t0 = performance.now(); let did = 0;
      while (did < n && performance.now() - t0 < 45 && m.epoch < MAXEP) { m.trainEpoch(); did++; }
      if (did) refresh();
    }
    if (M().epoch >= MAXEP) training = false;
    if (training) raf = requestAnimationFrame(loop); else refresh();
  }
  $('#ae-train').addEventListener('click', () => { training = !training; lastT = 0; if (training) raf = requestAnimationFrame(loop); else refresh(); });
  $('#ae-step').addEventListener('click', () => { training = false; M().trainEpoch(); refresh(); });
  $('#ae-reset').addEventListener('click', () => { training = false; models[k] = new AE(k); viewBox = null; refresh(); });
  $('#ae-new').addEventListener('click', () => { pickExamples(); drawRec(); });
  root.querySelectorAll('#ae-k button').forEach((b) => b.addEventListener('click', () => {
    root.querySelectorAll('#ae-k button').forEach((x) => x.classList.remove('on')); b.classList.add('on');
    training = false; k = +b.dataset.k; viewBox = null; hoverPt = null; refresh();
  }));
  root.querySelectorAll('#ae-tabs button').forEach((b) => b.addEventListener('click', () => {
    root.querySelectorAll('#ae-tabs button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); tab = b.dataset.t;
    $('#ae-lat').style.display = tab === 'lat' ? 'block' : 'none'; $('#ae-draw').style.display = tab === 'draw' ? 'block' : 'none'; $('#ae-bars').style.display = tab === 'bars' ? 'block' : 'none';
    refresh();
  }));

  /* latent hover */
  const sc = $('#ae-sc');
  function scMove(e) {
    if (!viewBox) return; const p = U.pos(sc, e), S = sc._S, v = viewBox;
    hoverPt = { x: v.x0 + (p.x / S) * (v.x1 - v.x0), y: v.y0 + ((S - p.y) / S) * (v.y1 - v.y0) }; drawScatter(); updateVerdict();
  }
  sc.addEventListener('pointermove', scMove); sc.addEventListener('pointerdown', scMove);
  sc.addEventListener('pointerleave', () => { hoverPt = null; drawScatter(); updateVerdict(); });

  /* draw-your-own */
  const ci = $('#ae-in'); let lastCell = null;
  function dcell(e) { const p = U.pos(ci, e); return { r: U.clamp(Math.floor(p.y / 24), 0, 7), c: U.clamp(Math.floor(p.x / 24), 0, 7) }; }
  ci.addEventListener('pointerdown', (e) => { ci.setPointerCapture(e.pointerId); drawing = true; const c = dcell(e); drawImg[c.r * 8 + c.c] = e.shiftKey ? 0 : 1; lastCell = c; drawDraw(); });
  ci.addEventListener('pointermove', (e) => { if (!drawing) return; const c = dcell(e); drawImg[c.r * 8 + c.c] = e.shiftKey ? 0 : 1; lastCell = c; drawDraw(); });
  ci.addEventListener('pointerup', () => { drawing = false; });
  $('#ae-dr-digit').addEventListener('click', () => { drawImg = Float64Array.from(Xte.subarray(Math.floor(Math.random() * NTE) * 64).subarray(0, 64)); drawDraw(); });
  $('#ae-dr-noise').addEventListener('click', () => { for (let i = 0; i < 64; i++) if (Math.random() < .2) drawImg[i] = Math.random() < .5 ? 1 : 0; drawDraw(); });
  $('#ae-dr-clear').addEventListener('click', () => { drawImg = new Float64Array(64); drawDraw(); });
  $('#ae-dr-cross').addEventListener('click', () => { drawImg = new Float64Array(64); for (let i = 0; i < 8; i++) { drawImg[i * 8 + i] = 1; drawImg[i * 8 + 7 - i] = 1; } drawDraw(); });
  drawImg = Float64Array.from(Xte.subarray(0, 64));

  $('#ae-leg').innerHTML = PAL.map((c, i) => `<span><i style="background:${c}"></i>${i}</span>`).join('');
  Deck.on('aelab', (s, resized) => { if (!resized) { training = false; } refresh(); });

  /* ---------------- PCA baseline (for the "autoencoder vs PCA" slide) ---------------- */
  (function pcaWidget() {
    const rootP = document.getElementById('w-pca'); if (!rootP) return;
    const n = NTRAIN, C = new Float64Array(64 * 64);
    for (let i = 0; i < n; i++) for (let a = 0; a < 64; a++) { const da = Xtr[i * 64 + a] - mean[a]; for (let b = 0; b < 64; b++) C[a * 64 + b] += da * (Xtr[i * 64 + b] - mean[b]) / n; }
    const comps = [], Cc = Float64Array.from(C);
    for (let c = 0; c < 16; c++) {
      let v = new Float64Array(64).map((_, j) => Math.sin(j * 1.3 + c + 1));
      for (let it = 0; it < 120; it++) { const w = new Float64Array(64); for (let a = 0; a < 64; a++) { let t = 0; for (let b = 0; b < 64; b++) t += Cc[a * 64 + b] * v[b]; w[a] = t; } const nr = Math.hypot(...w) || 1; v = w.map((x) => x / nr); }
      let lam = 0; for (let a = 0; a < 64; a++) { let t = 0; for (let b = 0; b < 64; b++) t += Cc[a * 64 + b] * v[b]; lam += t * v[a]; }
      for (let a = 0; a < 64; a++) for (let b = 0; b < 64; b++) Cc[a * 64 + b] -= lam * v[a] * v[b];
      comps.push(v);
    }
    function recon(x, k) { const o = Float64Array.from(mean); for (let j = 0; j < k; j++) { let p = 0; for (let a = 0; a < 64; a++) p += (x[a] - mean[a]) * comps[j][a]; for (let a = 0; a < 64; a++) o[a] += p * comps[j][a]; } return o; }
    const err = []; for (let k = 1; k <= 16; k++) { let e = 0; for (let i = 0; i < NTE; i++) { const x = Xte.subarray(i * 64, i * 64 + 64), r = recon(x, k); for (let a = 0; a < 64; a++) e += (r[a] - x[a]) ** 2; } err[k] = e / (NTE * 64); }
    rootP.innerHTML = `
      <div class="grid g-side" style="align-items:start">
        <div>
          <label class="ctl" style="max-width:28rem"><span>Number of values kept in the latent vector: <span class="val" id="pc2-k">2</span></span><input type="range" id="pc2-s" min="1" max="16" value="2"></label>
          <div class="mt"><canvas id="pc2-rec" class="cv"></canvas><div class="tiny" style="margin-top:.2rem">Top: original digits. Bottom: rebuilt by <b>PCA</b> from the same number of values.</div></div>
        </div>
        <div>
          <canvas id="pc2-ch" class="cv"></canvas>
          <div class="tiny" style="margin-top:.2rem"><span style="color:#2859e0;font-weight:800">●</span> PCA test error · <span style="color:var(--gen);font-weight:800">●</span> the autoencoder you trained (only for k you trained on the previous slide)</div>
          <div class="callout mt" id="pc2-msg" style="font-size:.9rem"></div>
        </div>
      </div>`;
    const q = (x) => rootP.querySelector(x);
    function draw() {
      const k = +q('#pc2-s').value; q('#pc2-k').textContent = k;
      const px = 6, gap = 6, w = 8 * (8 * px + gap) - gap, ctx = U.fit(q('#pc2-rec'), w, 2 * 8 * px + 8); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, 2 * 8 * px + 8);
      for (let i = 0; i < 8; i++) { const x = Xte.subarray(i * 64, i * 64 + 64), r = recon(x, k).map((v) => U.clamp(v, 0, 1)); U.drawGrid(ctx, x, 8, i * (8 * px + gap), 0, px); U.drawGrid(ctx, r, 8, i * (8 * px + gap), 8 * px + 8, px); }
      const W = Math.min(rootP.clientWidth * 0.42, 420), H = 190, c2 = U.fit(q('#pc2-ch'), W, H); c2.fillStyle = '#fff'; c2.fillRect(0, 0, W, H);
      const X = (kk) => 36 + ((kk - 1) / 15) * (W - 50), ymax = 0.075, Y = (v) => H - 24 - (U.clamp(v, 0, ymax) / ymax) * (H - 36);
      c2.strokeStyle = '#2859e0'; c2.lineWidth = 2.5; c2.beginPath(); for (let kk = 1; kk <= 16; kk++) { kk === 1 ? c2.moveTo(X(kk), Y(err[kk])) : c2.lineTo(X(kk), Y(err[kk])); } c2.stroke();
      c2.fillStyle = '#2859e0'; for (let kk = 1; kk <= 16; kk++) { c2.beginPath(); c2.arc(X(kk), Y(err[kk]), kk === k ? 6 : 3, 0, 6.3); c2.fill(); }
      c2.fillStyle = '#e8712d'; KS.forEach((kk) => { const e = models[kk].testErr; if (e != null) { c2.beginPath(); c2.arc(X(kk), Y(e), 6, 0, 6.3); c2.fill(); } });
      c2.fillStyle = '#667180'; c2.font = '600 10px system-ui'; c2.textAlign = 'center'; c2.fillText('latent size k →', W / 2, H - 1); c2.textAlign = 'right'; c2.fillText('test error', 30, 12);
      [1, 4, 8, 12, 16].forEach((kk) => { c2.textAlign = 'center'; c2.fillText(kk, X(kk), H - 13); });
      const ae = KS.includes(k) && models[k].testErr != null ? models[k].testErr : null;
      q('#pc2-msg').innerHTML = 'PCA with k = ' + k + ': test error <b>' + err[k].toFixed(4) + '</b>. ' + (ae != null ? 'Your autoencoder with k = ' + k + ': <b>' + ae.toFixed(4) + '</b>' + (ae < err[k] ? ' — lower: the non-linear layers capture more.' : ' — similar or higher: a linear autoencoder would match PCA; more training or non-linearity helps.') : 'Train the autoencoder lab (k = 2, 4, 8, 16) to see its points here.');
    }
    q('#pc2-s').addEventListener('input', draw);
    Deck.on('pca', draw);
  })();

  window.__AE = { models, AE };   // handy for the browser console when debugging
})();
