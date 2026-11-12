/* A real, tiny GAN trained in the browser on the 8x8 handwritten digits (one class at a time).
   Generator: z (4 numbers) -> 24 tanh units -> 64 pixels (sigmoid).  Discriminator: 64 pixels -> 24 leaky-ReLU units -> 1 (sigmoid).
   Plain SGD-style training with Adam (lr 0.002, beta1 0.5), batches of 32, non-saturating generator loss. Manual backpropagation.
   Used by two slides:  #w-ganarch (architecture, before / after training)  and  #w-gantrain (how the images improve over time). */
(function () {
  'use strict';
  const ZD = 4, H = 24, B = 32;

  /* ---------- tiny linear algebra on flat Float64Arrays ---------- */
  function mm(a, n, k, b, m) { const o = new Float64Array(n * m); for (let i = 0; i < n; i++) for (let t = 0; t < k; t++) { const v = a[i * k + t]; if (v === 0) continue; for (let j = 0; j < m; j++) o[i * m + j] += v * b[t * m + j]; } return o; }
  function mtA(a, n, k, b, m) { const o = new Float64Array(k * m); for (let i = 0; i < n; i++) for (let t = 0; t < k; t++) { const v = a[i * k + t]; for (let j = 0; j < m; j++) o[t * m + j] += v * b[i * m + j]; } return o; }
  function mBt(a, n, k, b, m) { const o = new Float64Array(n * m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { let s = 0; for (let t = 0; t < k; t++) s += a[i * k + t] * b[j * k + t]; o[i * m + j] = s; } return o; }
  const sig = (x) => 1 / (1 + Math.exp(-x));
  const colSum = (a, n, m) => { const o = new Float64Array(m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) o[j] += a[i * m + j]; return o; };

  function Adam(params, lr, b1) { const m = params.map((p) => new Float64Array(p.length)), v = params.map((p) => new Float64Array(p.length)); let t = 0; return { step(grads) { t++; params.forEach((p, i) => { const g = grads[i]; for (let j = 0; j < p.length; j++) { m[i][j] = b1 * m[i][j] + (1 - b1) * g[j]; v[i][j] = 0.999 * v[i][j] + 0.001 * g[j] * g[j]; p[j] -= lr * (m[i][j] / (1 - Math.pow(b1, t))) / (Math.sqrt(v[i][j] / (1 - Math.pow(0.999, t))) + 1e-8); } }); } }; }

  function MiniGAN(cls, seed) {
    const rnd = U.mulberry32(seed), gs = () => U.gauss(rnd), Dg = U.digits(), real = [];
    for (let i = 0; i < Dg.n; i++) if (cls < 0 || Dg.y[i] === cls) real.push(Dg.X.subarray(i * 64, i * 64 + 64));
    const rn = (n, s) => { const a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = gs() * s; return a; };
    const G = [rn(ZD * H, Math.sqrt(2 / (ZD + H))), new Float64Array(H), rn(H * 64, Math.sqrt(2 / (H + 64))), new Float64Array(64)];
    const D = [rn(64 * H, Math.sqrt(2 / (64 + H))), new Float64Array(H), rn(H, Math.sqrt(2 / (H + 1))), new Float64Array(1)];
    const oG = Adam(G, 2e-3, 0.5), oD = Adam(D, 2e-3, 0.5);
    const noise = (n) => { const z = new Float64Array(n * ZD); for (let i = 0; i < z.length; i++) z[i] = gs(); return z; };
    const batchReal = (n) => { const x = new Float64Array(n * 64); for (let i = 0; i < n; i++) x.set(real[Math.floor(rnd() * real.length)], i * 64); return x; };

    function gF(z, n) { const a = mm(z, n, ZD, G[0], H), h = new Float64Array(a.length); for (let i = 0; i < n; i++) for (let j = 0; j < H; j++) { a[i * H + j] += G[1][j]; h[i * H + j] = Math.tanh(a[i * H + j]); } const o = mm(h, n, H, G[2], 64); for (let i = 0; i < n; i++) for (let j = 0; j < 64; j++) o[i * 64 + j] = sig(o[i * 64 + j] + G[3][j]); return { h, o }; }
    function dF(x, n) { const a = mm(x, n, 64, D[0], H), h = new Float64Array(a.length); for (let i = 0; i < n; i++) for (let j = 0; j < H; j++) { a[i * H + j] += D[1][j]; h[i * H + j] = a[i * H + j] > 0 ? a[i * H + j] : 0.2 * a[i * H + j]; } const p = new Float64Array(n); for (let i = 0; i < n; i++) { let s = D[3][0]; for (let j = 0; j < H; j++) s += h[i * H + j] * D[2][j]; p[i] = sig(s); } return { a, h, p }; }
    function dB(x, n, f, dS) { // dS: d loss / d logit, length n
      const gW2 = mtA(f.h, n, H, dS, 1), gb2 = new Float64Array([dS.reduce((s, v) => s + v, 0)]), da = new Float64Array(n * H);
      for (let i = 0; i < n; i++) for (let j = 0; j < H; j++) da[i * H + j] = dS[i] * D[2][j] * (f.a[i * H + j] > 0 ? 1 : 0.2);
      return { g: [mtA(x, n, 64, da, H), colSum(da, n, H), gW2, gb2], dx: mBt(da, n, H, D[0], 64) };
    }
    const add = (A, Bq) => A.map((a, i) => a.map((v, j) => v + Bq[i][j]));

    const gan = { cls, step: 0, hist: { lD: [], lG: [], pr: [], pf: [] }, snaps: [], last: {}, ZD };
    const fixedZ = noise(8);
    gan.fixedZ = fixedZ;
    gan.sample = (z, n) => gF(z, n).o;
    gan.score = (x, n) => dF(x, n).p;
    gan.realImgs = (n) => batchReal(n);
    gan.noise = noise;

    gan.dStep = function () {
      const xr = batchReal(B), z = noise(B), xf = gF(z, B).o, fr = dF(xr, B), ff = dF(xf, B);
      let lr_ = 0, lf = 0; for (let i = 0; i < B; i++) { lr_ -= Math.log(fr.p[i] + 1e-8); lf -= Math.log(1 - ff.p[i] + 1e-8); } lr_ /= B; lf /= B;
      const g1 = dB(xr, B, fr, fr.p.map((p) => (p - 1) / B)), g2 = dB(xf, B, ff, ff.p.map((p) => p / B));
      oD.step(add(g1.g, g2.g));
      gan.last.d = { xr: xr.slice(0, 6 * 64), xf: xf.slice(0, 6 * 64), pr: Array.from(fr.p.slice(0, 6)), pf: Array.from(ff.p.slice(0, 6)), mr: fr.p.reduce((s, v) => s + v, 0) / B, mf: ff.p.reduce((s, v) => s + v, 0) / B, lr: lr_, lf: lf, loss: lr_ + lf };
      return gan.last.d;
    };
    gan.gStep = function () {
      const z = noise(B), fg = gF(z, B), ff = dF(fg.o, B);
      let lg = 0; for (let i = 0; i < B; i++) lg -= Math.log(ff.p[i] + 1e-8); lg /= B;
      const gd = dB(fg.o, B, ff, ff.p.map((p) => (p - 1) / B)), dx = gd.dx;
      const dpre = new Float64Array(B * 64); for (let i = 0; i < B * 64; i++) dpre[i] = dx[i] * fg.o[i] * (1 - fg.o[i]);
      const dh = mBt(dpre, B, 64, G[2], H), da = new Float64Array(B * H); for (let i = 0; i < B * H; i++) da[i] = dh[i] * (1 - fg.h[i] * fg.h[i]);
      const before = Array.from(fg.o.slice(0, 64)), grad = Array.from(dx.slice(0, 64)).map((v) => -v * B);
      oG.step([mtA(z, B, ZD, da, H), colSum(da, B, H), mtA(fg.h, B, H, dpre, 64), colSum(dpre, B, 64)]);
      const after = Array.from(gF(z.slice(0, ZD), 1).o), pAfter = dF(Float64Array.from(after), 1).p[0];
      gan.last.g = { before, grad, after, pBefore: ff.p[0], pAfter, lossG: lg, mf: ff.p.reduce((s, v) => s + v, 0) / B };
      return gan.last.g;
    };
    const SNAP = [0, 25, 100, 300, 1000, 2000];
    gan.snap = () => { if (SNAP.includes(gan.step)) gan.snaps.push({ step: gan.step, img: Array.from(gF(fixedZ.slice(0, ZD), 1).o) }); };
    gan.iter = function () { const d = gan.dStep(), g = gan.gStep(); gan.step++; gan.hist.lD.push(d.loss); gan.hist.lG.push(g.lossG); gan.hist.pr.push(d.mr); gan.hist.pf.push(d.mf); gan.snap(); return { d, g }; };
    gan.train = (n) => { for (let i = 0; i < n; i++) gan.iter(); };
    gan.snap();
    return gan;
  }
  window.MiniGAN = MiniGAN;

  function paint(cv, arr, cs, mode, mx) {
    const ctx = U.fit(cv, 8 * cs, 8 * cs);
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) { const v = arr[r * 8 + c]; ctx.fillStyle = mode === 'div' ? U.div(v / (mx || 1)) : U.ink(v); ctx.fillRect(c * cs, r * cs, cs - (mode === 'div' ? 0.6 : 0), cs - (mode === 'div' ? 0.6 : 0)); }
    cv.style.border = '1px solid rgba(23,32,44,.25)';
  }
  const sc = (p) => `<span style="color:${p > 0.5 ? '#1f8a4c' : '#8e44ad'};font-weight:800">${p.toFixed(2)}</span>`;

  /* =================================================================
     ARCHITECTURE with real pictures (before / after training)
     ================================================================= */
  (function arch() {
    const root = document.getElementById('w-ganarch'); if (!root) return;
    let gan = null, stage = 0;
    root.innerHTML = `
      <div class="row" style="margin-bottom:.5rem;gap:.7rem"><button class="btn" id="ga-un">Untrained G</button><button class="btn primary" id="ga-tr">Train 1,500 steps ▸</button><button class="btn small" id="ga-nz">New noise &amp; new real images</button><span class="small" id="ga-st"></span></div>
      <div class="card flat" style="padding:.5rem"><svg id="ga-svg" viewBox="0 0 900 340" style="width:100%;height:auto;display:block;image-rendering:pixelated" role="img" aria-label="GAN diagram: noise vectors go into the generator, which produces fake images; real images come from the dataset; both go into the discriminator, which says real or fake; the discriminator's verdict is fed back to the generator."></svg></div>`;
    const $ = (s) => root.querySelector(s);
    const url = (arr, blur) => { const c = document.createElement('canvas'); c.width = c.height = 8; const x = c.getContext('2d'); for (let i = 0; i < 64; i++) { const v = Math.round(255 * (1 - U.clamp(arr[i], 0, 1))); x.fillStyle = `rgb(${v},${v},${v})`; x.fillRect(i % 8, Math.floor(i / 8), 1, 1); } return c.toDataURL(); };
    const scTxt = (p) => `<tspan fill="${p > 0.5 ? '#1f8a4c' : '#8e44ad'}" font-weight="800">${p.toFixed(2)}</tspan>`;
    function show() {
      const n = 4, z = gan.noise(n), fk = gan.sample(z, n), rl = gan.realImgs(n), sf = gan.score(fk, n), sr = gan.score(rl, n);
      let s = `<defs><marker id="gx" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#667180"/></marker><marker id="go" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#e8712d"/></marker></defs>`;
      // top lane: noise -> G -> fakes
      s += `<text x="20" y="58" font-size="12" font-weight="800" fill="#3d4856">random noise z</text>`;
      for (let i = 0; i < n; i++) for (let j = 0; j < ZD; j++) s += `<rect x="${20 + i * 22}" y="${68 + j * 14}" width="20" height="13" fill="${U.div(z[i * ZD + j] / 2.2)}" stroke="#17202c33"/>`;
      s += `<text x="20" y="140" font-size="10" fill="#667180">4 numbers each</text>`;
      s += `<line x1="118" y1="96" x2="166" y2="96" stroke="#667180" stroke-width="2.5" marker-end="url(#gx)"/>`;
      s += `<rect x="166" y="56" width="136" height="80" rx="12" fill="#fdeadd" stroke="#e8712d" stroke-width="3"/><text x="234" y="92" text-anchor="middle" font-weight="800" font-size="15" fill="#a14a12">Generator G</text><text x="234" y="112" text-anchor="middle" font-size="10.5" fill="#a14a12">noise → image</text>`;
      s += `<line x1="302" y1="96" x2="344" y2="96" stroke="#667180" stroke-width="2.5" marker-end="url(#gx)"/>`;
      s += `<text x="350" y="58" font-size="12" font-weight="800" fill="#3d4856">fake images G(z)</text>`;
      for (let i = 0; i < n; i++) s += `<image href="${url(fk.subarray(i * 64, i * 64 + 64))}" x="${350 + i * 60}" y="68" width="54" height="54" style="image-rendering:pixelated"/><rect x="${350 + i * 60}" y="68" width="54" height="54" fill="none" stroke="#17202c55"/><text x="${377 + i * 60}" y="140" text-anchor="middle" font-size="12">${scTxt(sf[i])}</text>`;
      // bottom lane: dataset -> real images
      s += `<rect x="14" y="214" width="140" height="56" rx="10" fill="#e9e6dc" stroke="#b9b49f"/><text x="84" y="238" text-anchor="middle" font-weight="800" font-size="12.5">training dataset</text><text x="84" y="256" text-anchor="middle" font-size="10.5" fill="#555">1,000 real digits</text>`;
      s += `<line x1="152" y1="242" x2="344" y2="242" stroke="#667180" stroke-width="2.5" marker-end="url(#gx)"/><text x="248" y="232" text-anchor="middle" font-size="11" fill="#667180">random batch</text>`;
      s += `<text x="350" y="204" font-size="12" font-weight="800" fill="#3d4856">real images x</text>`;
      for (let i = 0; i < n; i++) s += `<image href="${url(rl.subarray(i * 64, i * 64 + 64))}" x="${350 + i * 60}" y="214" width="54" height="54" style="image-rendering:pixelated"/><rect x="${350 + i * 60}" y="214" width="54" height="54" fill="none" stroke="#17202c55"/><text x="${377 + i * 60}" y="286" text-anchor="middle" font-size="12">${scTxt(sr[i])}</text>`;
      // converge into D
      s += `<path d="M596 95 H632 V158 H654" fill="none" stroke="#667180" stroke-width="2.5" marker-end="url(#gx)"/><path d="M596 241 H632 V184 H654" fill="none" stroke="#667180" stroke-width="2.5" marker-end="url(#gx)"/>`;
      s += `<rect x="658" y="130" width="200" height="82" rx="12" fill="#e6edfd" stroke="#2859e0" stroke-width="3"/><text x="758" y="166" text-anchor="middle" font-weight="800" font-size="16" fill="#1c3f9e">Discriminator D</text><text x="758" y="186" text-anchor="middle" font-size="10.5" fill="#1c3f9e">image → probability “real”</text>`;
      s += `<line x1="758" y1="214" x2="758" y2="256" stroke="#667180" stroke-width="2.5" marker-end="url(#gx)"/><rect x="658" y="260" width="200" height="70" rx="10" fill="#fff" stroke="#2859e0" stroke-width="2"/><text x="758" y="280" text-anchor="middle" font-weight="800" font-size="14">real or fake?</text><text x="758" y="298" text-anchor="middle" font-size="10" fill="#667180">the number under each image</text><text x="758" y="312" text-anchor="middle" font-size="10" fill="#667180">is D’s probability “real”</text>`;
      // feedback
      s += `<path d="M758 128 V24 H235 V54" fill="none" stroke="#e8712d" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#go)"/><text x="496" y="17" text-anchor="middle" font-size="11.5" font-weight="800" fill="#e8712d">feedback: D’s verdict is the only signal G receives (G never sees real images)</text>`;
      $('#ga-svg').innerHTML = s;
      $('#ga-st').textContent = gan.step === 0 ? 'Untrained: G outputs blobs and D has no opinion yet (≈ 0.5).' : 'After ' + gan.step.toLocaleString('en-US') + ' training steps: the fakes look like zeros, and D can no longer tell (≈ 0.5).';
    }
    function reset() { gan = MiniGAN(0, 7); show(); }
    $('#ga-un').addEventListener('click', reset);
    $('#ga-tr').addEventListener('click', () => { if (gan.step === 0) { gan.train(1500); show(); } });
    $('#ga-nz').addEventListener('click', show);
    Deck.on('ganarch', () => { reset(); });
  })();

  /* =================================================================
     HOW THE IMAGES GET BETTER OVER TIME
     ================================================================= */
  (function train() {
    const root = document.getElementById('w-gantrain'); if (!root) return;
    let gan = null, cls = 0, timer = null;
    const MARKS = [0, 25, 100, 300, 1000, 2000];
    root.innerHTML = `
      <div class="row" style="margin-bottom:.6rem;gap:.7rem">
        <label class="ctl" style="margin:0">Real data<div class="seg" id="gt-c"><button data-c="0" class="on">zeros</button><button data-c="3">threes</button><button data-c="7">sevens</button></div></label>
        <span class="spacer"></span><button class="btn primary" id="gt-run">▶ Train</button><button class="btn small" id="gt-reset">Reset</button>
      </div>
      <div class="gt1">
        <div class="gt-p"><div class="gt-h">The same noise vector z, over training <span class="tiny" style="font-weight:500">(number = training steps)</span></div><div class="gt-imgs" id="gt-snaps"></div></div>
        <div class="gt-p"><div class="gt-h">Sixteen new fakes <span id="gt-n" style="float:right;font-weight:600;color:var(--muted)"></span></div><div class="gt-imgs gt-wrap" id="gt-smp"></div>
          <div class="tiny" id="gt-l" style="margin-top:.3rem"></div></div>
        <div class="gt-p"><div class="gt-h">For comparison: real digits</div><div class="gt-imgs gt-wrap" id="gt-real"></div></div>
      </div>`;
    const $ = (s) => root.querySelector(s);
    function tiles(box, arr, n, cs, labels) { box.innerHTML = ''; for (let i = 0; i < n; i++) { const w = document.createElement('div'); w.innerHTML = '<canvas></canvas>' + (labels ? '<div class="tiny" style="text-align:center">' + labels[i] + '</div>' : ''); box.appendChild(w); paint(w.querySelector('canvas'), arr.subarray ? arr.subarray(i * 64, (i + 1) * 64) : arr[i], cs); } }
    function views() {
      const sn = gan.snaps, box = $('#gt-snaps'); box.innerHTML = ''; sn.forEach((s) => { const w = document.createElement('div'); w.innerHTML = '<canvas></canvas><div class="tiny" style="text-align:center">' + s.step + '</div>'; box.appendChild(w); paint(w.querySelector('canvas'), s.img, 8); });
      const z = gan.noise(16), smp = gan.sample(z, 16); tiles($('#gt-smp'), smp, 16, 6);
      $('#gt-n').textContent = gan.step.toLocaleString('en-US') + ' steps';
      const h = gan.hist, n = h.lD.length; if (n) { const a = (arr) => arr.slice(-20).reduce((x, y) => x + y, 0) / Math.min(20, arr.length); $('#gt-l').innerHTML = 'Discriminator loss ' + a(h.lD).toFixed(2) + ' · generator loss ' + a(h.lG).toFixed(2) + ' (recent average; D’s loss tends to 1.39 when it can only guess)'; } else $('#gt-l').textContent = '';
    }
    function reset() { stop(); gan = MiniGAN(cls, 3 + cls); const r = gan.realImgs(16); tiles($('#gt-real'), r, 16, 6); views(); }
    function stop() { clearInterval(timer); timer = null; $('#gt-run').textContent = '▶ Train'; }
    $('#gt-run').addEventListener('click', () => { if (timer) { stop(); return; } $('#gt-run').textContent = '❚❚ Pause'; timer = setInterval(() => { gan.train(12); views(); if (gan.step >= 2400) stop(); }, 60); });
    $('#gt-reset').addEventListener('click', reset);
    root.querySelectorAll('#gt-c button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#gt-c button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); cls = +b.dataset.c; reset(); }));
    Deck.on('gantrain', () => { if (!gan) reset(); else { stop(); views(); } });
  })();
})();
