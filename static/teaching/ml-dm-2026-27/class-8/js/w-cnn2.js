/* Widgets for the CNN part (second batch)
   1. kernel   : "what is a convolution" — a worked patch × kernel example with designed vs learned kernels (slide "convdef")
   2. poolnum  : pooling / strided convolution animated on a grid of numbers                              (slide "pooling")
   3. hier     : what units of trained networks respond to, with real images                              (slide "hier")
   4. dropout  : randomly switch units off during training                                                (slide "dropout") */
(function () {
  'use strict';

  /* Filters learned by a small CNN (conv 6 filters 3x3 -> ReLU -> dense 10) trained for 40 epochs on 800 of the
     1,000 digits (own numpy implementation, 97 % accuracy on the other 200). Exported once, embedded here. */
  const LEARNED = [
    [1.37, -0.06, -0.11, 0.47, -0.27, 0.09, -1.5, -0.51, 0.55],
    [-1.5, -0.71, -0.29, 0.09, 0.56, 0.23, 1.18, 0.4, 0.66],
    [0.77, 0.27, -0.79, 0.07, 1.56, 0.43, -1.08, 0.44, 0.42],
    [0.21, 2.22, 0.32, 0.63, 0.75, 0.13, -0.75, -0.17, -0.55],
    [-0.02, -0.42, -0.1, 0.18, 0.07, 0.26, 0.19, 1.29, 0.65],
    [0.04, -0.34, -0.15, -0.27, 0.82, 0.55, 0.62, 1.16, 0.3],
  ];
  window.LEARNED_FILTERS = LEARNED;

  /* =================================================================
     1. WHAT IS A CONVOLUTION: patch x kernel, designed vs learned
     ================================================================= */
  (function kernel() {
    const root = document.getElementById('w-kernel'); if (!root) return;
    const D = U.digits();
    const SOBEL = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    let di = 7, r0 = 3, c0 = 2, which = 'designed';
    const kers = () => (which === 'designed' ? SOBEL : LEARNED[3]);
    root.innerHTML = `
      <div class="row" style="margin-bottom:.5rem;gap:.7rem">
        <div class="seg" id="kn-w"><button data-w="designed" class="on">Kernel designed by hand (Sobel)</button><button data-w="learned">Kernel learned by a CNN</button></div>
        <button class="btn small" id="kn-next">Another digit</button>
        <span class="small">Click the digit to move the 3×3 window.</span>
      </div>
      <div class="row top" style="gap:1.4rem;align-items:center;flex-wrap:wrap">
        <div><div class="tiny" style="font-weight:800">IMAGE x (patch highlighted)</div><canvas id="kn-img" class="cv" style="cursor:pointer"></canvas></div>
        <div class="kn-op">×</div>
        <div><div class="tiny" style="font-weight:800">KERNEL w (9 weights)</div><canvas id="kn-ker" class="cv"></canvas></div>
        <div class="kn-op">=</div>
        <div class="calc" id="kn-calc" style="min-width:15rem"></div>
      </div>`;
    const $ = (s) => root.querySelector(s);
    const cvI = $('#kn-img'), cvK = $('#kn-ker');
    const px = (r, c) => D.X[di * 64 + r * 8 + c];
    const f = (v) => (Math.abs(v - Math.round(v)) < 0.005 ? String(Math.round(v)) : v.toFixed(1));
    function draw() {
      const cs = 38, ctx = U.fit(cvI, 8 * cs, 8 * cs);
      U.drawGrid(ctx, D.X.subarray(di * 64, di * 64 + 64), 8, 0, 0, cs);
      ctx.strokeStyle = 'rgba(23,32,44,.1)'; for (let i = 1; i < 8; i++) { ctx.beginPath(); ctx.moveTo(i * cs, 0); ctx.lineTo(i * cs, 8 * cs); ctx.moveTo(0, i * cs); ctx.lineTo(8 * cs, i * cs); ctx.stroke(); }
      ctx.strokeStyle = '#2859e0'; ctx.lineWidth = 3.5; ctx.strokeRect(c0 * cs + 1.5, r0 * cs + 1.5, 3 * cs - 3, 3 * cs - 3);
      const k = kers(), ks = 52, c2 = U.fit(cvK, 3 * ks, 3 * ks), m = Math.max(...k.map(Math.abs));
      for (let i = 0; i < 9; i++) {
        const r = Math.floor(i / 3), c = i % 3; c2.fillStyle = U.div(k[i] / m); c2.fillRect(c * ks, r * ks, ks - 1, ks - 1);
        c2.fillStyle = Math.abs(k[i] / m) > .6 ? '#fff' : '#17202c'; c2.font = '700 15px ui-monospace,Menlo,monospace'; c2.textAlign = 'center'; c2.textBaseline = 'middle'; c2.fillText(f(k[i]), c * ks + ks / 2, r * ks + ks / 2);
      }
      let sum = 0, rows = '';
      for (let i = 0; i < 3; i++) {
        const parts = [];
        for (let j = 0; j < 3; j++) { const p = Math.round(px(r0 + i, c0 + j) * 16), w = k[i * 3 + j]; sum += (p / 16) * w; parts.push(p + '×' + (w < 0 ? '(' + f(w) + ')' : f(w))); }
        rows += parts.join(' + ') + (i < 2 ? ' +' : '') + '<br>';
      }
      $('#kn-calc').innerHTML = '<span class="tiny">pixel values (0–16) × weights, summed:</span><br>' + rows + '= <span class="res">' + (sum * 16).toFixed(1) + '</span>';
    }
    cvI.addEventListener('pointerdown', (e) => { const p = U.pos(cvI, e), cs = cvI.clientWidth / 8; c0 = U.clamp(Math.floor(p.x / cs) - 1, 0, 5); r0 = U.clamp(Math.floor(p.y / cs) - 1, 0, 5); draw(); });
    root.querySelectorAll('#kn-w button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#kn-w button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); which = b.dataset.w; draw(); }));
    $('#kn-next').addEventListener('click', () => { di = (di * 7 + 13) % D.n; draw(); });
    Deck.on('convdef', draw);
  })();

  /* =================================================================
     2. POOLING / STRIDE ON NUMBERS
     ================================================================= */
  (function poolnum() {
    const root = document.getElementById('w-poolnum'); if (!root) return;
    const IN = [[1, 3, 2, 1, 0, 2], [4, 6, 5, 1, 2, 3], [2, 1, 0, 7, 8, 1], [0, 2, 9, 3, 4, 5], [3, 1, 2, 6, 1, 0], [5, 4, 1, 0, 2, 7]];
    const KER = [[1, -1], [1, -1]];            // a 2x2 "learned" kernel for the strided-convolution option
    let op = 'max', stride = 2, step = 0, timer = null;
    root.innerHTML = `
      <div class="row" style="margin-bottom:.6rem;gap:.9rem">
        <label class="ctl" style="margin:0">Operation<div class="seg" id="pn-op"><button data-o="max" class="on">Max pooling</button><button data-o="mean">Mean pooling</button><button data-o="conv">Convolution (2×2 kernel)</button></div></label>
        <label class="ctl" style="margin:0">Stride<div class="seg" id="pn-st"><button data-s="1">1</button><button data-s="2" class="on">2</button></div></label>
        <span class="spacer"></span>
        <button class="btn" id="pn-back">◂ Back</button><button class="btn primary" id="pn-next">Next step ▸</button><button class="btn" id="pn-play">▶ Play</button><button class="btn small" id="pn-reset">Reset</button>
      </div>
      <div class="row top" style="gap:1.4rem;align-items:flex-start;flex-wrap:wrap">
        <div><div class="tiny" style="font-weight:800;margin-bottom:.2rem">INPUT: a 6×6 feature map</div><canvas id="pn-in" class="cv"></canvas></div>
        <div class="kn-op">→</div>
        <div><div class="tiny" style="font-weight:800;margin-bottom:.2rem" id="pn-oh">OUTPUT</div><canvas id="pn-out" class="cv"></canvas></div>
        <div style="flex:1;min-width:14rem">
          <div class="calc" id="pn-calc" style="min-height:5.2rem"></div>
          <div class="grid g2 mt" style="gap:.6rem"><div class="stat"><div class="n" id="pn-size" style="font-size:1.1rem">–</div><div class="l">output size = (6 − 2) / stride + 1</div></div><div class="stat cool"><div class="n" id="pn-ratio" style="font-size:1.1rem">–</div><div class="l">numbers to process afterwards</div></div></div>
        </div>
      </div>`;
    const $ = (s) => root.querySelector(s), cvI = $('#pn-in'), cvO = $('#pn-out');
    const outN = () => (6 - 2) / stride + 1;
    const total = () => outN() * outN();
    const win = (k) => { const n = outN(), r = Math.floor(k / n) * stride, c = (k % n) * stride; return { r, c, v: [IN[r][c], IN[r][c + 1], IN[r + 1][c], IN[r + 1][c + 1]] }; };
    const val = (k) => { const w = win(k).v; if (op === 'max') return Math.max(...w); if (op === 'mean') return (w[0] + w[1] + w[2] + w[3]) / 4; return w[0] * KER[0][0] + w[1] * KER[0][1] + w[2] * KER[1][0] + w[3] * KER[1][1]; };
    const fmt = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/0$/, ''));
    function draw() {
      const cs = 46, n = outN();
      let ctx = U.fit(cvI, 6 * cs, 6 * cs);
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) { ctx.fillStyle = U.ink(IN[r][c] / 12); ctx.fillRect(c * cs, r * cs, cs - 1, cs - 1); ctx.fillStyle = IN[r][c] / 12 > .5 ? '#fff' : '#17202c'; ctx.font = '700 17px ui-monospace,Menlo,monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(IN[r][c], c * cs + cs / 2, r * cs + cs / 2); }
      if (step > 0) { const w = win(step - 1); ctx.fillStyle = 'rgba(232,113,45,.25)'; ctx.fillRect(w.c * cs, w.r * cs, 2 * cs, 2 * cs); ctx.strokeStyle = '#e8712d'; ctx.lineWidth = 3.5; ctx.strokeRect(w.c * cs + 1.5, w.r * cs + 1.5, 2 * cs - 3, 2 * cs - 3); }
      ctx = U.fit(cvO, n * cs, n * cs);
      ctx.fillStyle = '#f3f1ea'; ctx.fillRect(0, 0, n * cs, n * cs);
      const mx = op === 'conv' ? 8 : 9;
      for (let k = 0; k < Math.min(step, total()); k++) { const r = Math.floor(k / n), c = k % n, v = val(k); ctx.fillStyle = op === 'conv' ? U.div(v / mx) : 'rgb(' + U.mix([255, 255, 255], [150, 215, 205], U.clamp(v / 9, 0, 1)).join(',') + ')'; ctx.fillRect(c * cs, r * cs, cs - 1, cs - 1); ctx.fillStyle = '#17202c'; ctx.font = '700 17px ui-monospace,Menlo,monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(fmt(v), c * cs + cs / 2, r * cs + cs / 2); }
      ctx.strokeStyle = 'rgba(23,32,44,.15)'; for (let i = 0; i <= n; i++) { ctx.beginPath(); ctx.moveTo(i * cs, 0); ctx.lineTo(i * cs, n * cs); ctx.moveTo(0, i * cs); ctx.lineTo(n * cs, i * cs); ctx.stroke(); }
      if (step > 0 && step <= total()) { const k = step - 1, r = Math.floor(k / n), c = k % n; ctx.strokeStyle = '#e8712d'; ctx.lineWidth = 3.5; ctx.strokeRect(c * cs + 1.5, r * cs + 1.5, cs - 3, cs - 3); }
      $('#pn-oh').textContent = 'OUTPUT: ' + n + '×' + n;
      let msg;
      if (step === 0) msg = 'Press <b>Next step</b>: a 2×2 window slides over the input, <b>' + stride + ' pixel' + (stride > 1 ? 's' : '') + '</b> at a time.';
      else { const w = win(step - 1), v = w.v; msg = op === 'max' ? 'max(' + v.join(', ') + ') = <span class="res">' + fmt(val(step - 1)) + '</span>' : op === 'mean' ? 'mean(' + v.join(', ') + ') = ' + v.reduce((a, b) => a + b, 0) + ' / 4 = <span class="res">' + fmt(val(step - 1)) + '</span>' : '1·' + v[0] + ' + (−1)·' + v[1] + ' + 1·' + v[2] + ' + (−1)·' + v[3] + ' = <span class="res">' + fmt(val(step - 1)) + '</span>'; }
      $('#pn-calc').innerHTML = '<span class="tiny">' + (op === 'conv' ? 'convolution with a 2×2 kernel [[1, −1], [1, −1]] — in a CNN these weights are learned' : op === 'max' ? 'keep the strongest value in each window — no parameters' : 'average each window — no parameters') + '</span><br>' + msg;
      $('#pn-size').textContent = n + ' × ' + n + ' = ' + n * n; $('#pn-ratio').textContent = Math.round(n * n / 36 * 100) + ' % of the input';
    }
    function setStep(s) { step = U.clamp(s, 0, total()); draw(); }
    function stop() { clearInterval(timer); timer = null; $('#pn-play').textContent = '▶ Play'; }
    $('#pn-next').addEventListener('click', () => { stop(); if (step >= total()) step = 0; setStep(step + 1); });
    $('#pn-back').addEventListener('click', () => { stop(); setStep(step - 1); });
    $('#pn-reset').addEventListener('click', () => { stop(); setStep(0); });
    $('#pn-play').addEventListener('click', () => { if (timer) { stop(); return; } if (step >= total()) step = 0; $('#pn-play').textContent = '❚❚ Pause'; timer = setInterval(() => { if (step >= total()) { stop(); return; } setStep(step + 1); }, 700); });
    root.querySelectorAll('#pn-op button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#pn-op button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); op = b.dataset.o; stop(); setStep(0); }));
    root.querySelectorAll('#pn-st button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#pn-st button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); stride = +b.dataset.s; stop(); setStep(0); }));
    Deck.on('pooling', () => { stop(); setStep(0); });
  })();

  /* tabs on the pooling slide: numbers / stability on the digit */
  (function tabs() {
    const bar = document.getElementById('pool-tabs'); if (!bar) return;
    const a = document.getElementById('w-poolnum'), b = document.getElementById('w-pool');
    bar.querySelectorAll('button').forEach((btn) => btn.addEventListener('click', () => {
      bar.querySelectorAll('button').forEach((x) => x.classList.remove('on')); btn.classList.add('on');
      const n = btn.dataset.t === 'num'; a.style.display = n ? '' : 'none'; b.style.display = n ? 'none' : '';
      if (!n && window.__POOL && window.__POOL.draw) window.__POOL.draw();
    }));
    b.style.display = 'none';
  })();

  /* =================================================================
     3. REAL EXAMPLE: what units respond to
     ================================================================= */
  (function hier() {
    const root = document.getElementById('w-hier'); if (!root) return;
    const cols = [
      { lvl: 'First layer', what: 'edges and strokes', src: 'our digit CNN', cv: true },
      { lvl: 'Early layers', what: 'repeated patterns', src: 'GoogLeNet · mixed4a', img: 'img/fv/mixed4a_97_optimized.png' },
      { lvl: 'Early-middle layers', what: 'textures (here: fur)', src: 'GoogLeNet · mixed4a', img: 'img/fv/mixed4a_143_optimized.png' },
      { lvl: 'Late layers', what: 'object parts (animal faces)', src: 'GoogLeNet · mixed4e', img: 'img/fv/mixed4e_55_optimized.png' },
      { lvl: 'Last convolutional layers', what: 'whole objects (balls)', src: 'GoogLeNet · mixed5a', img: 'img/fv/mixed5a_9_optimized.png' },
    ];
    root.innerHTML = `<div class="hier-row">${cols.map((c, i) => `
      <div class="hier-col step"><div class="hier-pic">${c.cv ? '<canvas id="hr-f"></canvas>' : `<img src="${c.img}" alt="Image synthesised to excite one unit of ${c.src}: ${c.what}">`}</div>
      <div class="tiny" style="font-weight:800;letter-spacing:.05em;color:var(--enc)">${c.lvl.toUpperCase()}</div><div style="font-weight:800">${c.what}</div><div class="tiny">${c.src}</div></div>`).join('')}</div>`;
    Deck.on('hier', () => {
      const cv = root.querySelector('#hr-f'), cs = 18, g = 4, w = 3 * cs * 3 + 2 * g * 2 + 2, h = 3 * cs * 2 + g;
      const ctx = U.fit(cv, 3 * (3 * cs) + 2 * g, 2 * (3 * cs) + g); ctx.clearRect(0, 0, 999, 999);
      LEARNED.forEach((k, i) => { const x0 = (i % 3) * (3 * cs + g), y0 = Math.floor(i / 3) * (3 * cs + g), m = Math.max(...k.map(Math.abs)); for (let j = 0; j < 9; j++) { ctx.fillStyle = U.div(k[j] / m); ctx.fillRect(x0 + (j % 3) * cs, y0 + Math.floor(j / 3) * cs, cs - 1, cs - 1); } });
    });
  })();

  /* =================================================================
     4. DROPOUT
     ================================================================= */
  (function dropout() {
    const root = document.getElementById('w-drop'); if (!root) return;
    const L = [4, 6, 6, 2], W = 620, H = 300, X = [60, 220, 380, 540];
    const pos = L.map((n, l) => Array.from({ length: n }, (_, i) => ({ x: X[l], y: 30 + (H - 60) * (n === 1 ? .5 : i / (n - 1)) })));
    let p = 0.5, mask = null, steps = 0, test = false, timer = null, rnd = U.mulberry32(11);
    root.innerHTML = `
      <div class="row" style="margin-bottom:.5rem;gap:.9rem">
        <label class="ctl" style="margin:0;min-width:11rem">Dropout probability p <span class="val" id="dp-pv">0.5</span><input type="range" id="dp-p" min="0" max="0.8" step="0.1" value="0.5"></label>
        <button class="btn primary" id="dp-step">New training step</button><button class="btn" id="dp-auto">▶ Auto</button><button class="btn" id="dp-test">At test time</button>
      </div>
      <div class="grid g-side" style="align-items:center">
        <div class="card flat" style="padding:.4rem"><svg id="dp-svg" viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" role="img" aria-label="A small neural network in which some hidden units are switched off at random at every training step"></svg></div>
        <div>
          <div class="stat"><div class="n" id="dp-n" style="font-size:1.3rem">–</div><div class="l">hidden units active in this step</div></div>
          <div class="stat mt"><div class="n" id="dp-s" style="font-size:1.3rem">–</div><div class="l">different sub-networks possible (2<sup>12</sup> hidden units)</div></div>
          <div class="callout mt" id="dp-msg" style="font-size:.95rem"></div>
        </div>
      </div>`;
    const $ = (s) => root.querySelector(s), svg = $('#dp-svg');
    function newMask() { mask = L.map((n, l) => Array.from({ length: n }, () => (l === 0 || l === L.length - 1) ? true : rnd() >= p)); steps++; }
    function draw() {
      const on = (l, i) => test || (mask ? mask[l][i] : true);
      let s = '';
      for (let l = 0; l < L.length - 1; l++) for (let i = 0; i < L[l]; i++) for (let j = 0; j < L[l + 1]; j++) { const act = on(l, i) && on(l + 1, j); s += `<line x1="${pos[l][i].x}" y1="${pos[l][i].y}" x2="${pos[l + 1][j].x}" y2="${pos[l + 1][j].y}" stroke="${act ? '#8a93a3' : '#e2dfd5'}" stroke-width="${act ? 1.3 : .8}" ${act ? '' : 'stroke-dasharray="3 4"'}/>`; }
      pos.forEach((col, l) => col.forEach((q, i) => { const a = on(l, i), hid = l > 0 && l < L.length - 1; s += `<circle cx="${q.x}" cy="${q.y}" r="14" fill="${a ? (hid ? '#dff3ef' : '#e6edfd') : '#f3f1ea'}" stroke="${a ? (hid ? '#0d8f7f' : '#2859e0') : '#cfcabb'}" stroke-width="${a ? 2.5 : 1.5}" ${a ? '' : 'stroke-dasharray="3 3"'}/>${a ? '' : `<path d="M${q.x - 6} ${q.y - 6} L${q.x + 6} ${q.y + 6} M${q.x + 6} ${q.y - 6} L${q.x - 6} ${q.y + 6}" stroke="#c0392b" stroke-width="2.5"/>`}`; }));
      ['input', 'hidden 1', 'hidden 2', 'output'].forEach((t, l) => { s += `<text x="${X[l]}" y="14" text-anchor="middle" font-size="11" font-weight="800" fill="#667180">${t.toUpperCase()}</text>`; });
      svg.innerHTML = s;
      const hid = (mask && !test) ? mask[1].concat(mask[2]).filter(Boolean).length : 12;
      $('#dp-n').textContent = hid + ' / 12'; $('#dp-s').textContent = '4,096';
      $('#dp-msg').innerHTML = test ? '<b>At test time</b> every unit is used (its outgoing weights are scaled by 1 − p, so outputs keep the same size).' : (steps ? '<b>Training step ' + steps + '.</b> A different random subset is dropped each time, so no unit can rely on a specific partner being there.' : 'Press <b>New training step</b>.');
    }
    $('#dp-step').addEventListener('click', () => { test = false; newMask(); draw(); });
    $('#dp-test').addEventListener('click', () => { test = true; stopAuto(); draw(); });
    $('#dp-p').addEventListener('input', (e) => { p = +e.target.value; $('#dp-pv').textContent = p.toFixed(1); test = false; newMask(); draw(); });
    function stopAuto() { clearInterval(timer); timer = null; $('#dp-auto').textContent = '▶ Auto'; }
    $('#dp-auto').addEventListener('click', () => { if (timer) { stopAuto(); return; } test = false; $('#dp-auto').textContent = '❚❚ Stop'; timer = setInterval(() => { newMask(); draw(); }, 650); });
    Deck.on('dropout', () => { stopAuto(); if (!mask) newMask(); test = false; draw(); });
  })();
})();
