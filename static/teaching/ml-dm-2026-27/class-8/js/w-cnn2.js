/* Widgets for the CNN part (second batch)
   1. kernel   : "what is a convolution" — a worked patch × kernel example with designed vs learned kernels (slide "convdef")
   2. poolnum  : pooling / strided convolution animated on a grid of numbers                              (slide "pooling")
   3. hier     : what units of trained networks respond to, with real images                              (slide "hier")
   5. photo    : the same kernels applied to a real photograph                                           (slide "filter")
   6. arch     : a real forward pass through a small trained CNN                                          (slide "arch") */
(function () {
  'use strict';

  /* Filters learned by a small CNN (conv 6 filters 3x3 -> ReLU -> max-pool -> dense 10) trained for this class on 800 of the 1,000 digits
     (own numpy implementation, 97 % accuracy on the other 200). Weights: data/cnn.js. */
  const LEARNED = window.CNN_MODEL.conv;
  window.LEARNED_FILTERS = LEARNED;

  /* =================================================================
     1. WHAT IS A CONVOLUTION: patch x kernel, designed vs learned
     ================================================================= */
  (function kernel() {
    const root = document.getElementById('w-kernel'); if (!root) return;
    const D = U.digits();
    const SOBEL = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    let di = 7, r0 = 3, c0 = 2, which = 'designed';
    const kers = () => (which === 'designed' ? SOBEL : LEARNED[1]);
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

  /* tabs on the filter slide: real photograph / draw your own */
  (function filtTabs() {
    const bar = document.getElementById('filt-tabs'); if (!bar) return;
    const a = document.getElementById('w-photo'), b = document.getElementById('w-conv');
    bar.querySelectorAll('button').forEach((btn) => btn.addEventListener('click', () => {
      bar.querySelectorAll('button').forEach((x) => x.classList.remove('on')); btn.classList.add('on');
      const p = btn.dataset.t === 'photo'; a.style.display = p ? '' : 'none'; b.style.display = p ? 'none' : '';
      window.dispatchEvent(new Event('resize'));
    }));
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
     5. THE SAME KERNELS ON A REAL PHOTOGRAPH
     ================================================================= */
  (function photo() {
    const root = document.getElementById('w-photo'); if (!root) return;
    const F = [
      { n: 'Original', k: [0, 0, 0, 0, 1, 0, 0, 0, 0], m: 'direct', t: 'The photograph in greyscale: one grid of numbers.' },
      { n: 'Vertical edges', k: [-1, 0, 1, -2, 0, 2, -1, 0, 1], m: 'abs', g: 1.1, t: 'Bright where brightness changes from left to right: the outlines of the body and the white chest patch.' },
      { n: 'Horizontal edges', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1], m: 'abs', g: 1.1, t: 'Bright where brightness changes from top to bottom: the ground line, the top of the head.' },
      { n: 'Blur', k: Array(9).fill(1 / 9), m: 'direct', t: 'Averages the neighbourhood: removes fine detail, keeps the overall shape.' },
      { n: 'Sharpen', k: [0, -1, 0, -1, 5, -1, 0, -1, 0], m: 'direct', t: 'Exaggerates differences with the neighbours: fine detail (fur, leaves) stands out.' },
      { n: 'Outline', k: [-1, -1, -1, -1, 8, -1, -1, -1, -1], m: 'abs', g: 0.7, t: 'Responds to any change in any direction: a line drawing.' },
    ];
    root.innerHTML = `<div class="row top" style="gap:1.2rem;flex-wrap:wrap;align-items:flex-start">
        <div><div class="tiny" style="font-weight:800">PHOTO</div><canvas id="ph-a" class="cv"></canvas></div>
        <div style="font-size:2rem;font-weight:800;color:var(--muted);align-self:center">⊛</div>
        <div><div class="tiny" style="font-weight:800">KERNEL</div><canvas id="ph-k" class="cv"></canvas></div>
        <div style="font-size:2rem;font-weight:800;color:var(--muted);align-self:center">=</div>
        <div><div class="tiny" style="font-weight:800">FEATURE MAP: <span id="ph-n"></span></div><canvas id="ph-b" class="cv"></canvas></div>
        <div style="flex:1;min-width:13rem"><div class="seg" id="ph-s" style="flex-wrap:wrap;max-width:100%"></div><div class="callout mt" id="ph-t" style="font-size:.92rem"></div></div></div>`;
    const $ = (s) => root.querySelector(s); let g = null, W = 0, H = 0, sel = 1, img = null;
    const seg = $('#ph-s'); F.forEach((f, i) => { const b = document.createElement('button'); b.textContent = f.n; b.addEventListener('click', () => { sel = i; draw(); }); seg.appendChild(b); });
    function load() {
      img = new Image(); img.onload = () => {
        const sc = 200 / Math.max(img.width, img.height); W = Math.round(img.width * sc); H = Math.round(img.height * sc);
        const c = document.createElement('canvas'); c.width = W; c.height = H; const x = c.getContext('2d'); x.drawImage(img, 0, 0, W, H);
        const d = x.getImageData(0, 0, W, H).data; g = new Float32Array(W * H);
        for (let i = 0; i < W * H; i++) g[i] = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255;
        draw();
      }; img.src = 'img/photos/dog.jpg';
    }
    function paintGray(cv, arr) { const ctx = U.fit(cv, W, H), id = ctx.createImageData(W, H); for (let i = 0; i < W * H; i++) { const v = Math.round(255 * U.clamp(arr[i], 0, 1)); id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255; } const t = document.createElement('canvas'); t.width = W; t.height = H; t.getContext('2d').putImageData(id, 0, 0); ctx.drawImage(t, 0, 0, W, H); }
    function draw() {
      if (!g) return; const f = F[sel], out = new Float32Array(W * H);
      for (let r = 1; r < H - 1; r++) for (let c = 1; c < W - 1; c++) { let s = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += g[(r + i - 1) * W + c + j - 1] * f.k[i * 3 + j]; out[r * W + c] = f.m === 'abs' ? Math.abs(s) * f.g : s; }
      paintGray($('#ph-a'), g); paintGray($('#ph-b'), out);
      const ks = 40, ctx = U.fit($('#ph-k'), 3 * ks, 3 * ks), m = Math.max(...f.k.map(Math.abs)), fm = (v) => (Math.abs(v - Math.round(v)) < .005 ? String(Math.round(v)) : v.toFixed(2));
      for (let i = 0; i < 9; i++) { ctx.fillStyle = U.div(f.k[i] / m); ctx.fillRect((i % 3) * ks, Math.floor(i / 3) * ks, ks - 1, ks - 1); ctx.fillStyle = Math.abs(f.k[i] / m) > .6 ? '#fff' : '#17202c'; ctx.font = '700 12px ui-monospace,Menlo,monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(fm(f.k[i]), (i % 3) * ks + ks / 2, Math.floor(i / 3) * ks + ks / 2); }
      $('#ph-n').textContent = f.n.toUpperCase(); $('#ph-t').textContent = f.t;
      seg.querySelectorAll('button').forEach((b, i) => b.classList.toggle('on', i === sel));
    }
    load();
  })();

  /* =================================================================
     6. A REAL FORWARD PASS THROUGH A SMALL TRAINED CNN
     ================================================================= */
  ['w-arch', 'w-arch2'].forEach(function (rid) {
    const root = document.getElementById(rid); if (!root) return; const EX = root.dataset.mode === 'ex';
    const M = window.CNN_MODEL, D = U.digits(); let di = 7;
    root.innerHTML = `<div class="row" style="margin-bottom:.5rem"><button class="btn small" id="ar-next">Another digit</button><span class="small" id="ar-res"></span></div>
      <div class="ar">
        <div class="ar-col"><canvas id="ar-in"></canvas><div class="ar-cap">Input<br><b>8×8×1</b></div></div>
        <div class="ar-op"><div class="ar-box conv">Convolution<br>6 filters 3×3<br>+ ReLU</div></div>
        <div class="ar-col"><canvas id="ar-fm"></canvas><div class="ar-cap">Feature maps<br><b>6×6×6</b></div></div>
        <div class="ar-op"><div class="ar-box pool">Max-pool<br>2×2</div></div>
        <div class="ar-col"><canvas id="ar-pl"></canvas><div class="ar-cap">Pooled maps<br><b>3×3×6</b></div></div>
        <div class="ar-op"><div class="ar-box flat">Flatten</div></div>
        <div class="ar-col"><canvas id="ar-fl"></canvas><div class="ar-cap">Feature vector<br><b>54</b></div></div>
        <div class="ar-op"><div class="ar-box dense">Dense 10<br>+ softmax</div></div>
        <div class="ar-col"><canvas id="ar-out"></canvas><div class="ar-cap">Class probabilities<br><b>10</b></div></div>
      </div>
      <div class="ar-brace"><div style="flex:5.2;border-color:var(--enc);color:var(--enc)">FEATURE EXTRACTOR: convolution + pooling</div><div style="flex:3;border-color:var(--accent);color:var(--accent)">CLASSIFIER: flatten + dense (as in an MLP)</div></div>`;
    const $ = (s) => root.querySelector(s);
    function forward(x) {
      const maps = [], pooled = [];
      for (let f = 0; f < 6; f++) { const k = M.conv[f], m = new Float32Array(36); for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) { let s = M.cb[f]; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += x[(r + i) * 8 + c + j] * k[i * 3 + j]; m[r * 6 + c] = Math.max(0, s); } maps.push(m);
        const p = new Float32Array(9); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) p[r * 3 + c] = Math.max(m[2 * r * 6 + 2 * c], m[2 * r * 6 + 2 * c + 1], m[(2 * r + 1) * 6 + 2 * c], m[(2 * r + 1) * 6 + 2 * c + 1]); pooled.push(p); }
      const fl = new Float32Array(54); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let f = 0; f < 6; f++) fl[(r * 3 + c) * 6 + f] = pooled[f][r * 3 + c];
      const z = M.db.map((b, o) => { let s = b; for (let i = 0; i < 54; i++) s += fl[i] * M.dense[o][i]; return s; }), mx = Math.max(...z), e = z.map((v) => Math.exp(v - mx)), t = e.reduce((a, b) => a + b, 0);
      return { maps, pooled, fl, p: e.map((v) => v / t) };
    }
    function grid(cv, arr, n, cs, shape) { const g = 4, w = 3 * n * cs + 2 * g, h = 2 * n * cs + g, ctx = U.fit(cv, w, h); arr.forEach((m, i) => { const x0 = (i % 3) * (n * cs + g), y0 = Math.floor(i / 3) * (n * cs + g), mx = Math.max(1e-6, ...m); for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { ctx.fillStyle = U.ink(m[r * n + c] / mx); ctx.fillRect(x0 + c * cs, y0 + r * cs, cs - .4, cs - .4); } ctx.strokeStyle = 'rgba(23,32,44,.3)'; ctx.strokeRect(x0 + .5, y0 + .5, n * cs - 1, n * cs - 1); }); }
    function draw() {
      const x = D.X.subarray(di * 64, di * 64 + 64), o = forward(x);
      let ctx = U.fit($('#ar-in'), 96, 96); U.drawGrid(ctx, x, 8, 0, 0, 12); ctx.strokeStyle = 'rgba(23,32,44,.4)'; ctx.strokeRect(.5, .5, 95, 95);
      grid($('#ar-fm'), o.maps, 6, 9); grid($('#ar-pl'), o.pooled, 3, 18);
      ctx = U.fit($('#ar-fl'), 9 * 12, 6 * 12); const mx = Math.max(1e-6, ...o.fl); for (let i = 0; i < 54; i++) { ctx.fillStyle = U.ink(o.fl[i] / mx); ctx.fillRect((i % 9) * 12, Math.floor(i / 9) * 12, 11.3, 11.3); }
      ctx = U.fit($('#ar-out'), 130, 10 * 17); const best = o.p.indexOf(Math.max(...o.p));
      for (let k = 0; k < 10; k++) { ctx.fillStyle = k === best ? '#e8712d' : '#cfcabb'; ctx.fillRect(16, k * 17 + 2, Math.max(1, o.p[k] * 110), 13); ctx.fillStyle = '#17202c'; ctx.font = '700 11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(k, 2, k * 17 + 9); }
      if (EX) { $('#ar-res').innerHTML = 'A real digit goes through the network. The pictures are real feature maps.'; return; }
      $('#ar-res').innerHTML = 'Real forward pass: the network predicts <b>' + best + '</b> (probability ' + o.p[best].toFixed(2) + '); true label: <b>' + D.y[di] + '</b>.';
    }
    $('#ar-next').addEventListener('click', () => { di = (di * 7 + 13) % D.n; draw(); });
    Deck.on(root.closest('.slide').id, draw);
  });
})();
