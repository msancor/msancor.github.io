/* Convolution lab: draw an image, edit a 3x3 kernel, see the feature map, the arithmetic, ReLU and max-pooling. */
(function () {
  'use strict';
  const root = document.getElementById('w-conv'); if (!root) return;

  const IN = 14, K = 3, OUT = IN - K + 1, POOL = OUT / 2;   // 14 -> 12 -> 6
  const PRESETS = [
    { n: 'Identity',         k: [0, 0, 0, 0, 1, 0, 0, 0, 0] },
    { n: 'Vertical edges',   k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
    { n: 'Horizontal edges', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
    { n: 'Blur',             k: Array(9).fill(1 / 9) },
    { n: 'Sharpen',          k: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
    { n: 'Outline',          k: [-1, -1, -1, -1, 8, -1, -1, -1, -1] },
  ];
  const SEVEN = [
    '..............', '..............', '..##########..', '..##########..', '.........##...', '........##....', '.......##.....',
    '......##......', '.....##.......', '.....##.......', '....##........', '....##........', '..............', '..............'];
  function fromAscii(rows) { const a = new Float32Array(IN * IN); rows.forEach((s, r) => [...s].forEach((ch, c) => { a[r * IN + c] = ch === '#' ? 1 : 0; })); return a; }
  function make(fn) { const a = new Float32Array(IN * IN); for (let r = 0; r < IN; r++) for (let c = 0; c < IN; c++) a[r * IN + c] = fn(r, c) ? 1 : 0; return a; }
  const IMAGES = [
    { n: '7',        d: () => fromAscii(SEVEN) },
    { n: 'Block',    d: () => make((r, c) => r >= 3 && r <= 10 && c >= 3 && c <= 8) },
    { n: 'Plus',     d: () => make((r, c) => (r >= 6 && r <= 7 && c >= 2 && c <= 11) || (c >= 6 && c <= 7 && r >= 2 && r <= 11)) },
    { n: 'Ring',     d: () => make((r, c) => { const d = Math.hypot(r - 6.5, c - 6.5); return d >= 3.2 && d <= 5.2; }) },
    { n: 'Diagonal', d: () => make((r, c) => Math.abs(r - c) <= 1) },
  ];

  root.innerHTML = `
    <div class="row top" style="gap:1.3rem;align-items:flex-start;flex-wrap:wrap">
      <div>
        <div class="cap" style="margin:0 0 .3rem;text-align:left">① INPUT IMAGE <span class="tiny" style="font-weight:500">— draw on it!</span></div>
        <canvas id="cv-in" class="cv"></canvas>
        <div class="row" style="gap:.35rem;margin-top:.5rem">
          <div class="seg" id="cv-tool"><button data-t="draw" class="on">Draw</button><button data-t="erase">Erase</button></div>
          <button class="btn small" id="cv-clear">Clear</button>
          <span class="tiny">Shift image:</span>
          <button class="btn small" data-sh="l" title="shift left">←</button><button class="btn small" data-sh="r" title="shift right">→</button>
          <button class="btn small" data-sh="u" title="shift up">↑</button><button class="btn small" data-sh="d" title="shift down">↓</button>
        </div>
        <div class="row" style="gap:.35rem;margin-top:.4rem" id="cv-imgs"><span class="tiny">Load:</span></div>
      </div>

      <div class="card flat" style="padding:.8rem .9rem">
        <div class="cap" style="margin:0 0 .3rem;text-align:left">② FILTER <span class="tiny" style="font-weight:500">— 3×3 = 9 numbers</span></div>
        <div class="kgrid" id="cv-k"></div>
        <div class="tiny" style="margin-top:.35rem">Orange = positive weight · blue = negative</div>
        <div class="row" id="cv-opts" style="gap:.35rem;margin-top:.6rem">
          <button class="btn small" id="cv-relu">ReLU: off</button>
          <button class="btn small" id="cv-pool">Max-pool: off</button>
          <button class="btn small" id="cv-nums">Numbers: off</button>
        </div>
        <button class="btn primary" id="cv-play" style="margin-top:.6rem;width:100%">▶ Slide the filter</button>
        <div class="tiny" id="cv-name" style="margin-top:.45rem;font-weight:700;color:var(--ink-2)"></div>
      </div>

      <div>
        <div class="cap" style="margin:0 0 .3rem;text-align:left">③ FEATURE MAP <span class="tiny" style="font-weight:500">— where does the pattern appear?</span></div>
        <canvas id="cv-out" class="cv"></canvas>
        <div class="tiny" style="margin-top:.3rem"><span style="color:var(--pos);font-weight:800">■</span> strong match &nbsp; <span style="color:var(--neg);font-weight:800">■</span> opposite pattern &nbsp; □ nothing</div>
      </div>

      <div id="cv-poolbox" style="display:none">
        <div class="cap" style="margin:0 0 .3rem;text-align:left">④ MAX-POOL 2×2</div>
        <canvas id="cv-pl" class="cv"></canvas>
        <div class="tiny" style="margin-top:.3rem;max-width:11rem">Keeps the strongest response in each 2×2 block: half the size, tolerant to tiny shifts.</div>
      </div>
    </div>

    <div class="grid g-side mt" style="align-items:start">
      <div>
        <h3>A layer has many filters → a stack of feature maps <span class="tiny" style="font-weight:500">(click one to load it)</span></h3>
        <div class="thumbs" id="cv-th"></div>
      </div>
      <div>
        <h3>The arithmetic behind the highlighted pixel</h3>
        <div class="calc" id="cv-calc">Hover over the input or the feature map.</div>
      </div>
    </div>`;

  const $ = (s) => root.querySelector(s);
  const BASIC = root.dataset.mode === 'basic';
  if (BASIC) { $('#cv-opts').style.display = 'none'; }
  const cin = $('#cv-in'), cout = $('#cv-out'), cpl = $('#cv-pl');
  let img = fromAscii(SEVEN), ker = PRESETS[1].k.slice();
  let relu = false, pool = false, nums = false, tool = 'draw', hov = null, cell = 24, timer = null, drawing = false, last = null;
  let raw = null, scale = 1;

  /* ---------- computation ---------- */
  function convWith(k, im) {
    const o = new Float32Array(OUT * OUT);
    for (let r = 0; r < OUT; r++) for (let c = 0; c < OUT; c++) {
      let s = 0;
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) s += im[(r + i) * IN + (c + j)] * k[i * K + j];
      o[r * OUT + c] = s;
    }
    return o;
  }
  function compute() {
    raw = convWith(ker, img);
    scale = Math.max(1e-6, ...raw.map(Math.abs));
  }
  const shown = (v) => (relu ? Math.max(0, v) : v);
  function pooled() {
    const p = new Float32Array(POOL * POOL);
    for (let r = 0; r < POOL; r++) for (let c = 0; c < POOL; c++) {
      let m = -Infinity;
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) m = Math.max(m, shown(raw[(2 * r + i) * OUT + 2 * c + j]));
      p[r * POOL + c] = m;
    }
    return p;
  }

  /* ---------- drawing ---------- */
  function sizeCells() {
    const W = root.clientWidth || 1000, H = window.innerHeight;
    const cw = Math.floor((W - 280 - 70) / (IN + OUT + (pool ? POOL * 1.5 : 0) + 1));
    const ch = Math.floor((H - 540) / IN);
    cell = U.clamp(Math.min(cw, ch), 14, 30);
  }
  function gridLines(ctx, n, cs) {
    ctx.strokeStyle = 'rgba(23,32,44,.09)'; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 1; i < n; i++) { ctx.moveTo(i * cs + .5, 0); ctx.lineTo(i * cs + .5, n * cs); ctx.moveTo(0, i * cs + .5); ctx.lineTo(n * cs, i * cs + .5); }
    ctx.stroke();
  }
  function drawIn() {
    const ctx = U.fit(cin, IN * cell, IN * cell);
    U.drawGrid(ctx, img, IN, 0, 0, cell);
    gridLines(ctx, IN, cell);
    if (hov) {
      const { r, c } = hov;
      ctx.fillStyle = 'rgba(40,89,224,.13)'; ctx.fillRect(c * cell, r * cell, K * cell, K * cell);
      ctx.strokeStyle = '#2859e0'; ctx.lineWidth = 3; ctx.strokeRect(c * cell + 1.5, r * cell + 1.5, K * cell - 3, K * cell - 3);
    }
  }
  function drawOut() {
    const ctx = U.fit(cout, OUT * cell, OUT * cell);
    for (let r = 0; r < OUT; r++) for (let c = 0; c < OUT; c++) {
      const v = shown(raw[r * OUT + c]);
      ctx.fillStyle = U.div(v / scale); ctx.fillRect(c * cell, r * cell, cell + .5, cell + .5);
      if (nums && cell >= 20 && Math.abs(v) > 0.05) {
        ctx.fillStyle = Math.abs(v / scale) > .65 ? '#fff' : '#17202c'; ctx.font = '600 ' + Math.round(cell * .38) + 'px ui-monospace,Menlo,monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(Math.abs(v) >= 10 ? Math.round(v) : v.toFixed(1), c * cell + cell / 2, r * cell + cell / 2);
      }
    }
    gridLines(ctx, OUT, cell);
    if (hov) { ctx.strokeStyle = '#2859e0'; ctx.lineWidth = 3; ctx.strokeRect(hov.c * cell + 1.5, hov.r * cell + 1.5, cell - 3, cell - 3); }
    if (hov && pool) { // show which 2x2 block would be pooled
      const pr = Math.floor(hov.r / 2) * 2, pc = Math.floor(hov.c / 2) * 2;
      ctx.strokeStyle = 'rgba(23,32,44,.55)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 2; ctx.strokeRect(pc * cell + 1, pr * cell + 1, 2 * cell - 2, 2 * cell - 2); ctx.setLineDash([]);
    }
  }
  function drawPool() {
    $('#cv-poolbox').style.display = pool ? 'block' : 'none';
    if (!pool) return;
    const cs = Math.round(cell * 1.5), p = pooled(), ctx = U.fit(cpl, POOL * cs, POOL * cs);
    for (let r = 0; r < POOL; r++) for (let c = 0; c < POOL; c++) { ctx.fillStyle = U.div(p[r * POOL + c] / scale); ctx.fillRect(c * cs, r * cs, cs + .5, cs + .5); }
    gridLines(ctx, POOL, cs);
    if (hov) { const pr = Math.floor(hov.r / 2), pc = Math.floor(hov.c / 2); ctx.strokeStyle = '#2859e0'; ctx.lineWidth = 3; ctx.strokeRect(pc * cs + 1.5, pr * cs + 1.5, cs - 3, cs - 3); }
  }
  const f1 = (v) => (Math.abs(v - Math.round(v)) < 0.005 ? String(Math.round(v)) : v.toFixed(1));
  const f2 = (v) => (Math.abs(v - Math.round(v)) < 0.005 ? String(Math.round(v)) : v.toFixed(2));
  function drawCalc() {
    const box = $('#cv-calc');
    if (!hov) { box.innerHTML = 'Hover over the input or the feature map to see how one output pixel is computed.'; return; }
    const { r, c } = hov; let sum = 0, rows = '';
    for (let i = 0; i < K; i++) {
      const parts = [];
      for (let j = 0; j < K; j++) {
        const p = img[(r + i) * IN + c + j], w = ker[i * K + j]; sum += p * w;
        parts.push(f1(p) + '×' + (w < 0 ? '(' + f2(w) + ')' : f2(w)));
      }
      rows += parts.join(' + ') + (i < K - 1 ? ' +' : '') + '<br>';
    }
    const out = relu ? Math.max(0, sum) : sum;
    box.innerHTML = '<span class="tiny">3×3 patch × filter weights, then add everything up:</span><br>' + rows +
      '= <span class="res">' + f2(sum) + '</span>' + (relu ? '  → ReLU → <span class="res">' + f2(out) + '</span>' : '');
  }
  function nameOf() {
    const m = PRESETS.find((p) => p.k.every((v, i) => Math.abs(v - ker[i]) < 0.011));
    $('#cv-name').textContent = m ? 'Preset: ' + m.n : 'Custom filter';
    root.querySelectorAll('.thumb').forEach((t, i) => t.classList.toggle('on', PRESETS[i] === m));
  }
  function drawThumbs() {
    PRESETS.forEach((p, i) => {
      const cv = root.querySelector('#th' + i), m = convWith(p.k, img), mx = Math.max(1e-6, ...m.map(Math.abs)), cs = 4;
      const ctx = U.fit(cv, OUT * cs, OUT * cs);
      for (let r = 0; r < OUT; r++) for (let c = 0; c < OUT; c++) { const v = relu ? Math.max(0, m[r * OUT + c]) : m[r * OUT + c]; ctx.fillStyle = U.div(v / mx); ctx.fillRect(c * cs, r * cs, cs + .3, cs + .3); }
      cv.style.width = '100%'; cv.style.height = 'auto';
    });
  }
  function drawKernelInputs() {
    root.querySelectorAll('#cv-k input').forEach((inp, i) => {
      if (document.activeElement !== inp) inp.value = f2(ker[i]);
      inp.className = ker[i] > 0.005 ? 'pos' : (ker[i] < -0.005 ? 'neg' : '');
    });
  }
  function redraw() { compute(); sizeCells(); drawIn(); drawOut(); drawPool(); drawCalc(); drawThumbs(); drawKernelInputs(); nameOf(); }

  /* ---------- UI construction ---------- */
  const kbox = $('#cv-k');
  for (let i = 0; i < 9; i++) {
    const inp = document.createElement('input'); inp.type = 'number'; inp.step = '0.1';
    inp.addEventListener('input', () => { const v = parseFloat(inp.value); ker[i] = isNaN(v) ? 0 : v; compute(); sizeCells(); drawIn(); drawOut(); drawPool(); drawCalc(); drawThumbs(); inp.className = ker[i] > 0.005 ? 'pos' : (ker[i] < -0.005 ? 'neg' : ''); nameOf(); });
    kbox.appendChild(inp);
  }
  const th = $('#cv-th');
  PRESETS.forEach((p, i) => {
    const d = document.createElement('div'); d.className = 'thumb'; d.innerHTML = '<canvas id="th' + i + '"></canvas><span>' + p.n + '</span>';
    d.addEventListener('click', () => { ker = p.k.slice(); redraw(); });
    th.appendChild(d);
  });
  const imgs = $('#cv-imgs');
  IMAGES.forEach((m) => {
    const b = document.createElement('button'); b.className = 'btn small'; b.textContent = m.n;
    b.addEventListener('click', () => { img = m.d(); hov = null; redraw(); }); imgs.appendChild(b);
  });
  function toggle(id, label, get, set) {
    const b = $(id); b.addEventListener('click', () => { set(!get()); b.textContent = label + ': ' + (get() ? 'on' : 'off'); b.classList.toggle('on', get()); redraw(); });
  }
  toggle('#cv-relu', 'ReLU', () => relu, (v) => (relu = v));
  toggle('#cv-pool', 'Max-pool', () => pool, (v) => (pool = v));
  toggle('#cv-nums', 'Numbers', () => nums, (v) => (nums = v));
  root.querySelectorAll('#cv-tool button').forEach((b) => b.addEventListener('click', () => {
    root.querySelectorAll('#cv-tool button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); tool = b.dataset.t;
  }));
  $('#cv-clear').addEventListener('click', () => { img = new Float32Array(IN * IN); hov = null; redraw(); });
  root.querySelectorAll('[data-sh]').forEach((b) => b.addEventListener('click', () => {
    const d = b.dataset.sh, n = new Float32Array(IN * IN);
    for (let r = 0; r < IN; r++) for (let c = 0; c < IN; c++) {
      const sr = r + (d === 'd' ? -1 : d === 'u' ? 1 : 0), sc = c + (d === 'r' ? -1 : d === 'l' ? 1 : 0);
      n[r * IN + c] = (sr >= 0 && sr < IN && sc >= 0 && sc < IN) ? img[sr * IN + sc] : 0;
    }
    img = n; redraw();
  }));

  /* ---------- pointer interaction ---------- */
  function cellAt(canvas, e, n) { const p = U.pos(canvas, e); return { r: U.clamp(Math.floor(p.y / (canvas.clientHeight / n)), 0, n - 1), c: U.clamp(Math.floor(p.x / (canvas.clientWidth / n)), 0, n - 1) }; }
  function paint(rc, erase) { img[rc.r * IN + rc.c] = erase ? 0 : 1; }
  function line(a, b, erase) { // Bresenham so fast strokes have no gaps
    let { r: r0, c: c0 } = a; const { r: r1, c: c1 } = b, dr = Math.abs(r1 - r0), dc = Math.abs(c1 - c0), sr = r0 < r1 ? 1 : -1, sc = c0 < c1 ? 1 : -1; let err = dc - dr;
    for (;;) { paint({ r: r0, c: c0 }, erase); if (r0 === r1 && c0 === c1) break; const e2 = 2 * err; if (e2 > -dr) { err -= dr; c0 += sc; } if (e2 < dc) { err += dc; r0 += sr; } }
  }
  function stopPlay() { if (timer) { clearInterval(timer); timer = null; $('#cv-play').textContent = '▶ Slide the filter'; } }
  cin.addEventListener('pointerdown', (e) => { stopPlay(); cin.setPointerCapture(e.pointerId); drawing = true; const rc = cellAt(cin, e, IN); last = rc; paint(rc, tool === 'erase' || e.button === 2 || e.shiftKey); redraw(); });
  cin.addEventListener('pointermove', (e) => {
    const rc = cellAt(cin, e, IN);
    if (drawing) { line(last, rc, tool === 'erase' || e.shiftKey); last = rc; }
    if (!timer) hov = { r: U.clamp(rc.r - 1, 0, OUT - 1), c: U.clamp(rc.c - 1, 0, OUT - 1) };
    redraw();
  });
  cin.addEventListener('pointerup', () => { drawing = false; });
  cin.addEventListener('contextmenu', (e) => e.preventDefault());
  cin.addEventListener('pointerleave', () => { if (!drawing && !timer) { hov = null; redraw(); } });
  cout.addEventListener('pointermove', (e) => { if (timer) return; hov = cellAt(cout, e, OUT); redraw(); });
  cout.addEventListener('pointerleave', () => { if (!timer) { hov = null; redraw(); } });
  cpl.addEventListener('pointermove', (e) => { if (timer) return; const p = cellAt(cpl, e, POOL); hov = { r: p.r * 2, c: p.c * 2 }; redraw(); });
  cpl.addEventListener('pointerleave', () => { if (!timer) { hov = null; redraw(); } });

  $('#cv-play').addEventListener('click', () => {
    if (timer) { stopPlay(); return; }
    let n = 0; $('#cv-play').textContent = '■ Stop';
    timer = setInterval(() => { hov = { r: Math.floor(n / OUT), c: n % OUT }; redraw(); n = (n + 1) % (OUT * OUT); }, 55);
  });

  Deck.on('filter', (s, resized) => { if (!resized) stopPlay(); redraw(); });
  compute(); redraw();
})();
