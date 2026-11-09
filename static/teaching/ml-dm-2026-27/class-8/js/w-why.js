/* Section "Why not an MLP?"  —  three small widgets:
   1. params   : parameter explosion calculator (dense vs conv)
   2. shuffle  : flattening / pixel permutation demo
   3. connect  : dense vs local vs local+shared weights (1-D picture)        */
(function () {
  'use strict';

  /* =================================================================
     1. PARAMETER CALCULATOR
     ================================================================= */
  (function params() {
    const root = document.getElementById('w-params'); if (!root) return;
    root.innerHTML = `
      <div class="grid g-side-r">
        <div class="card flat">
          <div class="grid g2" style="gap:.7rem">
            <label class="ctl">Image side (pixels)<select id="pc-n"><option>28</option><option>64</option><option>128</option><option selected>224</option><option>512</option></select></label>
            <label class="ctl">Channels<div class="seg" id="pc-c"><button data-v="1">1 · grey</button><button data-v="3" class="on">3 · RGB</button></div></label>
            <label class="ctl">Dense: hidden units<select id="pc-h"><option>100</option><option selected>1000</option><option>10000</option></select></label>
            <label class="ctl">Conv: filters<select id="pc-f"><option>8</option><option>16</option><option>32</option><option selected>64</option><option>128</option></select></label>
            <label class="ctl">Conv: kernel size<div class="seg" id="pc-k"><button data-v="3" class="on">3×3</button><button data-v="5">5×5</button><button data-v="7">7×7</button></div></label>
          </div>
          <div class="small mt">Reference point: 28×28 grey, 1,000 units → <b>785,000</b> weights + biases; add a 10-unit output layer (10,010) → <b>795,010</b>, the dense MNIST network from the course exercise. Presets: <button class="btn small" id="pc-mnist">MNIST</button> <button class="btn small" id="pc-photo">224×224 photo</button></div>
        </div>
        <div>
          <div class="grid g3" style="gap:.7rem">
            <div class="stat"><div class="n" id="pr-in">–</div><div class="l">input values<br><span id="pr-in-f"></span></div></div>
            <div class="stat hot"><div class="n" id="pr-d">–</div><div class="l">dense layer parameters<br><span id="pr-d-f"></span></div></div>
            <div class="stat cool"><div class="n" id="pr-c">–</div><div class="l">conv layer parameters<br><span id="pr-c-f"></span></div></div>
          </div>
          <div class="mt">
            <div class="tiny" style="margin-bottom:.2rem">Size on a <b>log scale</b> (each tick = ×10)</div>
            <div id="pr-bars"></div>
          </div>
          <div class="callout mt" id="pr-msg"></div>
        </div>
      </div>`;

    const $ = (s) => root.querySelector(s);
    const state = { c: 3, k: 3 };
    function seg(id, key) {
      root.querySelectorAll(id + ' button').forEach((b) => b.addEventListener('click', () => {
        root.querySelectorAll(id + ' button').forEach((x) => x.classList.remove('on')); b.classList.add('on');
        state[key] = +b.dataset.v; update();
      }));
    }
    seg('#pc-c', 'c'); seg('#pc-k', 'k');
    ['#pc-n', '#pc-h', '#pc-f'].forEach((s) => $(s).addEventListener('change', update));
    function preset(n, c) { $('#pc-n').value = n; state.c = c; root.querySelectorAll('#pc-c button').forEach((x) => x.classList.toggle('on', +x.dataset.v === c)); update(); }
    $('#pc-mnist').addEventListener('click', () => preset(28, 1)); $('#pc-photo').addEventListener('click', () => preset(224, 3));

    function update() {
      const N = +$('#pc-n').value, H = +$('#pc-h').value, F = +$('#pc-f').value, C = state.c, k = state.k;
      const inp = N * N * C, dense = inp * H + H, conv = F * (k * k * C + 1);
      $('#pr-in').textContent = U.commas(inp);            $('#pr-in-f').textContent = N + ' × ' + N + ' × ' + C;
      $('#pr-d').textContent = U.human(dense);            $('#pr-d-f').textContent = U.commas(inp) + ' × ' + U.commas(H) + ' + ' + U.commas(H);
      $('#pr-c').textContent = U.commas(conv);            $('#pr-c-f').textContent = F + ' × (' + k + '×' + k + '×' + C + ' + 1)';
      const max = 10, w = (v) => Math.max(1.5, Math.log10(v) / max * 100);
      let ticks = ''; for (let i = 0; i <= max; i += 1) ticks += '<span style="position:absolute;left:' + (i / max * 100) + '%;top:0;bottom:0;border-left:1px solid #0001"></span>';
      $('#pr-bars').innerHTML =
        '<div style="position:relative;padding:.25rem 0">' + ticks +
        '<div style="position:relative;height:1.1rem;background:var(--gen);width:' + w(dense) + '%;border-radius:4px"></div>' +
        '<div style="position:relative;height:1.1rem;background:var(--enc);width:' + w(conv) + '%;border-radius:4px;margin-top:.3rem"></div></div>' +
        '<div class="tiny" style="display:flex;justify-content:space-between"><span>1</span><span>10²</span><span>10⁴</span><span>10⁶</span><span>10⁸</span><span>10¹⁰</span></div>';
      const mb = dense * 4 / 1e6;
      $('#pr-msg').innerHTML = 'The conv layer is <b>' + U.commas(dense / conv) + '× smaller</b>, and its size does not depend on the image size at all. ' +
        'The dense layer’s weights alone would need ≈ <b>' + (mb >= 1000 ? (mb / 1000).toFixed(1) + ' GB' : Math.round(mb) + ' MB') + '</b> (32-bit floats).';
    }
    update();
  })();

  /* =================================================================
     2. PIXEL SHUFFLE
     ================================================================= */
  (function shuffle() {
    const root = document.getElementById('w-shuffle'); if (!root) return;
    root.innerHTML = `
      <div class="row top" style="gap:1.6rem;justify-content:center">
        <div><canvas id="sh-a" class="cv"></canvas><div class="cap" id="sh-a-cap">What <i>you</i> see</div><canvas id="sh-sa" class="cv" style="margin-top:.5rem"></canvas><div class="cap">…what an MLP receives (64 numbers in a row)</div></div>
        <div style="align-self:center;font-size:1.6rem;color:var(--muted)">⇄<div class="tiny" style="text-align:center">same pixels,<br>re-ordered</div></div>
        <div><canvas id="sh-b" class="cv"></canvas><div class="cap">Same image, pixels randomly permuted</div><canvas id="sh-sb" class="cv" style="margin-top:.5rem"></canvas><div class="cap">…what an MLP receives</div></div>
      </div>
      <div class="row mt" style="justify-content:center">
        <button class="btn" id="sh-next">Next digit</button>
        <button class="btn" id="sh-perm">New shuffle</button>
        <span class="small">Hover a pixel: its 8 neighbours light up in both images.</span>
      </div>`;
    const A = root.querySelector('#sh-a'), B = root.querySelector('#sh-b'), SA = root.querySelector('#sh-sa'), SB = root.querySelector('#sh-sb');
    const S = 8, CELL = 26, W = S * CELL;
    let idx = 0, perm = [], inv = [], hover = -1;
    const rnd = U.mulberry32(11);

    function newPerm() {
      perm = Array.from({ length: 64 }, (_, i) => i);
      for (let i = 63; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
      inv = []; perm.forEach((p, i) => (inv[p] = i));
    }
    function nextDigit() { const d = U.digits(); idx = Math.floor(rnd() * d.n); }
    function neigh(i) {
      const r = Math.floor(i / 8), c = i % 8, out = [];
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue; const rr = r + dr, cc = c + dc;
        if (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) out.push(rr * 8 + cc);
      }
      return out;
    }
    function drawImg(canvas, data, marks) {
      const ctx = U.fit(canvas, W, W);
      U.drawGrid(ctx, data, S, 0, 0, CELL, { gap: 1 });
      if (marks) {
        marks.nb.forEach((p) => { ctx.strokeStyle = '#2859e0'; ctx.lineWidth = 2; ctx.strokeRect((p % 8) * CELL + 2, Math.floor(p / 8) * CELL + 2, CELL - 4, CELL - 4); });
        const p = marks.c; ctx.strokeStyle = '#e8712d'; ctx.lineWidth = 3.5; ctx.strokeRect((p % 8) * CELL + 1.5, Math.floor(p / 8) * CELL + 1.5, CELL - 3, CELL - 3);
      }
    }
    function drawStrip(canvas, data, marks) {
      const w = W, h = 22, ctx = U.fit(canvas, w, h), cw = w / 64;
      for (let i = 0; i < 64; i++) { ctx.fillStyle = U.ink(data[i]); ctx.fillRect(i * cw, 0, cw + .4, h); }
      if (marks) {
        marks.nb.forEach((p) => { ctx.fillStyle = 'rgba(40,89,224,.95)'; ctx.fillRect(p * cw, h - 6, cw + .4, 6); });
        ctx.fillStyle = '#e8712d'; ctx.fillRect(marks.c * cw, 0, cw + .4, h);
        ctx.fillStyle = 'rgba(232,113,45,.0)';
      }
    }
    function draw() {
      const D = U.digits(), orig = D.X.subarray(idx * 64, idx * 64 + 64);
      const shuf = new Float32Array(64); for (let i = 0; i < 64; i++) shuf[perm[i]] = orig[i];
      let ma = null, mb = null;
      if (hover >= 0) { // hover is index in the ORIGINAL image
        const nb = neigh(hover);
        ma = { c: hover, nb }; mb = { c: perm[hover], nb: nb.map((i) => perm[i]) };
      }
      drawImg(A, orig, ma); drawImg(B, shuf, mb); drawStrip(SA, orig, ma); drawStrip(SB, shuf, mb);
      root.querySelector('#sh-a-cap').textContent = 'What you see — a handwritten “' + D.y[idx] + '”';
    }
    function bind(canvas, fromShuffled) {
      canvas.addEventListener('pointermove', (e) => {
        const p = U.pos(canvas, e), c = Math.floor(p.x / CELL), r = Math.floor(p.y / CELL);
        if (c < 0 || c > 7 || r < 0 || r > 7) return;
        const i = fromShuffled ? inv[r * 8 + c] : r * 8 + c;
        if (i !== hover) { hover = i; draw(); }
      });
      canvas.addEventListener('pointerleave', () => { hover = -1; draw(); });
    }
    bind(A, false); bind(B, true);
    root.querySelector('#sh-next').addEventListener('click', () => { nextDigit(); draw(); });
    root.querySelector('#sh-perm').addEventListener('click', () => { newPerm(); draw(); });
    newPerm(); nextDigit();
    Deck.on('shuffle', draw);
  })();

  /* =================================================================
     3. DENSE vs LOCAL vs LOCAL+SHARED  (1-D picture)
     ================================================================= */
  (function connect() {
    const root = document.getElementById('w-connect'); if (!root) return;
    root.innerHTML = `
      <div class="row" style="margin-bottom:.4rem">
        <div class="seg" id="cn-mode">
          <button data-m="dense" class="on">① Dense (MLP)</button>
          <button data-m="local">② Local connections</button>
          <button data-m="shared">③ Local + shared weights (conv)</button>
        </div>
      </div>
      <div class="card flat" style="padding:.6rem"><svg id="cn-svg" viewBox="0 0 640 290" style="width:100%;height:auto;display:block"></svg></div>
      <div class="row mt">
        <div class="stat" style="min-width:12rem"><div class="n" id="cn-n">48</div><div class="l" id="cn-l">weights (8 inputs × 6 outputs)</div></div>
        <div class="callout" id="cn-msg" style="flex:1"></div>
      </div>`;
    const svg = root.querySelector('#cn-svg'), NS = 'http://www.w3.org/2000/svg';
    const x = [0, 0, 1, 0, 0, 0, 1, 0], w = [-1, 2, -1], NI = 8, NO = 6;
    const COL = ['#e8712d', '#0d8f7f', '#7c4fd6'];
    let mode = 'dense', hov = -1;
    const xi = (i) => 60 + i * (520 / 7), yi = 52, yo = 232;
    const msg = {
      dense: '<b>Every</b> output looks at <b>every</b> input, with its own weight. Nothing says that nearby pixels belong together — the network has to discover that by itself, from data.',
      local: 'Each output only looks at a <b>small neighbourhood</b> (3 inputs). Far fewer weights — but each position still has its <b>own</b> weights, so a pattern must be learned separately at every location.',
      shared: 'Same neighbourhood, but the <b>same 3 weights</b> are reused at every position (same colour = same number). Here <code>[−1, 2, −1]</code> detects an isolated blip — and fires at <b>both</b> places where it occurs.',
    };
    function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; }
    function draw() {
      svg.innerHTML = '';
      const links = [];
      for (let j = 0; j < NO; j++) {
        const ins = mode === 'dense' ? [0, 1, 2, 3, 4, 5, 6, 7] : [j, j + 1, j + 2];
        ins.forEach((i, o) => links.push({ i, j, o }));
      }
      links.forEach((l) => {
        const dim = hov >= 0 && l.j !== hov, hot = hov >= 0 && l.j === hov;
        const col = mode === 'shared' ? COL[l.o] : (mode === 'local' ? '#5d6b86' : '#8a93a3');
        el('line', { x1: xi(l.i), y1: yi + 14, x2: xi(l.j + 1), y2: yo - 14, stroke: col, 'stroke-width': hot ? 3 : (mode === 'dense' ? 1.1 : 2), opacity: dim ? .1 : (hot ? 1 : (mode === 'dense' ? .5 : .75)) });
      });
      for (let i = 0; i < NI; i++) {
        const on = mode === 'shared' && x[i] === 1;
        el('circle', { cx: xi(i), cy: yi, r: 14, fill: on ? '#17202c' : '#fff', stroke: '#17202c', 'stroke-width': 2 });
        if (mode === 'shared') { const t = el('text', { x: xi(i), y: yi + 5, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, fill: on ? '#fff' : '#17202c' }); t.textContent = x[i]; }
      }
      for (let j = 0; j < NO; j++) {
        let fill = '#fff', txt = '';
        if (mode === 'shared') { const y = w[0] * x[j] + w[1] * x[j + 1] + w[2] * x[j + 2]; fill = U.div(y / 2); txt = y; }
        const c = el('circle', { cx: xi(j + 1), cy: yo, r: 14, fill, stroke: j === hov ? '#2859e0' : '#17202c', 'stroke-width': j === hov ? 3.5 : 2, style: 'cursor:pointer' });
        c.addEventListener('pointerenter', () => { hov = j; draw(); }); c.addEventListener('pointerleave', () => { hov = -1; draw(); });
        if (txt !== '') { const t = el('text', { x: xi(j + 1), y: yo + 5, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: '#17202c', 'pointer-events': 'none' }); t.textContent = txt; }
      }
      const lab = (t, y) => { const e = el('text', { x: 6, y, 'font-size': 12, fill: '#667180', 'font-weight': 700 }); e.textContent = t; };
      lab('INPUT', yi - 24); lab('OUTPUT', yo + 36);
      if (mode === 'shared') {
        const lg = el('text', { x: 634, y: 284, 'text-anchor': 'end', 'font-size': 12, 'font-weight': 700 });
        [['w₁ = −1', 0], ['w₂ = 2', 1], ['w₃ = −1', 2]].forEach(([t, k], n) => { const s = el('tspan', { fill: COL[k], dx: n ? 14 : 0 }, lg); s.textContent = '■ ' + t; });
      }
      const nW = mode === 'dense' ? NI * NO : (mode === 'local' ? NO * 3 : 3);
      root.querySelector('#cn-n').textContent = nW;
      root.querySelector('#cn-l').textContent = mode === 'dense' ? 'weights (8 inputs × 6 outputs)' : mode === 'local' ? 'weights (6 outputs × 3 inputs each)' : 'weights — shared by all 6 outputs';
      root.querySelector('#cn-msg').innerHTML = msg[mode];
    }
    root.querySelectorAll('#cn-mode button').forEach((b) => b.addEventListener('click', () => {
      root.querySelectorAll('#cn-mode button').forEach((z) => z.classList.remove('on')); b.classList.add('on'); mode = b.dataset.m; draw();
    }));
    draw();
  })();
})();
