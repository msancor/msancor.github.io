/* Receptive field explorer: stack conv / pool layers, click any unit, see which input pixels it depends on. */
(function () {
  'use strict';
  const root = document.getElementById('w-rf'); if (!root) return;

  const N0 = 16, MAXL = 5;
  const T = { conv: { k: 3, s: 1, name: 'Conv 3×3' }, pool: { k: 2, s: 2, name: 'Pool 2×2' } };
  let layers = ['conv', 'conv', 'conv'];
  let sel = null;            // {l, r, c}
  let digit = 0;
  let geo = null;            // layout for hit-testing

  root.innerHTML = `
    <div class="row" style="margin-bottom:.5rem">
      <button class="btn" id="rf-conv">+ Conv 3×3</button>
      <button class="btn" id="rf-pool">+ Pool 2×2</button>
      <button class="btn small" id="rf-undo">Undo</button>
      <button class="btn small" id="rf-reset">Reset</button>
      <button class="btn small" id="rf-digit">New image</button>
      <span class="spacer"></span>
      <span class="small">Click any unit in any layer ➜ see what it “looks at” in the image.</span>
    </div>
    <div class="card flat" style="padding:.6rem .7rem;overflow-x:auto"><canvas id="rf-cv"></canvas></div>
    <div class="grid g-side mt" style="align-items:stretch">
      <div class="stat" style="display:flex;gap:1.2rem;align-items:center">
        <div><div class="n" id="rf-n">–</div><div class="l">pixels seen by the selected unit</div></div>
        <div><div class="n" id="rf-p" style="color:var(--accent)">–</div><div class="l">of the image</div></div>
      </div>
      <div class="callout" id="rf-msg"></div>
    </div>`;

  const cv = root.querySelector('#rf-cv');
  const sizes = () => { const s = [N0]; layers.forEach((t) => s.push(t === 'conv' ? s[s.length - 1] - 2 : Math.floor(s[s.length - 1] / 2))); return s; };
  function back(l, r, c) { // ranges (inclusive) in every layer below l
    const out = []; out[l] = { r0: r, r1: r, c0: c, c1: c };
    for (let i = l; i >= 1; i--) {
      const t = T[layers[i - 1]], q = out[i];
      out[i - 1] = { r0: q.r0 * t.s, r1: q.r1 * t.s + t.k - 1, c0: q.c0 * t.s, c1: q.c1 * t.s + t.k - 1 };
    }
    return out;
  }
  function defaultSel() { const s = sizes(), l = layers.length; return { l, r: Math.floor((s[l] - 1) / 2), c: Math.floor((s[l] - 1) / 2) }; }

  function draw() {
    const S = sizes(), L = S.length, gap = 34, head = 34, W = root.clientWidth - 24;
    const totalCells = S.reduce((a, b) => a + b, 0);
    const cell = U.clamp(Math.floor(Math.min((W - gap * (L - 1)) / totalCells, (window.innerHeight - 580) / N0)), 8, 24);
    const H = N0 * cell + head + 6;
    let width = totalCells * cell + gap * (L - 1);
    const ctx = U.fit(cv, Math.max(width, 10), H);
    geo = [];
    if (!sel || sel.l > layers.length || sel.r >= S[sel.l] || sel.c >= S[sel.l]) sel = defaultSel();
    const rng = back(sel.l, sel.r, sel.c);
    const D = U.digits();
    let x = 0;
    for (let l = 0; l < L; l++) {
      const n = S[l], y0 = head + ((N0 - n) * cell) / 2;
      geo.push({ x, y: y0, n, cell });
      // header
      ctx.fillStyle = '#3d4856'; ctx.font = '700 11px system-ui,sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(l === 0 ? 'INPUT' : T[layers[l - 1]].name.toUpperCase(), x, 13);
      ctx.fillStyle = '#667180'; ctx.font = '600 11px system-ui,sans-serif'; ctx.fillText(n + ' × ' + n, x, 27);
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
        if (l === 0) { ctx.fillStyle = U.ink(D.X[digit * 64 + Math.floor(r / 2) * 8 + Math.floor(c / 2)]); }
        else ctx.fillStyle = '#e4f1ee';
        ctx.fillRect(x + c * cell, y0 + r * cell, cell - 1, cell - 1);
      }
      // highlight receptive-field region
      const q = rng[l];
      if (q) {
        ctx.fillStyle = l === 0 ? 'rgba(40,89,224,.38)' : 'rgba(40,89,224,.30)';
        ctx.fillRect(x + q.c0 * cell, y0 + q.r0 * cell, (q.c1 - q.c0 + 1) * cell - 1, (q.r1 - q.r0 + 1) * cell - 1);
        ctx.strokeStyle = '#2859e0'; ctx.lineWidth = 2.2;
        ctx.strokeRect(x + q.c0 * cell + 1, y0 + q.r0 * cell + 1, (q.c1 - q.c0 + 1) * cell - 3, (q.r1 - q.r0 + 1) * cell - 3);
      }
      if (l === sel.l) { ctx.strokeStyle = '#e8712d'; ctx.lineWidth = 3; ctx.strokeRect(x + sel.c * cell + 1, y0 + sel.r * cell + 1, cell - 3, cell - 3); }
      x += n * cell + gap;
    }
    // connectors
    ctx.fillStyle = '#667180'; ctx.font = '700 16px system-ui,sans-serif'; ctx.textAlign = 'center';
    let xx = 0; for (let l = 0; l < L - 1; l++) { xx += S[l] * cell + gap / 2; ctx.fillText('→', xx, head + (N0 * cell) / 2 + 5); xx += gap / 2; }

    // stats
    const q0 = rng[0], side = q0.r1 - q0.r0 + 1;
    root.querySelector('#rf-n').textContent = side + ' × ' + side;
    root.querySelector('#rf-p').textContent = Math.round((side * side) / (N0 * N0) * 100) + '%';
    // receptive field after each layer
    let rf = 1, jump = 1; const chain = [];
    layers.forEach((t) => { rf += (T[t].k - 1) * jump; jump *= T[t].s; chain.push(rf); });
    const hasPool = layers.includes('pool'), nConv = layers.filter((t) => t === 'conv').length;
    let m = '<b>Receptive field after each layer:</b> ' + (chain.length ? chain.join(' → ') : '–') + ' pixels wide. ';
    if (!layers.length) m = 'Add a layer to begin.';
    else if (!hasPool) m += 'Each extra 3×3 conv adds just 2 pixels. Stacking <b>' + nConv + '</b> small filters reaches the view of one ' + (2 * nConv + 1) + '×' + (2 * nConv + 1) + ' filter, with ' + nConv * 9 + ' instead of ' + Math.pow(2 * nConv + 1, 2) + ' weights per channel.';
    else m += '<b>Pooling makes the receptive field grow much faster</b> — later units see a large part of the image although every filter is tiny.';
    root.querySelector('#rf-msg').innerHTML = m;
    root.querySelector('#rf-conv').disabled = layers.length >= MAXL || S[L - 1] < 4;
    root.querySelector('#rf-pool').disabled = layers.length >= MAXL || S[L - 1] < 2;
    root.querySelector('#rf-undo').disabled = layers.length <= 1;
  }

  function hit(e) {
    const p = U.pos(cv, e);
    for (let l = 1; l < geo.length; l++) {
      const g = geo[l], c = Math.floor((p.x - g.x) / g.cell), r = Math.floor((p.y - g.y) / g.cell);
      if (c >= 0 && c < g.n && r >= 0 && r < g.n) return { l, r, c };
    }
    return null;
  }
  cv.addEventListener('pointerdown', (e) => { const h = hit(e); if (h) { sel = h; draw(); } });
  cv.addEventListener('pointermove', (e) => { cv.style.cursor = hit(e) ? 'pointer' : 'default'; if (e.buttons) { const h = hit(e); if (h) { sel = h; draw(); } } });
  root.querySelector('#rf-conv').addEventListener('click', () => { layers.push('conv'); sel = null; draw(); });
  root.querySelector('#rf-pool').addEventListener('click', () => { layers.push('pool'); sel = null; draw(); });
  root.querySelector('#rf-undo').addEventListener('click', () => { layers.pop(); sel = null; draw(); });
  root.querySelector('#rf-reset').addEventListener('click', () => { layers = ['conv', 'conv', 'conv']; sel = null; draw(); });
  root.querySelector('#rf-digit').addEventListener('click', () => { digit = Math.floor(Math.random() * U.digits().n); draw(); });

  Deck.on('rf', draw);
})();
