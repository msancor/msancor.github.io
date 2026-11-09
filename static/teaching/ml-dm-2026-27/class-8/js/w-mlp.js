/* Act 1 widgets
   1. mlpfwd : forward pass / loss / backprop of a small MLP, step by step           (slide "mlp")
   2. repr   : a hidden layer re-represents XOR data so a straight line suffices     (slide "repr")  */
(function () {
  'use strict';

  /* =================================================================
     1. MLP FORWARD PASS
     ================================================================= */
  (function mlp() {
    const root = document.getElementById('w-mlp'); if (!root) return;
    root.innerHTML = `
      <div class="grid g-side" style="align-items:start">
        <div>
          <div class="card flat" style="padding:.5rem"><svg id="mf-svg" viewBox="0 0 640 290" style="width:100%;height:auto;display:block" role="img" aria-label="A small multilayer perceptron with 3 inputs, two hidden layers of 4 units and 2 outputs"></svg></div>
          <div class="row mt"><button class="btn small" id="mf-prev">◂ Back</button><button class="btn primary" id="mf-next">Next step ▸</button><button class="btn small" id="mf-reset">Reset</button><span class="tiny" id="mf-n"></span></div>
        </div>
        <div class="card flat" id="mf-info" style="min-height:14rem"></div>
      </div>`;
    const svg = root.querySelector('#mf-svg'), NS = 'http://www.w3.org/2000/svg';
    const sizes = [3, 4, 4, 2], X = [70, 225, 385, 560], DY = 54, CY = 130, R = 15;
    const labels = ['x', 'h¹', 'h²', 'ŷ'];
    const steps = [
      { t: 'Input', f: 'x = (x₁, x₂, x₃)', n: 'A vector of numbers: normalised, no missing values. Nothing about the order or arrangement of the features is assumed.', p: '', act: -1 },
      { t: 'Layer 1 — dense', f: 'h¹ = σ( W¹x + b¹ )', n: 'Every hidden unit is connected to every input: a weighted sum plus bias, then a non-linearity σ.', p: '3 × 4 + 4 = <b>16</b> parameters', act: 1 },
      { t: 'Layer 2 — dense', f: 'h² = σ( W²h¹ + b² )', n: 'Without σ, stacked linear maps would collapse into a single linear map — the non-linearity is what makes depth useful.', p: '4 × 4 + 4 = <b>20</b> parameters', act: 2 },
      { t: 'Output layer', f: 'ŷ = softmax( W³h² + b³ )', n: 'Softmax turns the scores into class probabilities that sum to 1.', p: '4 × 2 + 2 = <b>10</b> parameters', act: 3 },
      { t: 'Loss', f: 'L = −Σₖ yₖ log ŷₖ', n: 'Cross-entropy compares the prediction ŷ with the true label y. Training means making L small, on average over the data.', p: 'Total: 16 + 20 + 10 = <b>46</b> parameters', act: 3, loss: true },
      { t: 'Backpropagation + update', f: '∂L/∂W^ℓ by the chain rule;  W ← W − η·∂L/∂W', n: 'Gradients flow backwards from the loss through every layer; an optimiser (SGD, Adam) updates all weights. Repeat over batches and epochs.', p: 'Same loop for every architecture in this class', act: 0, back: true },
    ];
    let s = 0;
    const el = (tag, a, parent) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); (parent || svg).appendChild(e); return e; };
    const ny = (l, i) => CY + (i - (sizes[l] - 1) / 2) * DY;
    function draw() {
      svg.innerHTML = '';
      const st = steps[s];
      for (let l = 0; l < 3; l++) for (let i = 0; i < sizes[l]; i++) for (let j = 0; j < sizes[l + 1]; j++) {
        const hot = st.act === l + 1 || st.back, col = st.back ? '#c0392b' : (hot ? '#2859e0' : '#c9cdd4');
        el('line', { x1: X[l] + R, y1: ny(l, i), x2: X[l + 1] - R, y2: ny(l + 1, j), stroke: col, 'stroke-width': hot ? 1.6 : 1, opacity: hot ? .8 : .55 });
      }
      for (let l = 0; l < 4; l++) {
        const on = (st.act === l) || (st.act === -1 && l === 0) || (st.loss && l === 3) || st.back;
        for (let i = 0; i < sizes[l]; i++) el('circle', { cx: X[l], cy: ny(l, i), r: R, fill: on ? (st.back ? '#fbe9e6' : '#e6edfd') : '#fff', stroke: on ? (st.back ? '#c0392b' : '#2859e0') : '#17202c', 'stroke-width': 2 });
        const t = el('text', { x: X[l], y: 270, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 800, fill: '#3d4856' }); t.textContent = labels[l];
      }
      if (st.back) { const t = el('text', { x: 320, y: 22, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 800, fill: '#c0392b' }); t.textContent = '◂ gradients flow backwards ◂'; }
      else { const t = el('text', { x: 320, y: 22, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 800, fill: '#2859e0' }); t.textContent = st.loss ? 'compare ŷ with the label y' : 'forward pass ▸'; }
      root.querySelector('#mf-info').innerHTML = `<div class="kicker" style="margin:0">Step ${s + 1} of ${steps.length}</div><h3 style="font-size:1.2rem">${st.t}</h3>
        <div class="mono" style="font-size:1.15rem;background:var(--surface-2);border-radius:8px;padding:.55rem .8rem;margin:.4rem 0">${st.f}</div>
        <p>${st.n}</p>${st.p ? `<p class="small" style="color:var(--ink-2)">${st.p}</p>` : ''}`;
      root.querySelector('#mf-n').textContent = '';
      root.querySelector('#mf-prev').disabled = s === 0;
      root.querySelector('#mf-next').disabled = s === steps.length - 1;
    }
    root.querySelector('#mf-next').addEventListener('click', () => { if (s < steps.length - 1) { s++; draw(); } });
    root.querySelector('#mf-prev').addEventListener('click', () => { if (s > 0) { s--; draw(); } });
    root.querySelector('#mf-reset').addEventListener('click', () => { s = 0; draw(); });
    draw();
  })();

  /* =================================================================
     2. REPRESENTATION: XOR with / without a hidden layer
     ================================================================= */
  (function repr() {
    const root = document.getElementById('w-repr'); if (!root) return;
    // Initialisations verified (in this very code) to reach >= 97% on XOR. 2 hidden units can get stuck in a local minimum
    // for ~75% of random starts (non-convex training, Class 5): the demo cycles through starts known to work.
    const SEEDS = [4, 8, 11, 15, 17, 22, 27, 30];
    const SIZE = 300, LIM = 2.1, N = 60;
    root.innerHTML = `
      <div class="row" style="margin-bottom:.6rem">
        <div class="seg" id="rp-mode"><button data-h="0">Linear model on the raw inputs</button><button data-h="2" class="on">MLP with 2 hidden units</button></div>
        <button class="btn primary" id="rp-train">▶ Train</button><button class="btn small" id="rp-reset">New random start</button>
        <span class="spacer"></span>
        <div class="stat" style="padding:.3rem .8rem"><span class="n" id="rp-acc" style="font-size:1.2rem">–</span> <span class="l">accuracy · epoch <span id="rp-ep">0</span></span></div>
      </div>
      <div class="row top" style="gap:1.4rem;justify-content:center">
        <div><canvas id="rp-in" class="cv"></canvas><div class="cap">① Input space (x₁, x₂) — the raw data and the model’s decision regions</div></div>
        <div style="align-self:center;font-size:1.8rem;color:var(--muted)" id="rp-arrow">→</div>
        <div><canvas id="rp-hid" class="cv"></canvas><div class="cap" id="rp-hid-cap">② Hidden space (h₁, h₂) — the learned representation</div></div>
      </div>`;
    const $ = (s) => root.querySelector(s), cin = $('#rp-in'), chid = $('#rp-hid');
    // data: XOR = four Gaussian blobs, label 1 when x1*x2 > 0
    const rnd = U.mulberry32(21), P = [];
    [[-1, -1], [1, 1], [-1, 1], [1, -1]].forEach(([cx, cy]) => { for (let i = 0; i < N; i++) P.push({ x: cx + U.gauss(rnd) * .38, y: cy + U.gauss(rnd) * .38, c: cx * cy > 0 ? 1 : 0 }); });
    let H = 2, seedIdx = 0, m = null, training = false, epoch = 0, raf = null, lastT = 0;
    const sig = (z) => 1 / (1 + Math.exp(-z));

    function make() {
      const r = U.mulberry32(SEEDS[seedIdx % SEEDS.length] * 7919 + H), g = () => U.gauss(r);
      m = H === 0 ? { W: [g() * .5, g() * .5], b: 0 } : { W1: [[g() * .8, g() * .8], [g() * .8, g() * .8]], b1: [0, 0], w2: [g() * .8, g() * .8], b2: 0 };
      m.t = 0; m.M = {}; m.V = {}; epoch = 0;
    }
    function fwd(x, y) {
      if (H === 0) return { o: sig(m.W[0] * x + m.W[1] * y + m.b) };
      const h = [Math.tanh(m.W1[0][0] * x + m.W1[1][0] * y + m.b1[0]), Math.tanh(m.W1[0][1] * x + m.W1[1][1] * y + m.b1[1])];
      return { h, o: sig(m.w2[0] * h[0] + m.w2[1] * h[1] + m.b2) };
    }
    function params() { return H === 0 ? { W0: m.W[0], W1: m.W[1], b: m.b } : { a: m.W1[0][0], b: m.W1[0][1], c: m.W1[1][0], d: m.W1[1][1], e: m.b1[0], f: m.b1[1], g: m.w2[0], h: m.w2[1], i: m.b2 }; }
    function setParams(p) { if (H === 0) { m.W[0] = p.W0; m.W[1] = p.W1; m.b = p.b; } else { m.W1[0][0] = p.a; m.W1[0][1] = p.b; m.W1[1][0] = p.c; m.W1[1][1] = p.d; m.b1[0] = p.e; m.b1[1] = p.f; m.w2[0] = p.g; m.w2[1] = p.h; m.b2 = p.i; } }
    function step() {   // one full-batch Adam step on the log-loss
      const p = params(), g = {}; for (const k in p) g[k] = 0; const n = P.length;
      P.forEach((q) => {
        const f = fwd(q.x, q.y), d = (f.o - q.c) / n;
        if (H === 0) { g.W0 += d * q.x; g.W1 += d * q.y; g.b += d; }
        else {
          g.g += d * f.h[0]; g.h += d * f.h[1]; g.i += d;
          const d0 = d * m.w2[0] * (1 - f.h[0] ** 2), d1 = d * m.w2[1] * (1 - f.h[1] ** 2);
          g.a += d0 * q.x; g.c += d0 * q.y; g.e += d0; g.b += d1 * q.x; g.d += d1 * q.y; g.f += d1;
        }
      });
      m.t++; const lr = .05, b1 = 1 - .9 ** m.t, b2 = 1 - .999 ** m.t;
      for (const k in p) { m.M[k] = .9 * (m.M[k] || 0) + .1 * g[k]; m.V[k] = .999 * (m.V[k] || 0) + .001 * g[k] ** 2; p[k] -= lr * (m.M[k] / b1) / (Math.sqrt(m.V[k] / b2) + 1e-8); }
      setParams(p); epoch++;
    }
    const X = (v, S) => ((v + LIM) / (2 * LIM)) * S, Yc = (v, S) => S - ((v + LIM) / (2 * LIM)) * S;
    function heat(ctx, S, lim, fn) {
      const n = 46, c = S / n;
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const x = ((i + .5) / n) * 2 * lim - lim, y = lim - ((j + .5) / n) * 2 * lim, p = fn(x, y);
        const col = p > .5 ? U.mix([255, 255, 255], [232, 113, 45], (p - .5) * 1.5) : U.mix([255, 255, 255], [40, 89, 224], (.5 - p) * 1.5);
        ctx.fillStyle = 'rgb(' + col.join(',') + ')'; ctx.fillRect(i * c, j * c, c + .6, c + .6);
      }
    }
    function dots(ctx, S, lim, get) {
      P.forEach((q) => { const [a, b] = get(q); ctx.fillStyle = q.c ? '#e8712d' : '#2859e0'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(((a + lim) / (2 * lim)) * S, S - ((b + lim) / (2 * lim)) * S, 4, 0, 6.3); ctx.fill(); ctx.stroke(); });
    }
    function draw() {
      const a = U.fit(cin, SIZE, SIZE); heat(a, SIZE, LIM, (x, y) => fwd(x, y).o); dots(a, SIZE, LIM, (q) => [q.x, q.y]);
      a.strokeStyle = 'rgba(23,32,44,.25)'; a.beginPath(); a.moveTo(SIZE / 2, 0); a.lineTo(SIZE / 2, SIZE); a.moveTo(0, SIZE / 2); a.lineTo(SIZE, SIZE / 2); a.stroke();
      const b = U.fit(chid, SIZE, SIZE);
      if (H === 2) {
        const L = 1.15; heat(b, SIZE, L, (h1, h2) => sig(m.w2[0] * h1 + m.w2[1] * h2 + m.b2)); dots(b, SIZE, L, (q) => { const f = fwd(q.x, q.y); return f.h; });
        b.strokeStyle = 'rgba(23,32,44,.25)'; b.beginPath(); b.moveTo(SIZE / 2, 0); b.lineTo(SIZE / 2, SIZE); b.moveTo(0, SIZE / 2); b.lineTo(SIZE, SIZE / 2); b.stroke();
        // decision line w.h + b = 0
        const w0 = m.w2[0], w1 = m.w2[1], bb = m.b2; b.strokeStyle = '#17202c'; b.lineWidth = 2.5; b.beginPath();
        if (Math.abs(w1) > Math.abs(w0)) { const y1 = (-bb - w0 * -L) / w1, y2 = (-bb - w0 * L) / w1; b.moveTo(0, SIZE - ((y1 + L) / (2 * L)) * SIZE); b.lineTo(SIZE, SIZE - ((y2 + L) / (2 * L)) * SIZE); }
        else { const x1 = (-bb - w1 * -L) / w0, x2 = (-bb - w1 * L) / w0; b.moveTo(((x1 + L) / (2 * L)) * SIZE, SIZE); b.lineTo(((x2 + L) / (2 * L)) * SIZE, 0); }
        b.stroke();
      } else {
        b.fillStyle = '#f3f1ea'; b.fillRect(0, 0, SIZE, SIZE); b.fillStyle = '#667180'; b.font = '600 13px system-ui'; b.textAlign = 'center';
        b.fillText('No hidden layer:', SIZE / 2, SIZE / 2 - 8); b.fillText('the model can only draw a', SIZE / 2, SIZE / 2 + 10); b.fillText('straight line through ①', SIZE / 2, SIZE / 2 + 28);
      }
      let ok = 0; P.forEach((q) => { if ((fwd(q.x, q.y).o > .5) === !!q.c) ok++; });
      $('#rp-acc').textContent = Math.round(ok / P.length * 100) + '%'; $('#rp-ep').textContent = epoch;
      $('#rp-hid-cap').textContent = H === 2 ? '② Hidden space (h₁, h₂): the learned representation — one line separates the classes' : '② (no hidden space)';
      $('#rp-arrow').style.visibility = H === 2 ? 'visible' : 'hidden';
      $('#rp-train').textContent = training ? '❚❚ Pause' : '▶ Train';
    }
    function loop(now) {
      if (!training) return;
      if (!lastT) lastT = now;
      let n = Math.min(Math.floor((now - lastT) * 45 / 1000), 25);
      if (n >= 1) { lastT = now; for (let i = 0; i < n; i++) step(); if (epoch >= 600) training = false; draw(); }
      if (training) raf = requestAnimationFrame(loop);
    }
    $('#rp-train').addEventListener('click', () => { training = !training; lastT = 0; if (training) raf = requestAnimationFrame(loop); draw(); });
    $('#rp-reset').addEventListener('click', () => { training = false; seedIdx++; make(); draw(); });
    root.querySelectorAll('#rp-mode button').forEach((b) => b.addEventListener('click', () => {
      root.querySelectorAll('#rp-mode button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); H = +b.dataset.h; training = false; make(); draw();
    }));
    window.__REPR = { setSeeds(a) { SEEDS.length = 0; a.forEach((x) => SEEDS.push(x)); }, test(h, seed, n) { const oh = H; H = h; seedIdx = 0; SEEDS.length = 0; SEEDS.push(seed); make(); for (let i = 0; i < n; i++) step(); let ok = 0; P.forEach((q) => { if ((fwd(q.x, q.y).o > .5) === !!q.c) ok++; }); H = oh; make(); return ok / P.length; } };
    make();
    Deck.on('repr', (s, resized) => { if (!resized) training = false; draw(); });
  })();
})();
