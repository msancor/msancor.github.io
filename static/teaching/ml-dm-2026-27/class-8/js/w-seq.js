/* Act 5 widgets (sequences)
   1. order : same words, different order -> identical bag-of-words input           (slide "order")
   2. rnn   : unrolled recurrent network, one token at a time                        (slide "rnn")
   3. grad  : gradient signal over distance, plain RNN vs gated (LSTM-like) path      (slide "lstm") */
(function () {
  'use strict';

  /* =================================================================
     1. ORDER MATTERS
     ================================================================= */
  (function order() {
    const root = document.getElementById('w-order'); if (!root) return;
    const PAIRS = [
      ['Dog bites man', 'Man bites dog'],
      ['Alice sent Bob money', 'Bob sent Alice money'],
      ['The film was good, not bad', 'The film was bad, not good'],
    ];
    root.innerHTML = `
      <div class="row" style="margin-bottom:.6rem"><span class="small">Examples:</span><div class="seg" id="or-pre">${PAIRS.map((p, i) => `<button data-i="${i}" class="${i === 0 ? 'on' : ''}">${p[0].split(' ').slice(0, 3).join(' ')}…</button>`).join('')}</div><span class="small">or type your own:</span></div>
      <div class="grid g2" style="gap:1rem">
        ${[0, 1].map((k) => `<div class="card flat"><input id="or-in${k}" type="text" style="width:100%;font:inherit;font-size:1.05rem;padding:.4rem .6rem;border:1px solid var(--line-strong);border-radius:8px">
          <div class="tiny mt" style="font-weight:700">AS A SEQUENCE (order kept)</div><div class="row" id="or-seq${k}" style="gap:.25rem;margin:.2rem 0 .5rem"></div>
          <div class="tiny" style="font-weight:700">AS WORD COUNTS (what an MLP on a bag of words receives)</div><div id="or-bow${k}" style="margin-top:.2rem"></div></div>`).join('')}
      </div>
      <div class="verdict mt" id="or-v"></div>`;
    const $ = (s) => root.querySelector(s);
    const toks = (s) => s.toLowerCase().replace(/[^a-z' ]/g, ' ').split(/\s+/).filter(Boolean);
    const bow = (t) => { const c = {}; t.forEach((w) => (c[w] = (c[w] || 0) + 1)); return c; };
    function render() {
      const T = [toks($('#or-in0').value), toks($('#or-in1').value)], B = T.map(bow);
      const vocab = Array.from(new Set(T[0].concat(T[1]))).sort();
      [0, 1].forEach((k) => {
        $('#or-seq' + k).innerHTML = T[k].map((w, i) => `<span style="background:var(--accent-soft);border-radius:6px;padding:.1rem .4rem;font-weight:700;font-size:.85rem"><sub style="color:var(--muted);font-weight:600">${i + 1}</sub> ${w}</span>`).join('');
        $('#or-bow' + k).innerHTML = '<table class="t" style="font-size:.8rem"><tr>' + vocab.map((w) => `<th style="padding:.15rem .3rem">${w}</th>`).join('') + '</tr><tr>' + vocab.map((w) => `<td style="padding:.15rem .3rem;font-weight:700;text-align:center">${B[k][w] || 0}</td>`).join('') + '</tr></table>';
      });
      const same = vocab.every((w) => (B[0][w] || 0) === (B[1][w] || 0)), differ = T[0].join(' ') !== T[1].join(' ');
      const v = $('#or-v');
      if (same && differ) { v.className = 'verdict warn'; v.innerHTML = '<b>Identical counts, different meaning.</b> An MLP that receives word counts cannot tell these sentences apart — the order, which carries the meaning, is gone.'; }
      else if (!differ) { v.className = 'verdict'; v.textContent = 'Type two different sentences.'; }
      else { v.className = 'verdict ok'; v.textContent = 'These two have different word counts, so a bag-of-words model can see a difference — but only through the vocabulary, not through order.'; }
    }
    function load(i) { $('#or-in0').value = PAIRS[i][0]; $('#or-in1').value = PAIRS[i][1]; render(); }
    root.querySelectorAll('#or-pre button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#or-pre button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); load(+b.dataset.i); }));
    ['#or-in0', '#or-in1'].forEach((s) => $(s).addEventListener('input', render));
    load(0);
  })();

  /* =================================================================
     2. UNROLLED RNN
     ================================================================= */
  (function rnn() {
    const root = document.getElementById('w-rnn'); if (!root) return;
    const W = ['The', 'keys', 'to', 'the', 'cabinet', 'are'];
    root.innerHTML = `
      <div class="card flat" style="padding:.6rem"><svg id="rn-svg" viewBox="0 0 800 270" style="width:100%;height:auto;display:block" role="img" aria-label="A recurrent network unrolled over six words"></svg></div>
      <div class="row mt"><button class="btn small" id="rn-prev">◂ Back</button><button class="btn primary" id="rn-next">Read next word ▸</button><button class="btn small" id="rn-reset">Reset</button><span class="spacer"></span><span class="tiny">h₀ = 0 · the same weights W<sub>x</sub>, W<sub>h</sub> are used at every step</span></div>
      <div class="callout mt" id="rn-msg"></div>`;
    const svg = root.querySelector('#rn-svg'), NS = 'http://www.w3.org/2000/svg';
    let t = 0; const CW = 128, X0 = 22;
    const el = (tag, a, p) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); (p || svg).appendChild(e); return e; };
    function draw() {
      svg.innerHTML = '';
      const defs = el('defs', {}); defs.innerHTML = '<marker id="rn-ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#667180"/></marker>';
      W.forEach((w, i) => {
        const x = X0 + i * CW, done = i < t, cur = i === t - 1;
        el('rect', { x, y: 100, width: 100, height: 60, rx: 12, fill: cur ? '#fdeadd' : (done ? '#fef3ea' : '#f4f2ea'), stroke: cur ? '#e8712d' : (done ? '#e8712d' : '#c9cdd4'), 'stroke-width': cur ? 3.5 : 2 });
        const a = el('text', { x: x + 50, y: 128, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 800, fill: done ? '#a14a12' : '#9aa3b0' }); a.textContent = 'h' + (i + 1);
        const b = el('text', { x: x + 50, y: 147, 'text-anchor': 'middle', 'font-size': 10, fill: '#667180' }); b.textContent = done ? 'read ' + (i + 1) + (i ? ' words' : ' word') : '—';
        // input word
        el('rect', { x: x + 10, y: 205, width: 80, height: 34, rx: 8, fill: cur ? '#e6edfd' : '#fff', stroke: cur ? '#2859e0' : '#c9cdd4', 'stroke-width': 2 });
        const c = el('text', { x: x + 50, y: 227, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 700, fill: done ? '#17202c' : '#9aa3b0' }); c.textContent = w;
        el('line', { x1: x + 50, y1: 203, x2: x + 50, y2: 162, stroke: done ? '#2859e0' : '#c9cdd4', 'stroke-width': 2, 'marker-end': 'url(#rn-ah)' });
        const d = el('text', { x: x + 56, y: 188, 'font-size': 10, fill: '#667180' }); d.textContent = 'x' + (i + 1);
        // recurrent arrow
        if (i < W.length - 1) el('line', { x1: x + 102, y1: 130, x2: x + CW - 2, y2: 130, stroke: i < t - 1 ? '#e8712d' : '#c9cdd4', 'stroke-width': i < t - 1 ? 3.5 : 2, 'marker-end': 'url(#rn-ah)' });
        if (i < W.length - 1 && i === 0) { const e = el('text', { x: x + 115, y: 118, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700, fill: '#a14a12' }); e.textContent = 'W_h'; }
      });
      const h0 = el('text', { x: 4, y: 134, 'font-size': 11, fill: '#667180' }); h0.textContent = '';
      const hd = el('text', { x: 400, y: 26, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 800, fill: '#3d4856' }); hd.textContent = 'h_t = tanh( W_x · x_t + W_h · h_(t−1) + b )';
      const msg = root.querySelector('#rn-msg');
      if (t === 0) msg.innerHTML = 'Press <b>Read next word</b>. The network reads the sentence <b>one word at a time</b>, keeping a running summary <b>h</b> (the hidden state) of everything read so far.';
      else if (t < W.length) msg.innerHTML = `<b>Step ${t}:</b> reads “<b>${W[t - 1]}</b>” and combines it with the memory <b>h${t - 1 || '₀'}</b> of what came before → new memory <b>h${t}</b>. Same weights as in every other step.`;
      else msg.innerHTML = `<b>Step ${t}:</b> to predict the verb form after “${W[5]}”, h₆ must still carry that the subject was <i>“keys”</i> (plural) — read <b>five steps ago</b> — squeezed into one fixed-size vector.`;
      root.querySelector('#rn-prev').disabled = t === 0; root.querySelector('#rn-next').disabled = t === W.length;
    }
    root.querySelector('#rn-next').addEventListener('click', () => { if (t < W.length) { t++; draw(); } });
    root.querySelector('#rn-prev').addEventListener('click', () => { if (t > 0) { t--; draw(); } });
    root.querySelector('#rn-reset').addEventListener('click', () => { t = 0; draw(); });
    draw();
  })();

  /* =================================================================
     3. VANISHING GRADIENT OVER TIME (scalar toy model)
     ================================================================= */
  (function grad() {
    const root = document.getElementById('w-grad'); if (!root) return;
    root.innerHTML = `
      <div class="grid g-side" style="align-items:start">
        <div><canvas id="gr-cv" class="cv"></canvas><div class="tiny" style="margin-top:.2rem">y-axis is logarithmic: each gridline is ×10.</div></div>
        <div>
          <div class="grid" style="gap:.8rem">
            <label class="ctl"><span><b style="color:var(--gen)">Plain RNN</b>: factor per step r = |W_h|·tanh′ ≈ <span class="val" id="gr-rv">0.90</span></span><input type="range" id="gr-r" min="0.5" max="1.15" step="0.01" value="0.9"></label>
            <label class="ctl"><span><b style="color:var(--enc)">LSTM cell path</b>: forget gate f = <span class="val" id="gr-fv">0.99</span></span><input type="range" id="gr-f" min="0.8" max="1" step="0.005" value="0.99"></label>
            <label class="ctl"><span>Sequence length T = <span class="val" id="gr-tv">60</span> steps</span><input type="range" id="gr-t" min="10" max="120" step="5" value="60"></label>
          </div>
          <div class="grid g2 mt" style="gap:.6rem">
            <div class="stat hot"><div class="n" id="gr-a">–</div><div class="l">plain RNN: gradient from <b>20 steps</b> back, relative to 1 step back</div></div>
            <div class="stat cool"><div class="n" id="gr-b">–</div><div class="l">LSTM cell path: same distance</div></div>
          </div>
          <div class="callout mt" id="gr-msg"></div>
        </div>
      </div>`;
    const $ = (s) => root.querySelector(s), cv = $('#gr-cv');
    const sci = (v) => { if (v >= 0.01 && v < 1000) return v.toFixed(v < 1 ? 3 : 1); const [m, e] = v.toExponential(1).split('e'); return m + '×10<sup>' + parseInt(e, 10) + '</sup>'; };
    function draw() {
      const r = +$('#gr-r').value, f = +$('#gr-f').value, T = +$('#gr-t').value;
      $('#gr-rv').textContent = r.toFixed(2); $('#gr-fv').textContent = f.toFixed(3).replace(/0$/, ''); $('#gr-tv').textContent = T;
      const W = Math.min(root.clientWidth * 0.58, 560), H = 280, ctx = U.fit(cv, W, H), L = 40, B = 26, lo = -8, hi = 4;
      const X = (k) => L + (k / T) * (W - L - 8), Y = (v) => 8 + (1 - (Math.log10(Math.max(v, 1e-12)) - lo) / (hi - lo)) * (H - B - 8);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
      ctx.font = '600 10px system-ui'; ctx.textAlign = 'right'; ctx.fillStyle = '#667180';
      for (let e = lo; e <= hi; e += 2) { ctx.strokeStyle = e === 0 ? '#9aa3b0' : '#eee'; ctx.beginPath(); ctx.moveTo(L, Y(10 ** e)); ctx.lineTo(W - 8, Y(10 ** e)); ctx.stroke(); ctx.fillText('10' + (e < 0 ? '⁻' : '') + String(Math.abs(e)).replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]), L - 5, Y(10 ** e) + 3); }
      ctx.textAlign = 'center'; ctx.fillText('how many steps back (distance between the loss and the input)  →', (L + W) / 2, H - 6);
      [[r, '#e8712d'], [f, '#0d8f7f']].forEach(([fac, col]) => { ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.beginPath(); for (let k = 0; k <= T; k++) { const x = X(k), y = Y(Math.pow(fac, k)); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); });
      const a = Math.pow(r, 19), b = Math.pow(f, 19);
      $('#gr-a').innerHTML = sci(a); $('#gr-b').innerHTML = sci(b);
      const m = $('#gr-msg');
      if (r < 0.97) m.innerHTML = '<b>Vanishing:</b> with r < 1 the learning signal from distant words shrinks exponentially (r<sup>k</sup>), so the network cannot learn that the verb depends on a subject 20 words back.';
      else if (r > 1.03) m.innerHTML = '<b>Exploding:</b> with r > 1 the signal grows exponentially — unstable training (usually tamed by gradient clipping).';
      else m.innerHTML = 'r ≈ 1 is a knife-edge: a plain RNN would have to stay exactly there for the whole training run.';
      m.innerHTML += ' <b>The LSTM’s additive cell-state path</b> multiplies by a <i>learned</i> gate f instead — it can stay close to 1 and carry information (and gradients) over long spans.';
    }
    ['#gr-r', '#gr-f', '#gr-t'].forEach((s) => $(s).addEventListener('input', draw));
    Deck.on('lstm', draw);
  })();
})();
