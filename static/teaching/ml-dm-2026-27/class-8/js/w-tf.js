/* Act 6 widgets (transformers)
   1. reach : how many steps must information travel between two tokens? CNN vs RNN vs attention  (slide "reach")
   2. attn  : scaled dot-product self-attention, computed on toy numbers                           (slide "attention")
   3. lm    : next-token prediction as a loop (bigram toy + temperature)                           (slide "lm")      */
(function () {
  'use strict';

  /* =================================================================
     1. REACH
     ================================================================= */
  (function reach() {
    const root = document.getElementById('w-reach'); if (!root) return;
    const TOK = ['The', 'keys', 'to', 'the', 'old', 'wooden', 'cabinet', 'in', 'the', 'hall', 'are', 'missing'];
    const KEYS = 1;                       // the word that determines the form of the verb
    let mode = 'cnn', L = 3, tgt = 10;
    root.innerHTML = `
      <div class="row" style="margin-bottom:.6rem">
        <div class="seg" id="rc-mode"><button data-m="cnn" class="on">1-D CNN (k = 3)</button><button data-m="rnn">RNN / LSTM (left → right)</button><button data-m="tf">Self-attention</button></div>
        <label class="ctl" id="rc-lw" style="flex-direction:row;align-items:center;gap:.5rem;margin-left:.6rem"><span>stacked conv layers: <span class="val" id="rc-lv">3</span></span><input type="range" id="rc-l" min="1" max="10" value="3" style="width:9rem"></label>
      </div>
      <div class="small" style="margin-bottom:.3rem">Click a word to see where its information can come from. Shading = <b>number of steps</b> information needs to travel to it.</div>
      <div class="row" id="rc-row" style="gap:.35rem;flex-wrap:nowrap;overflow-x:auto;padding:.2rem 0 .1rem"></div>
      <div class="grid g2 mt" style="align-items:stretch">
        <div class="callout" id="rc-msg"></div>
        <div class="stat"><div class="l" style="margin-bottom:.15rem">The grammatical agreement to be learned</div><div style="font-size:1.05rem">The <b style="color:var(--accent)">keys</b> to the old wooden cabinet in the hall <b>are</b> missing.</div><div class="tiny">“are” (plural) depends on “keys”, 9 tokens away — not on the nearer, singular “hall”.</div></div>
      </div>`;
    const $ = (s) => root.querySelector(s);
    function path(i, j) {                 // steps needed for token i to influence token j (null = cannot)
      if (i === j) return 0;
      if (mode === 'cnn') { const d = Math.abs(i - j); return d <= L ? d : null; }
      if (mode === 'rnn') return i < j ? j - i : null;
      return 1;
    }
    function draw() {
      $('#rc-lw').style.display = mode === 'cnn' ? 'flex' : 'none'; $('#rc-lv').textContent = L;
      const row = $('#rc-row'); row.innerHTML = '';
      TOK.forEach((w, i) => {
        const p = path(i, tgt), d = document.createElement('div');
        let bg = '#f1f0ea', fg = '#9aa3b0', bd = '#e2dfd5';
        if (p === 0) { bg = '#17202c'; fg = '#fff'; bd = '#17202c'; }
        else if (p != null) { const a = Math.max(.18, 1 - (p - 1) / 10); bg = 'rgba(40,89,224,' + a * .85 + ')'; fg = a > .5 ? '#fff' : '#17202c'; bd = '#2859e0'; }
        d.style.cssText = 'flex:1 1 0;min-width:4.2rem;text-align:center;border:2px solid ' + (i === KEYS ? '#e8712d' : bd) + ';background:' + bg + ';color:' + fg + ';border-radius:10px;padding:.5rem .2rem;cursor:pointer;font-weight:700;' + (i === KEYS ? 'box-shadow:0 0 0 2px #fdeadd;' : '');
        d.innerHTML = w + '<div style="font-size:.66rem;font-weight:600;opacity:.9">' + (p === 0 ? 'target' : p == null ? 'out of reach' : p + (p === 1 ? ' step' : ' steps')) + '</div>';
        d.addEventListener('click', () => { tgt = i; draw(); }); row.appendChild(d);
      });
      const p = path(KEYS, tgt), msg = $('#rc-msg'), tn = TOK[tgt];
      if (tgt === KEYS) msg.innerHTML = 'Select another word to see how far information must travel from <b>“keys”</b>.';
      else if (mode === 'cnn') msg.innerHTML = p != null ? `With <b>${L}</b> stacked 3-wide conv layers, “${tn}” can reach “keys” (<b>${p}</b> layers needed). Receptive field = 2·${L}+1 = <b>${2 * L + 1}</b> tokens.`
        : `<b>“${tn}” cannot see “keys”</b> with ${L} layers: the receptive field grows by only 1 token per side per layer — the same arithmetic as for images. Needs ≥ ${Math.abs(tgt - KEYS)} layers (or dilated convolutions).`;
      else if (mode === 'rnn') msg.innerHTML = p != null ? `Information about “keys” must pass through <b>${p} sequential updates</b> of one fixed-size memory vector — and each step can overwrite it. Reachable in principle, hard in practice (next slides). Also: steps cannot run in parallel.`
        : `A left-to-right RNN has not read “${tn}”’s <i>right-hand</i> context yet: later words are out of reach (bidirectional RNNs add a second pass).`;
      else msg.innerHTML = `<b>Every token is exactly 1 step from every other</b>: attention compares “${tn}” with all tokens directly. The price: n² pairwise scores (cost grows quadratically with length).`;
    }
    root.querySelectorAll('#rc-mode button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#rc-mode button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); mode = b.dataset.m; draw(); }));
    $('#rc-l').addEventListener('input', (e) => { L = +e.target.value; draw(); });
    draw();
  })();

  /* =================================================================
     2. SELF-ATTENTION, computed on toy numbers
     ================================================================= */
  (function attn() {
    const root = document.getElementById('w-attn'); if (!root) return;
    const T = ['animal', 'didn’t', 'cross', 'street', 'because', 'it'];
    const K = [[2.0, 0.8], [-1.6, 1.4], [-0.6, -1.8], [1.4, -1.5], [-2.1, -0.3], [1.6, 0.1]];    // toy keys (= values, for clarity)
    const Q = [[-1.4, 1.2], [-0.5, -1.7], [1.2, -1.4], [-0.5, -1.6], [-1.5, 1.3], [1.9, 0.8]];    // toy queries
    const COL = ['#e8712d', '#7c4fd6', '#0d8f7f', '#2859e0', '#8e8a7a', '#c0392b'];
    const q = Q.map((v) => v.slice()); let sel = 5, drag = false;
    root.innerHTML = `
      <div class="grid g-side-r" style="align-items:start">
        <div>
          <canvas id="at-cv" class="cv" style="cursor:crosshair"></canvas>
          <div class="tiny" style="margin-top:.25rem">Left: a toy map of the words. <b>Drag</b> the tip of the arrow (what this word is looking for). ★ = the blended result.</div>
        </div>
        <div>
          <div class="small" style="margin-bottom:.3rem">Pick the word that is “asking” (its <b>query</b>):</div>
          <div class="row" id="at-tok" style="gap:.3rem;margin-bottom:.6rem"></div>
          <svg id="at-arcs" viewBox="0 0 560 120" style="width:100%;height:auto;display:block" role="img" aria-label="Attention arcs from the selected word to every word"></svg>
          <table class="t" style="font-size:.85rem;margin-top:.3rem" id="at-tab"></table>
          <div class="callout mt" id="at-msg" style="font-size:.92rem"></div>
        </div>
      </div>`;
    const $ = (s) => root.querySelector(s), cv = $('#at-cv'), S = 330, LIM = 2.8;
    const X = (v) => ((v + LIM) / (2 * LIM)) * S, Y = (v) => S - ((v + LIM) / (2 * LIM)) * S;
    function weights(j) {
      const s = K.map((k) => (q[j][0] * k[0] + q[j][1] * k[1]) / Math.SQRT2), m = Math.max(...s), e = s.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
      return { s, w: e.map((v) => v / z) };
    }
    function draw() {
      const { s, w } = weights(sel), out = [0, 1].map((d) => w.reduce((a, wi, i) => a + wi * K[i][d], 0));
      const ctx = U.fit(cv, S, S); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, S, S);
      ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(0, S / 2); ctx.lineTo(S, S / 2); ctx.moveTo(S / 2, 0); ctx.lineTo(S / 2, S); ctx.stroke();
      K.forEach((k, i) => { ctx.fillStyle = COL[i]; ctx.globalAlpha = .35 + .65 * w[i]; ctx.beginPath(); ctx.arc(X(k[0]), Y(k[1]), 6 + 12 * w[i], 0, 6.3); ctx.fill(); ctx.globalAlpha = 1;
        ctx.fillStyle = '#17202c'; ctx.font = '700 11px system-ui'; ctx.textAlign = 'center'; ctx.fillText(T[i], X(k[0]), Y(k[1]) - 12 - 10 * w[i]); });
      // query arrow
      const qx = X(q[sel][0]), qy = Y(q[sel][1]); ctx.strokeStyle = '#17202c'; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(qx, qy); ctx.stroke();
      ctx.fillStyle = '#17202c'; ctx.beginPath(); ctx.arc(qx, qy, 8, 0, 6.3); ctx.fill(); ctx.fillStyle = '#fff'; ctx.font = '800 9px system-ui'; ctx.fillText('q', qx, qy + 3);
      // output star
      ctx.fillStyle = '#17202c'; ctx.font = '800 20px system-ui'; ctx.fillText('★', X(out[0]), Y(out[1]) + 7);
      // arcs
      const arcs = $('#at-arcs'), NS = 'http://www.w3.org/2000/svg'; arcs.innerHTML = '';
      const xs = T.map((_, i) => 46 + i * 93);
      T.forEach((t, i) => {
        const r = document.createElementNS(NS, 'rect'); r.setAttribute('x', xs[i] - 40); r.setAttribute('y', 88); r.setAttribute('width', 80); r.setAttribute('height', 26); r.setAttribute('rx', 7);
        r.setAttribute('fill', i === sel ? '#17202c' : '#f3f1ea'); arcs.appendChild(r);
        const tx = document.createElementNS(NS, 'text'); tx.setAttribute('x', xs[i]); tx.setAttribute('y', 106); tx.setAttribute('text-anchor', 'middle'); tx.setAttribute('font-size', 13); tx.setAttribute('font-weight', 700); tx.setAttribute('fill', i === sel ? '#fff' : '#17202c'); tx.textContent = t; arcs.appendChild(tx);
        if (i !== sel || true) {
          const a = document.createElementNS(NS, 'path'), x1 = xs[sel], x2 = xs[i], h = i === sel ? 26 : Math.min(82, 14 + Math.abs(x2 - x1) * .22);
          a.setAttribute('d', i === sel ? `M${x1 - 14} 86 C ${x1 - 14} ${86 - h * 1.6}, ${x1 + 14} ${86 - h * 1.6}, ${x1 + 14} 86` : `M${x1} 86 C ${x1} ${86 - h * 1.5}, ${x2} ${86 - h * 1.5}, ${x2} 86`);
          a.setAttribute('fill', 'none'); a.setAttribute('stroke', COL[i]); a.setAttribute('stroke-width', 1 + 11 * w[i]); a.setAttribute('opacity', .2 + .75 * w[i]); a.setAttribute('stroke-linecap', 'round'); arcs.appendChild(a);
        }
      });
      $('#at-tab').innerHTML = '<tr><th>word</th><th>match score</th><th>attention weight</th></tr>' + T.map((t, i) => `<tr><td><span style="color:${COL[i]};font-weight:800">●</span> ${t}</td><td class="mono">${s[i].toFixed(2)}</td><td><div style="display:flex;align-items:center;gap:.4rem"><div style="height:.7rem;border-radius:4px;background:${COL[i]};width:${Math.max(1, w[i] * 100)}%;max-width:70%"></div><b class="mono">${w[i].toFixed(2)}</b></div></td></tr>`).join('');
      const top = w.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])[0];
      $('#at-msg').innerHTML = `“<b>${T[sel]}</b>” now carries a blend of the other words, weighted by attention (★): mostly “${T[top[1]]}” (${Math.round(top[0] * 100)}%). Its meaning has become <b>contextual</b>.`;
      const tk = $('#at-tok'); if (!tk.children.length) T.forEach((t, i) => { const b = document.createElement('button'); b.className = 'btn small'; b.textContent = t; b.addEventListener('click', () => { sel = i; draw(); }); tk.appendChild(b); });
      Array.from(tk.children).forEach((b, i) => b.classList.toggle('on', i === sel));
    }
    function move(e) { const p = U.pos(cv, e); q[sel] = [U.clamp((p.x / S) * 2 * LIM - LIM, -LIM, LIM), U.clamp(((S - p.y) / S) * 2 * LIM - LIM, -LIM, LIM)]; draw(); }
    cv.addEventListener('pointerdown', (e) => { cv.setPointerCapture(e.pointerId); drag = true; move(e); });
    cv.addEventListener('pointermove', (e) => { if (drag) move(e); });
    cv.addEventListener('pointerup', () => { drag = false; });
    Deck.on('attention', draw);
  })();

  /* =================================================================
     3. NEXT-TOKEN PREDICTION (bigram toy with temperature)
     ================================================================= */
  (function lm() {
    const root = document.getElementById('w-lm'); if (!root) return;
    const CORPUS = `the model reads the text and predicts the next word . the model learns the structure of the language from the data . the network learns features from the data . the network reads the image and predicts the class . a convolutional network reads the image with small filters . a transformer reads the text and every word attends to every other word . the model predicts the next word and then the next word again . the model generates the text one word at a time . the transformer learns the structure of the text from the data . the network predicts the next word from the text .`;
    const toks = CORPUS.split(/\s+/), cnt = {};
    for (let i = 0; i < toks.length - 1; i++) { (cnt[toks[i]] = cnt[toks[i]] || {}); cnt[toks[i]][toks[i + 1]] = (cnt[toks[i]][toks[i + 1]] || 0) + 1; }
    let ctx = ['the'], temp = 1.0; const rnd = U.mulberry32(Date.now() % 100000);
    root.innerHTML = `
      <div class="card flat" style="padding:.8rem 1rem"><div class="tiny" style="font-weight:800;margin-bottom:.3rem">TEXT SO FAR (the “prompt” + what has been generated)</div><div id="lm-txt" style="font-size:1.25rem;line-height:1.6;min-height:3.2rem"></div></div>
      <div class="grid g-side mt" style="align-items:start">
        <div><div class="tiny" style="font-weight:800;margin-bottom:.25rem">THE MODEL’S DISTRIBUTION OVER THE NEXT TOKEN — click a bar to choose it</div><div id="lm-bars"></div></div>
        <div>
          <label class="ctl"><span>Temperature T = <span class="val" id="lm-tv">1.0</span> <span class="tiny">(low = safe &amp; repetitive · high = random)</span></span><input type="range" id="lm-t" min="0.2" max="2" step="0.1" value="1"></label>
          <div class="row mt"><button class="btn primary" id="lm-go">Sample next token</button><button class="btn small" id="lm-10">×10</button><button class="btn small" id="lm-reset">Reset</button></div>
          <div class="callout mt" style="font-size:.9rem">This toy looks <b>only at the previous word</b> (a table counted from a 111-word text). A real model looks at the whole conversation so far, but the <b>loop is the same</b>: predict → pick → append → repeat.</div>
        </div>
      </div>`;
    const $ = (s) => root.querySelector(s);
    function dist() {
      const last = ctx[ctx.length - 1], c = cnt[last] || { '.': 1 }, ent = Object.entries(c);
      const w = ent.map(([t, n]) => [t, Math.pow(n, 1 / temp)]), z = w.reduce((a, b) => a + b[1], 0);
      return w.map(([t, v]) => [t, v / z]).sort((a, b) => b[1] - a[1]);
    }
    function draw() {
      $('#lm-txt').innerHTML = ctx.map((t, i) => `<span style="${i === ctx.length - 1 ? 'background:#e6edfd;border-radius:6px;padding:0 .25rem;' : ''}">${t}</span>`).join(' ') + ' <span style="color:var(--muted)">▌</span>';
      const d = dist().slice(0, 8);
      $('#lm-bars').innerHTML = d.map(([t, p]) => `<div class="lm-b" data-t="${t}" style="display:flex;align-items:center;gap:.5rem;margin-bottom:.25rem;cursor:pointer"><span class="mono" style="width:6.5rem;text-align:right;font-weight:700">${t}</span><div style="flex:1;background:var(--surface-2);border-radius:5px"><div style="height:1.15rem;border-radius:5px;background:var(--accent);width:${p * 100}%;min-width:2px"></div></div><b class="mono" style="width:3rem">${(p * 100).toFixed(0)}%</b></div>`).join('');
      root.querySelectorAll('.lm-b').forEach((b) => b.addEventListener('click', () => { ctx.push(b.dataset.t); draw(); }));
    }
    function sample() { const d = dist(); let r = rnd(), a = 0; for (const [t, p] of d) { a += p; if (r <= a) return t; } return d[0][0]; }
    $('#lm-go').addEventListener('click', () => { ctx.push(sample()); draw(); });
    $('#lm-10').addEventListener('click', () => { for (let i = 0; i < 10; i++) ctx.push(sample()); draw(); });
    $('#lm-reset').addEventListener('click', () => { ctx = ['the']; draw(); });
    $('#lm-t').addEventListener('input', (e) => { temp = +e.target.value; $('#lm-tv').textContent = temp.toFixed(1); draw(); });
    draw();
  })();
})();
