/* Widgets for the representation-learning part
   1. par      : designed filters vs learned filters on the same digit        (slide "paradigm")
   2. routes   : two routes to learn a representation without labels          (slide "routes")
   3. aediag   : autoencoder diagram with a bottleneck toggle                 (slide "aediag")
   */
(function () {
  'use strict';

  /* ---------- helpers ---------- */
  function digitURL(arr, cell, blur) {
    const c = document.createElement('canvas'); c.width = c.height = 8;
    const x = c.getContext('2d'), d = blur ? boxBlur(arr) : arr;
    for (let r = 0; r < 8; r++) for (let q = 0; q < 8; q++) { const v = Math.round(255 * (1 - U.clamp(d[r * 8 + q], 0, 1))); x.fillStyle = `rgb(${v},${v},${v})`; x.fillRect(q, r, 1, 1); }
    return c.toDataURL();
  }
  function boxBlur(a) {
    const o = new Float32Array(64);
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) { let s = 0, n = 0; for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) { const rr = r + i, cc = c + j; if (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) { s += a[rr * 8 + cc]; n++; } } o[r * 8 + c] = s / n * .9 + a[r * 8 + c] * .1; }
    return o;
  }
  const conv3 = (img, k) => { const o = new Float32Array(36); for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) { let s = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += img[(r + i) * 8 + c + j] * k[i * 3 + j]; o[r * 6 + c] = Math.max(0, s); } return o; };

  /* =================================================================
     1. DESIGNED vs LEARNED FILTERS
     ================================================================= */
  (function par() {
    const root = document.getElementById('w-par'); if (!root) return;
    const D = U.digits();
    const DESIGNED = [{ n: 'vertical edges', k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] }, { n: 'horizontal edges', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] }, { n: 'blur', k: Array(9).fill(1 / 9) }];
    let di = 7;
    root.innerHTML = `
      <div class="row" style="margin-bottom:.4rem"><button class="btn small" id="pa-next">Another digit</button><span class="small">Same operation in both rows — only the origin of the 3×3 numbers differs.</span></div>
      <div class="par-row"><div class="par-lab" style="border-color:#8a93a3"><b>Before</b><br>numbers chosen by an expert</div><canvas id="pa-a"></canvas><div class="par-arrow">→ then train a classifier on these features</div></div>
      <div class="par-row"><div class="par-lab" style="border-color:var(--enc)"><b style="color:var(--enc)">Now</b><br>numbers learned from data</div><canvas id="pa-b"></canvas><div class="par-arrow">→ classifier <b>and</b> filters trained together</div></div>`;
    const $ = (s) => root.querySelector(s);
    function row(cv, list) {
      const ks = 12, ms = 14, gap = 30, colw = 6 * ms, img = 8 * 12, W = img + 24 + list.length * (colw + gap), H = 3 * ks + 8 + 6 * ms + 14;
      const ctx = U.fit(cv, W, H), x = D.X.subarray(di * 64, di * 64 + 64);
      U.drawGrid(ctx, x, 8, 0, 8, 12);
      ctx.fillStyle = '#667180'; ctx.font = '700 10px system-ui,sans-serif'; ctx.fillText('image', 0, 6);
      list.forEach((f, i) => {
        const x0 = img + 24 + i * (colw + gap), m = Math.max(...f.k.map(Math.abs));
        for (let j = 0; j < 9; j++) { ctx.fillStyle = U.div(f.k[j] / m); ctx.fillRect(x0 + (j % 3) * ks + (colw - 3 * ks) / 2, Math.floor(j / 3) * ks, ks - 1, ks - 1); }
        const fm = conv3(x, f.k), mx = Math.max(1e-6, ...fm);
        for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) { ctx.fillStyle = U.ink(fm[r * 6 + c] / mx); ctx.fillRect(x0 + c * ms, 3 * ks + 8 + r * ms, ms - .5, ms - .5); }
        if (f.n) { ctx.fillStyle = '#3d4856'; ctx.font = '700 10px system-ui,sans-serif'; ctx.fillText(f.n, x0, H - 2); }
      });
    }
    function draw() { row($('#pa-a'), DESIGNED); row($('#pa-b'), LEARNED_FILTERS.map((k, i) => ({ k, n: 'filter ' + (i + 1) }))); }
    $('#pa-next').addEventListener('click', () => { di = (di * 7 + 13) % D.n; draw(); });
    Deck.on('paradigm', draw);
  })();

  /* =================================================================
     2. TWO ROUTES: compress or predict
     ================================================================= */
  (function routes() {
    const root = document.getElementById('w-routes'); if (!root) return;
    const D = U.digits(); let di = 12, seed = 5;
    root.innerHTML = `
      <div class="grid g2" style="align-items:stretch">
        <div class="card dec"><h3>① Compression</h3><p class="small" style="color:var(--ink-2)">Squeeze the data through a <b>narrow bottleneck</b>, then rebuild it. To rebuild well, z must keep what matters.</p>
          <svg viewBox="0 0 380 120" style="width:100%;height:auto"><defs><marker id="rt1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#667180"/></marker></defs>
            <image id="rt-x" x="6" y="22" width="64" height="64" style="image-rendering:pixelated" /><text x="38" y="104" text-anchor="middle" font-size="11" font-weight="800">x</text>
            <polygon points="90,20 170,45 170,65 90,90" fill="#dff3ef" stroke="#0d8f7f" stroke-width="2"/><text x="130" y="60" text-anchor="middle" font-size="11" font-weight="800" fill="#0a6b5e">encoder</text>
            <rect x="183" y="42" width="14" height="26" rx="3" fill="#fff" stroke="#17202c" stroke-width="2"/><rect x="200" y="42" width="14" height="26" rx="3" fill="#fff" stroke="#17202c" stroke-width="2"/><text x="198" y="36" text-anchor="middle" font-size="11" font-weight="800">z</text><text x="198" y="85" text-anchor="middle" font-size="9" fill="#555">few numbers</text>
            <polygon points="226,45 226,65 306,90 306,20" fill="#ece5fa" stroke="#7c4fd6" stroke-width="2"/><text x="268" y="60" text-anchor="middle" font-size="11" font-weight="800" fill="#5b36a8">decoder</text>
            <image id="rt-xh" x="316" y="22" width="64" height="64" style="image-rendering:pixelated" /><text x="348" y="104" text-anchor="middle" font-size="11" font-weight="800">x̂ ≈ x</text>
            <g stroke="#667180" stroke-width="2" marker-end="url(#rt1)"><line x1="72" y1="54" x2="88" y2="54"/><line x1="170" y1="55" x2="181" y2="55"/><line x1="216" y1="55" x2="224" y2="55"/><line x1="306" y1="55" x2="314" y2="55"/></g></svg>
          <ul class="clean small" style="color:var(--ink-2)"><li><b>Autoencoder</b> (Hinton &amp; Salakhutdinov, 2006)</li><li>Target = the input itself</li></ul></div>
        <div class="card acc"><h3>② Prediction (self-supervised)</h3><p class="small" style="color:var(--ink-2)">Hide part of the data and <b>predict it from the rest</b>. The labels come from the data itself.</p>
          <div class="row" style="gap:1rem;align-items:center"><div><canvas id="rt-m" class="cv"></canvas><div class="tiny" style="text-align:center">visible patches</div></div><div style="font-size:1.6rem">→ <b style="color:var(--accent)">?</b></div><div><canvas id="rt-t" class="cv"></canvas><div class="tiny" style="text-align:center">hidden → predict</div></div><button class="btn small" id="rt-again">Mask again</button></div>
          <ul class="clean small" style="color:var(--ink-2)"><li><b>Masked autoencoder</b> for images (He et al., 2022)</li><li><b>word2vec</b>: predict a word from its neighbours: “the cat ___ on the mat” (Mikolov et al., 2013)</li></ul></div>
      </div>`;
    const $ = (s) => root.querySelector(s);
    function draw() {
      const x = D.X.subarray(di * 64, di * 64 + 64);
      $('#rt-x').setAttribute('href', digitURL(x)); $('#rt-xh').setAttribute('href', digitURL(x, 0, true));
      const rnd = U.mulberry32(seed), hidden = Array.from({ length: 16 }, () => rnd() < .45), cs = 12;
      const c1 = U.fit($('#rt-m'), 8 * cs, 8 * cs), c2 = U.fit($('#rt-t'), 8 * cs, 8 * cs);
      for (let br = 0; br < 4; br++) for (let bc = 0; bc < 4; bc++) {
        const h = hidden[br * 4 + bc];
        for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { const r = br * 2 + i, c = bc * 2 + j, v = x[r * 8 + c];
          c1.fillStyle = h ? '#cfcabb' : U.ink(v); c1.fillRect(c * cs, r * cs, cs - .5, cs - .5);
          c2.fillStyle = h ? U.ink(v) : '#ece8dc'; c2.fillRect(c * cs, r * cs, cs - .5, cs - .5); }
      }
    }
    $('#rt-again').addEventListener('click', () => { seed++; draw(); });
    Deck.on('routes', draw);
  })();

  /* =================================================================
     3. AUTOENCODER DIAGRAM
     ================================================================= */
  (function aediag() {
    const root = document.getElementById('w-aediag'); if (!root) return;
    const D = U.digits(); let di = 7, narrow = true;
    root.innerHTML = `
      <div class="row" style="margin-bottom:.5rem;gap:.8rem"><div class="seg" id="ad-m"><button data-m="n" class="on">Narrow bottleneck: z has 2 numbers</button><button data-m="w">No bottleneck: z as wide as x</button></div><button class="btn small" id="ad-next">Another digit</button></div>
      <div class="card flat" style="padding:.6rem"><svg viewBox="0 0 860 300" style="width:100%;height:auto;display:block" role="img" aria-label="Autoencoder: image x goes through an encoder to a latent vector z, then through a decoder to a reconstruction x-hat; the loss compares x-hat with x.">
        <defs><marker id="ad-ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#667180"/></marker><marker id="ad-rh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#c0392b"/></marker></defs>
        <image id="ad-x" x="20" y="60" width="110" height="110" style="image-rendering:pixelated"/><rect x="20" y="60" width="110" height="110" fill="none" stroke="#17202c" stroke-width="2"/>
        <text x="75" y="48" text-anchor="middle" font-weight="800" font-size="15">input x</text><text x="75" y="190" text-anchor="middle" font-size="11" fill="#555">64 numbers (8×8)</text>
        <polygon id="ad-enc" points="170,40 360,95 360,135 170,190" fill="#dff3ef" stroke="#0d8f7f" stroke-width="2.5"/><text x="265" y="120" text-anchor="middle" font-weight="800" fill="#0a6b5e" font-size="16">Encoder f</text><text x="265" y="140" text-anchor="middle" font-size="11" fill="#0a6b5e">compresses</text>
        <g id="ad-z"></g>
        <polygon id="ad-dec" points="500,95 500,135 690,190 690,40" fill="#ece5fa" stroke="#7c4fd6" stroke-width="2.5"/><text x="595" y="120" text-anchor="middle" font-weight="800" fill="#5b36a8" font-size="16">Decoder g</text><text x="595" y="140" text-anchor="middle" font-size="11" fill="#5b36a8">rebuilds</text>
        <image id="ad-xh" x="730" y="60" width="110" height="110" style="image-rendering:pixelated"/><rect x="730" y="60" width="110" height="110" fill="none" stroke="#17202c" stroke-width="2"/>
        <text x="785" y="48" text-anchor="middle" font-weight="800" font-size="15">output x̂</text><text x="785" y="190" text-anchor="middle" font-size="11" fill="#555">64 numbers</text>
        <g stroke="#667180" stroke-width="2.5" marker-end="url(#ad-ah)"><line x1="132" y1="115" x2="168" y2="115"/><line x1="362" y1="115" x2="388" y2="115"/><line x1="472" y1="115" x2="498" y2="115"/><line x1="692" y1="115" x2="728" y2="115"/></g>
        <path d="M785 200 L785 250 L75 250 L75 202" fill="none" stroke="#c0392b" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#ad-rh)"/>
        <rect x="270" y="236" width="320" height="28" rx="8" fill="#fff" stroke="#c0392b" stroke-width="2"/><text x="430" y="255" text-anchor="middle" font-weight="800" font-size="14" fill="#c0392b">loss: compare the output x̂ with the INPUT x</text>
        <text x="430" y="290" text-anchor="middle" font-size="12" font-weight="700" fill="#667180" id="ad-note"></text>
      </svg></div>`;
    const $ = (s) => root.querySelector(s);
    function draw() {
      const x = D.X.subarray(di * 64, di * 64 + 64);
      $('#ad-x').setAttribute('href', digitURL(x)); $('#ad-xh').setAttribute('href', digitURL(x, 0, narrow));
      let z = '';
      if (narrow) z = `<rect x="408" y="92" width="26" height="46" rx="5" fill="#fff" stroke="#17202c" stroke-width="3"/><rect x="438" y="92" width="26" height="46" rx="5" fill="#fff" stroke="#17202c" stroke-width="3"/><text x="436" y="78" text-anchor="middle" font-weight="800" font-size="14">latent z</text><text x="436" y="160" text-anchor="middle" font-size="11" fill="#555">2 numbers</text>`;
      else { for (let i = 0; i < 8; i++) z += `<rect x="${390 + i * 11}" y="92" width="9" height="46" rx="2" fill="#fff" stroke="#17202c" stroke-width="2"/>`; z += `<text x="436" y="78" text-anchor="middle" font-weight="800" font-size="14">latent z</text><text x="436" y="160" text-anchor="middle" font-size="11" fill="#555">64 numbers</text>`; }
      $('#ad-z').innerHTML = z;
      $('#ad-enc').setAttribute('points', narrow ? '170,40 360,95 360,135 170,190' : '170,60 360,70 360,160 170,170');
      $('#ad-dec').setAttribute('points', narrow ? '500,95 500,135 690,190 690,40' : '500,70 500,160 690,170 690,60');
      $('#ad-note').textContent = narrow ? 'The bottleneck forces the network to keep only what matters to rebuild the digit (sketch: the rebuilt image is drawn blurrier).' : 'Without a bottleneck the network can just copy x: the loss is ≈ 0, but z has learned nothing useful.';
    }
    root.querySelectorAll('#ad-m button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#ad-m button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); narrow = b.dataset.m === 'n'; draw(); }));
    $('#ad-next').addEventListener('click', () => { di = (di * 7 + 13) % D.n; draw(); });
    Deck.on('aediag', draw);
  })();
  /* =================================================================
     4. DECODING RANDOM LATENT VECTORS OF A REAL AUTOENCODER
     ================================================================= */
  (function bridge() {
    const root = document.getElementById('w-bridge'); if (!root || !window.__AE) return;
    const D = U.digits(), N = D.n, COL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
    let ae = null, codes = null, box = null, seed = 1, pts = [];
    root.innerHTML = `<div class="row" style="margin-bottom:.5rem;gap:.7rem"><span class="small" id="br-st">Training a small autoencoder (2-number latent vector)…</span><span class="spacer"></span><button class="btn primary" id="br-again" disabled>Draw new random vectors</button></div>
      <ol class="small" style="margin:.2rem 0 .6rem 1.1rem;padding:0;color:var(--ink-2)"><li>The encoder turns every real digit into <b>2 numbers</b>: a point on the map (grey dots).</li><li>The decoder turns a point back into an image. For the grey dots it works: the bottom row looks like digits.</li><li>Now pick points <b>at random</b> (orange numbers) and decode them. Points that land in empty regions give odd, unrealistic images.</li></ol>
      <div class="row top" style="gap:1.2rem;flex-wrap:wrap;align-items:flex-start">
        <div><div class="tiny" style="font-weight:800">LATENT SPACE: one dot per real digit</div><canvas id="br-sc" style="border:1px solid var(--line);border-radius:10px;background:#fff"></canvas><div class="tiny">grey dots: where the encoder puts the 1,000 real digits<br><b style="color:var(--gen)">orange numbers</b>: random vectors drawn uniformly from this square</div></div>
        <div style="flex:1;min-width:16rem"><div class="tiny" style="font-weight:800">DECODED FROM THE ORANGE VECTORS</div><div class="br-imgs" id="br-rnd"></div><div class="tiny" style="font-weight:800;margin-top:.8rem">DECODED FROM THE GREY DOTS (real digits’ own codes)</div><div class="br-imgs" id="br-real"></div></div></div>`;
    const $ = (s) => root.querySelector(s);
    const paint = (cv, a, cs) => { const c = U.fit(cv, 8 * cs, 8 * cs); for (let i = 0; i < 64; i++) { c.fillStyle = U.ink(a[i]); c.fillRect((i % 8) * cs, Math.floor(i / 8) * cs, cs, cs); } cv.style.border = '1px solid rgba(23,32,44,.25)'; };
    function tiles(el, vecs, labels) { el.innerHTML = ''; vecs.forEach((z, i) => { const o = ae.decode(Float64Array.from(z), 1).o, w = document.createElement('div'); w.innerHTML = '<canvas></canvas><div class="tiny" style="text-align:center;font-weight:800;color:var(--gen)">' + (labels ? i + 1 : '') + '</div>'; el.appendChild(w); paint(w.querySelector('canvas'), o, 7); }); }
    function draw() {
      const rnd = U.mulberry32(seed * 977); pts = Array.from({ length: 8 }, () => [U.lerp(box.x0, box.x1, rnd()), U.lerp(box.y0, box.y1, rnd())]);
      const S = 300, ctx = U.fit($('#br-sc'), S, S), X = (v) => 14 + (v - box.x0) / (box.x1 - box.x0) * (S - 28), Y = (v) => S - 14 - (v - box.y0) / (box.y1 - box.y0) * (S - 28);
      for (let i = 0; i < N; i++) { ctx.fillStyle = '#9aa3b2'; ctx.beginPath(); ctx.arc(X(codes[i * 2]), Y(codes[i * 2 + 1]), 2.6, 0, 7); ctx.fill(); }
      pts.forEach((p, i) => { ctx.fillStyle = '#e8712d'; ctx.font = '800 15px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 10, 0, 7); ctx.fill(); ctx.fillStyle = '#fff'; ctx.fillText(i + 1, X(p[0]), Y(p[1]) + .5); });
      tiles($('#br-rnd'), pts, true);
      const rr = U.mulberry32(seed * 31 + 5), real = Array.from({ length: 8 }, () => { const i = Math.floor(rr() * N); return [codes[i * 2], codes[i * 2 + 1]]; }); tiles($('#br-real'), real, false);
    }
    function start() {
      ae = new window.__AE.AE(2); let ep = 0;
      (function chunk() { for (let i = 0; i < 8 && ep < 160; i++, ep++) ae.trainEpoch(); $('#br-st').textContent = 'Training a small autoencoder (2-number latent vector)… epoch ' + ep + ' / 160'; if (ep < 160) { setTimeout(chunk, 0); return; }
        const X = new Float64Array(N * 64); for (let i = 0; i < N * 64; i++) X[i] = D.X[i]; const z = ae.encode(X, N).z; codes = z;
        let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; for (let i = 0; i < N; i++) { x0 = Math.min(x0, z[i * 2]); x1 = Math.max(x1, z[i * 2]); y0 = Math.min(y0, z[i * 2 + 1]); y1 = Math.max(y1, z[i * 2 + 1]); }
        const mx = (x1 - x0) * .05, my = (y1 - y0) * .05; box = { x0: x0 - mx, x1: x1 + mx, y0: y0 - my, y1: y1 + my };
        $('#br-st').textContent = 'A real autoencoder, trained in your browser on 800 digits (160 epochs).'; $('#br-again').disabled = false; draw(); })();
    }
    $('#br-again').addEventListener('click', () => { seed++; draw(); });
    let started = false; window.addEventListener('load', () => setTimeout(() => { if (!started) { started = true; start(); } }, 4000)); Deck.on('bridge', () => { if (!started) { started = true; setTimeout(start, 50); } else if (box) draw(); });
  })();
})();
