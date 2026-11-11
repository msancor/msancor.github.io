/* Widgets for the representation-learning part
   1. par      : designed filters vs learned filters on the same digit        (slide "paradigm")
   2. routes   : two routes to learn a representation without labels          (slide "routes")
   3. aediag   : autoencoder diagram with a bottleneck toggle                 (slide "aediag")
   4. pcai     : PCA intuition: the direction that keeps the most spread      (slide "pcaintuit") */
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
        <rect x="270" y="236" width="320" height="28" rx="8" fill="#fff" stroke="#c0392b" stroke-width="2"/><text x="430" y="255" text-anchor="middle" font-weight="800" font-size="14" fill="#c0392b">loss = ‖ x − x̂ ‖²   (compare with the INPUT)</text>
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
     4. PCA INTUITION
     ================================================================= */
  (function pcai() {
    const root = document.getElementById('w-pcai'); if (!root) return;
    const rnd = U.mulberry32(21), N = 60, pts = [];
    for (let i = 0; i < N; i++) { const t = U.gauss(rnd) * 1.0, e = U.gauss(rnd) * 0.28; const a = 0.52; pts.push([t * Math.cos(a) - e * Math.sin(a), t * Math.sin(a) + e * Math.cos(a)]); }
    const mean = [0, 1].map((k) => pts.reduce((s, p) => s + p[k], 0) / N);
    const P = pts.map((p) => [p[0] - mean[0], p[1] - mean[1]]);
    const Sxx = P.reduce((s, p) => s + p[0] * p[0], 0), Syy = P.reduce((s, p) => s + p[1] * p[1], 0), Sxy = P.reduce((s, p) => s + p[0] * p[1], 0), TOT = Sxx + Syy;
    const best = 0.5 * Math.atan2(2 * Sxy, Sxx - Syy);
    let th = 2.3, anim = null;
    root.innerHTML = `
      <div class="row" style="margin-bottom:.5rem;gap:.9rem"><label class="ctl" style="margin:0;min-width:15rem">Direction of the line <span class="val" id="pc-a">–</span><input type="range" id="pc-s" min="0" max="180" step="1" value="130"></label><button class="btn primary" id="pc-best">Find the best direction</button></div>
      <div class="grid g-side" style="align-items:center">
        <div class="card flat" style="padding:.4rem"><canvas id="pc-cv" style="display:block;width:100%"></canvas></div>
        <div>
          <div class="stat cool"><div class="n" id="pc-k" style="font-size:1.3rem">–</div><div class="l">of the spread kept by 1 number per point</div></div>
          <div class="stat hot mt"><div class="n" id="pc-e" style="font-size:1.3rem">–</div><div class="l">lost when rebuilding the point from that 1 number</div></div>
          <div class="callout mt" id="pc-m" style="font-size:.92rem"></div>
        </div>
      </div>`;
    const $ = (s) => root.querySelector(s), cv = $('#pc-cv');
    function draw() {
      const W = Math.max(300, Math.min(520, cv.parentElement.clientWidth - 12)), H = 300, ctx = U.fit(cv, W, H), sc = 62, cx = W / 2, cy = 150;
      const X = (v) => cx + v * sc, Y = (v) => cy - v * sc, u = [Math.cos(th), Math.sin(th)];
      ctx.strokeStyle = '#e2dfd5'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
      ctx.strokeStyle = '#e8712d'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(-3.4 * u[0]), Y(-3.4 * u[1])); ctx.lineTo(X(3.4 * u[0]), Y(3.4 * u[1])); ctx.stroke();
      let kept = 0;
      P.forEach((p) => { const t = p[0] * u[0] + p[1] * u[1]; kept += t * t; const fx = t * u[0], fy = t * u[1];
        ctx.strokeStyle = 'rgba(192,57,43,.55)'; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(X(p[0]), Y(p[1])); ctx.lineTo(X(fx), Y(fy)); ctx.stroke();
        ctx.fillStyle = '#e8712d'; ctx.beginPath(); ctx.arc(X(fx), Y(fy), 2.4, 0, 7); ctx.fill(); });
      P.forEach((p) => { ctx.fillStyle = '#2859e0'; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 4, 0, 7); ctx.fill(); });
      ctx.fillStyle = '#667180'; ctx.font = '600 11px system-ui'; ctx.fillText('blue: data (2 numbers each)', 8, 14); ctx.fillStyle = '#e8712d'; ctx.fillText('orange: rebuilt from 1 number', 8, 28); ctx.fillStyle = '#c0392b'; ctx.fillText('red: what is lost', 8, 42);
      const k = kept / TOT, deg = Math.round(th * 180 / Math.PI);
      $('#pc-a').textContent = deg + '°'; $('#pc-s').value = deg;
      $('#pc-k').textContent = (k * 100).toFixed(0) + ' %'; $('#pc-e').textContent = ((1 - k) * 100).toFixed(0) + ' %';
      const d = Math.abs(((th - best) % Math.PI + Math.PI * 1.5) % Math.PI - Math.PI / 2);
      $('#pc-m').innerHTML = d < 0.04 ? '<b>This is the first principal component</b>: the direction that keeps the most spread, hence loses the least.' : 'Turn the line: the more it follows the cloud, the less is lost.';
    }
    function norm(a) { return ((a % Math.PI) + Math.PI) % Math.PI; }
    $('#pc-s').addEventListener('input', (e) => { th = +e.target.value * Math.PI / 180; draw(); });
    $('#pc-best').addEventListener('click', () => { clearInterval(anim); const target = norm(best); let from = norm(th); if (target - from > Math.PI / 2) from += Math.PI; if (from - target > Math.PI / 2) from -= Math.PI; let t = 0; anim = setInterval(() => { t += 0.07; th = U.lerp(from, target, Math.min(1, t)); draw(); if (t >= 1) { clearInterval(anim); th = target; draw(); } }, 30); });
    Deck.on('pcaintuit', () => { clearInterval(anim); th = 2.3; draw(); });
  })();
  /* =================================================================
     5. LINEAR AUTOENCODER = PCA; NON-LINEAR CAN FOLLOW A CURVE
     ================================================================= */
  (function linae() {
    const root = document.getElementById('w-linae'); if (!root) return;
    const rnd = U.mulberry32(8), N = 70, pts = [];
    for (let i = 0; i < N; i++) { const t = -1.25 + 2.5 * (i + rnd() * .6) / N; pts.push([t, 0.9 * t * t - 0.45 + U.gauss(rnd) * 0.05]); }
    root.innerHTML = '<div class="grid g2"><div class="card flat" style="padding:.5rem"><div class="tiny" style="font-weight:800;color:#8a93a3">LINEAR: PCA / LINEAR AUTOENCODER</div><canvas id="la-a" style="width:100%"></canvas><div class="small" id="la-ea"></div></div><div class="card flat" style="padding:.5rem"><div class="tiny" style="font-weight:800;color:var(--dec)">NON-LINEAR AUTOENCODER</div><canvas id="la-b" style="width:100%"></canvas><div class="small" id="la-eb"></div></div></div>';
    const $ = (s) => root.querySelector(s), f = (t) => 0.9 * t * t - 0.45;
    function draw() {
      const W = Math.max(260, Math.min(460, $('#la-a').parentElement.clientWidth - 12)), Hh = 230, sc = 70, X = (v) => W / 2 + v * sc, Y = (v) => 118 - v * sc;
      const mx = pts.reduce((s, p) => s + p[0], 0) / N, my = pts.reduce((s, p) => s + p[1], 0) / N;
      let Sxx = 0, Syy = 0, Sxy = 0; pts.forEach((p) => { const a = p[0] - mx, b = p[1] - my; Sxx += a * a; Syy += b * b; Sxy += a * b; });
      const th = 0.5 * Math.atan2(2 * Sxy, Sxx - Syy), u = [Math.cos(th), Math.sin(th)];
      let eL = 0, eC = 0;
      [['#la-a', 0], ['#la-b', 1]].forEach(([q, mode]) => {
        const ctx = U.fit($(q), W, Hh); ctx.strokeStyle = '#e2dfd5'; ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.stroke();
        ctx.lineWidth = 3; ctx.strokeStyle = mode ? '#7c4fd6' : '#8a93a3'; ctx.beginPath();
        if (mode) { for (let t = -1.5; t <= 1.5; t += 0.05) { t === -1.5 ? ctx.moveTo(X(t), Y(f(t))) : ctx.lineTo(X(t), Y(f(t))); } } else { ctx.moveTo(X(mx - 2 * u[0]), Y(my - 2 * u[1])); ctx.lineTo(X(mx + 2 * u[0]), Y(my + 2 * u[1])); }
        ctx.stroke();
        pts.forEach((p) => { let q2;
          if (mode) { let best = 1e9, bt = 0; for (let t = -1.5; t <= 1.5; t += 0.01) { const d = (t - p[0]) ** 2 + (f(t) - p[1]) ** 2; if (d < best) { best = d; bt = t; } } q2 = [bt, f(bt)]; eC += best; }
          else { const t = (p[0] - mx) * u[0] + (p[1] - my) * u[1]; q2 = [mx + t * u[0], my + t * u[1]]; eL += (p[0] - q2[0]) ** 2 + (p[1] - q2[1]) ** 2; }
          ctx.strokeStyle = 'rgba(192,57,43,.5)'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(X(p[0]), Y(p[1])); ctx.lineTo(X(q2[0]), Y(q2[1])); ctx.stroke(); });
        pts.forEach((p) => { ctx.fillStyle = '#2859e0'; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.6, 0, 7); ctx.fill(); });
      });
      $('#la-ea').innerHTML = 'One number per point · average rebuild error <b>' + (eL / N).toFixed(3) + '</b>';
      $('#la-eb').innerHTML = 'One number per point · average rebuild error <b>' + (eC / N).toFixed(3) + '</b>';
    }
    Deck.on('linearae', draw);
  })();
})();
