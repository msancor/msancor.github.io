/* Discriminative vs generative: the same two classes, two kinds of model, one probe point. */
(function () {
  'use strict';
  const root = document.getElementById('w-gen'); if (!root) return;

  const MA = [-1.4, -0.6], MB = [1.4, 0.6], SG = 0.85, R = 4.5;           // data live in [-R, R]^2
  const rnd = U.mulberry32(5);
  const pts = []; for (let i = 0; i < 70; i++) { pts.push({ x: MA[0] + U.gauss(rnd) * SG, y: MA[1] + U.gauss(rnd) * SG, c: 0 }); pts.push({ x: MB[0] + U.gauss(rnd) * SG, y: MB[1] + U.gauss(rnd) * SG, c: 1 }); }
  let mode = 'disc', probe = { x: 3.1, y: -3.0 }, fakes = [], cell = 400, bg = null;

  const g = (x, y, m) => Math.exp(-((x - m[0]) ** 2 + (y - m[1]) ** 2) / (2 * SG * SG)) / (2 * Math.PI * SG * SG);
  const dens = (x, y) => 0.5 * g(x, y, MA) + 0.5 * g(x, y, MB);
  const DMAX = dens(MA[0], MA[1]);
  const pA = (x, y) => 1 / (1 + Math.exp(-(((x - MB[0]) ** 2 + (y - MB[1]) ** 2) - ((x - MA[0]) ** 2 + (y - MA[1]) ** 2)) / (2 * SG * SG)));

  root.innerHTML = `
    <div class="grid g-side-r" style="align-items:start">
      <div>
        <div class="row" style="margin-bottom:.5rem">
          <div class="seg" id="gn-mode"><button data-m="disc" class="on">Discriminative: learn the boundary</button><button data-m="gen">Generative: learn how the data looks</button></div>
        </div>
        <canvas id="gn-cv" class="cv" style="cursor:crosshair"></canvas>
        <div class="row" style="margin-top:.5rem;gap:.4rem">
          <button class="btn small" id="gn-sample" style="display:none">✨ Generate 10 new points</button>
          <button class="btn small" id="gn-clear" style="display:none">Clear generated</button>
          <button class="btn small" id="gn-far">Probe a far-away point</button>
          <span class="tiny">Click anywhere to place the probe ◎</span>
        </div>
      </div>
      <div>
        <div class="card flat" id="gn-expl"></div>
        <div class="grid g2 mt" style="gap:.7rem">
          <div class="card flat disc"><h3>Discriminative model</h3><div class="tiny">“Which class?” — models <b>p(class | x)</b></div><div class="stat mt" style="padding:.4rem .6rem"><div class="n" id="gn-pa">–</div><div class="l" id="gn-pa-l"></div></div></div>
          <div class="card flat gen"><h3>Generative model</h3><div class="tiny">“How likely is such data?” — models <b>p(x)</b></div><div class="stat mt" style="padding:.4rem .6rem"><div class="n" id="gn-d">–</div><div class="l" id="gn-d-l"></div></div></div>
        </div>
        <div class="verdict mt" id="gn-v"></div>
      </div>
    </div>`;
  const $ = (s) => root.querySelector(s), cv = $('#gn-cv');
  const X = (x) => ((x + R) / (2 * R)) * cell, Y = (y) => cell - ((y + R) / (2 * R)) * cell;
  const inv = (px, py) => ({ x: (px / cell) * 2 * R - R, y: ((cell - py) / cell) * 2 * R - R });
  const expl = {
    disc: '<h3>Discriminative: draw the line</h3><p style="margin:0">Learns only what separates the classes. Great for <b>classification</b> — everything you have done so far (logistic regression, MLPs, CNN classifiers). It cannot tell you what a typical example <i>looks like</i>, and has nothing to say about points far from the data.</p>',
    gen: '<h3>Generative: model the data itself</h3><p style="margin:0">Learns <b>what each class looks like</b> (here: two blobs). Because it models the data distribution, it can <b>sample brand-new examples</b> — and it knows when a point is implausible. Autoencoders (rebuild), GANs and language models (generate) live on this side.</p>',
  };

  function buildBg() {
    const n = 90, c = document.createElement('canvas'); c.width = c.height = n; const cx = c.getContext('2d'), im = cx.createImageData(n, n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = ((i + .5) / n) * 2 * R - R, y = R - ((j + .5) / n) * 2 * R; let col;
      if (mode === 'disc') { const p = pA(x, y); col = p > .5 ? U.mix([255, 255, 255], [40, 89, 224], (p - .5) * 1.7) : U.mix([255, 255, 255], [232, 113, 45], (.5 - p) * 1.7); }
      else { const d = Math.min(1, dens(x, y) / DMAX); col = U.mix([255, 255, 255], [13, 143, 127], Math.pow(d, .6) * .85); }
      const k = (j * n + i) * 4; im.data[k] = col[0]; im.data[k + 1] = col[1]; im.data[k + 2] = col[2]; im.data[k + 3] = 255;
    }
    cx.putImageData(im, 0, 0); bg = c;
  }
  function draw() {
    cell = Math.max(260, Math.min(430, Math.floor(window.innerHeight - 330), Math.floor((root.clientWidth || 800) * 0.55)));
    const ctx = U.fit(cv, cell, cell);
    if (!bg) buildBg();
    ctx.imageSmoothingEnabled = true; ctx.drawImage(bg, 0, 0, cell, cell);
    if (mode === 'disc') { // boundary line (perpendicular bisector)
      ctx.strokeStyle = '#17202c'; ctx.lineWidth = 2.5; ctx.beginPath();
      const mx = (MA[0] + MB[0]) / 2, my = (MA[1] + MB[1]) / 2, dx = MB[0] - MA[0], dy = MB[1] - MA[1];
      ctx.moveTo(X(mx + dy * 8), Y(my - dx * 8)); ctx.lineTo(X(mx - dy * 8), Y(my + dx * 8)); ctx.stroke();
    }
    pts.forEach((p) => { ctx.fillStyle = p.c ? '#e8712d' : '#2859e0'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 4.6, 0, 6.3); ctx.fill(); ctx.stroke(); });
    fakes.forEach((p) => { ctx.fillStyle = '#17202c'; ctx.font = '700 17px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', X(p.x), Y(p.y)); });
    ctx.strokeStyle = '#17202c'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(X(probe.x), Y(probe.y), 9, 0, 6.3); ctx.stroke(); ctx.fillStyle = '#17202c'; ctx.beginPath(); ctx.arc(X(probe.x), Y(probe.y), 2.6, 0, 6.3); ctx.fill();
    ctx.fillStyle = '#667180'; ctx.font = '700 10px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillText('feature 2 ↑', 6, 12); ctx.textAlign = 'right'; ctx.fillText('feature 1 →', cell - 6, cell - 6);
    readout();
  }
  function readout() {
    const p = pA(probe.x, probe.y), conf = Math.max(p, 1 - p), rel = dens(probe.x, probe.y) / DMAX;
    $('#gn-pa').textContent = Math.round(conf * 1000) / 10 + '%'; $('#gn-pa-l').textContent = 'sure it is class ' + (p >= .5 ? 'blue' : 'orange');
    $('#gn-d').textContent = rel > 0.995 ? '100%' : (rel >= .001 ? (rel * 100).toFixed(rel < .1 ? 1 : 0) + '%' : '≈ 0%'); $('#gn-d-l').textContent = 'as typical as the most typical point';
    const v = $('#gn-v');
    if (rel < .03 && conf > .9) { v.className = 'verdict warn'; v.innerHTML = '⚠ The classifier is <b>' + (Math.round(conf * 1000) / 10) + '% sure</b>, but this point is nothing like the training data. A boundary has no notion of “I have never seen this”; a model of <b>p(x)</b> does.'; }
    else if (rel < .03) { v.className = 'verdict warn'; v.innerHTML = 'Nothing like the training data — the generative model flags it.'; }
    else { v.className = 'verdict ok'; v.innerHTML = 'A typical region: both models are comfortable here. Try the far-away probe!'; }
  }
  function sampleFakes() { for (let i = 0; i < 10; i++) { const m = rnd() < .5 ? MA : MB; fakes.push({ x: m[0] + U.gauss(rnd) * SG, y: m[1] + U.gauss(rnd) * SG }); } draw(); }
  function setMode(m) {
    mode = m; bg = null; root.querySelectorAll('#gn-mode button').forEach((b) => b.classList.toggle('on', b.dataset.m === m));
    $('#gn-sample').style.display = $('#gn-clear').style.display = m === 'gen' ? 'inline-block' : 'none'; $('#gn-expl').innerHTML = expl[m]; draw();
  }
  root.querySelectorAll('#gn-mode button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.m)));
  $('#gn-sample').addEventListener('click', sampleFakes);
  $('#gn-clear').addEventListener('click', () => { fakes = []; draw(); });
  $('#gn-far').addEventListener('click', () => { probe = { x: 3.1, y: -3.0 }; draw(); });
  const mv = (e) => { const p = U.pos(cv, e); probe = inv(p.x, p.y); draw(); };
  cv.addEventListener('pointerdown', (e) => { cv.setPointerCapture(e.pointerId); mv(e); });
  cv.addEventListener('pointermove', (e) => { if (e.buttons) mv(e); });
  $('#gn-expl').innerHTML = expl.disc;
  Deck.on('genvsdisc', () => { draw(); });
})();
