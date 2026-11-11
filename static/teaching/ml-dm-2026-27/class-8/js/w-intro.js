/* Widgets that introduce CNNs and the break
   1. images : what a network receives when it handles an image (grid of numbers per colour channel)   (slide "images")
   3. break  : countdown timer                                                                           (slide "break")
   4. pool   : max / mean pooling and stability to small shifts                                          (slide "pooling")
   */
(function () {
  'use strict';

  /* =================================================================
     1. IMAGES ARE NUMBERS
     ================================================================= */
  (function images() {
    const root = document.getElementById('w-img'); if (!root) return;
    const N = 16, CELL = 20;
    // a tiny procedurally drawn colour picture (sky, sun, hill, house): 16 x 16 x 3
    const pic = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      let col;
      if (r < 10) { const t = r / 10; col = [Math.round(80 + t * 110), Math.round(150 + t * 70), 235]; }
      else { const t = r - 10; col = [Math.round(55 + t * 5), Math.round(150 - t * 9), Math.round(65 + t * 2)]; }
      if (Math.hypot(r - 3, c - 12) < 2.3) col = [255, 226, 60];
      if (r >= 8 && r <= 12 && c >= 3 && c <= 7) col = [196, 68, 58];
      if (r === 7 && c >= 2 && c <= 8) col = [112, 62, 42]; if (r === 6 && c >= 3 && c <= 7) col = [112, 62, 42]; if (r === 5 && c >= 4 && c <= 6) col = [112, 62, 42];
      if (c === 5 && r >= 10 && r <= 12) col = [86, 48, 30];
      if (r === 9 && c === 4) col = [250, 240, 160];
      pic.push(col);
    }
    const dig = U.digits(), DI = 7;                              // a handwritten digit, 8 x 8, one channel
    const digit = Array.from({ length: 64 }, (_, i) => Math.round(dig.X[DI * 64 + i] * 255));

    root.innerHTML = `
      <div class="row" style="margin-bottom:.6rem">
        <div class="seg" id="im-src"><button data-s="dig" class="on">Greyscale: a handwritten digit</button><button data-s="pic">Colour picture</button></div>
        <div class="seg" id="im-ch"><button data-c="rgb" class="on">Colour</button><button data-c="0">R</button><button data-c="1">G</button><button data-c="2">B</button></div>
        <span class="small" id="im-hint">Hover over a pixel.</span>
      </div>
      <div class="row top" style="gap:1.4rem;align-items:flex-start;flex-wrap:wrap">
        <div><canvas id="im-cv" class="cv" style="cursor:crosshair"></canvas><div class="cap" id="im-cap"></div></div>
        <div style="min-width:16rem;flex:1">
          <div class="card flat" id="im-info" style="min-height:11rem"></div>
          <div class="grid g3 mt" style="gap:.6rem">
            <div class="stat"><div class="n" id="im-n1" style="font-size:1.2rem">–</div><div class="l" id="im-l1">this image</div></div>
            <div class="stat hot"><div class="n" style="font-size:1.2rem">150,528</div><div class="l">224 × 224 colour photo<br><span class="tiny">224 × 224 × 3</span></div></div>
            <div class="stat hot"><div class="n" style="font-size:1.2rem">36 million</div><div class="l">12-megapixel phone photo<br><span class="tiny">12,000,000 × 3</span></div></div>
          </div>
        </div>
      </div>`;
    const $ = (s) => root.querySelector(s), cv = $('#im-cv');
    let src = 'dig', ch = 'rgb', hov = null;
    const size = () => (src === 'pic' ? N : 8), cell = () => (src === 'pic' ? CELL : 40);
    const val = (r, c, k) => (src === 'pic' ? pic[r * N + c][k] : digit[r * 8 + c]);

    function draw() {
      const n = size(), cs = cell(), ctx = U.fit(cv, n * cs, n * cs);
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
        let fill;
        if (src === 'dig') { const v = digit[r * 8 + c]; fill = `rgb(${v},${v},${v})`; }
        else if (ch === 'rgb') fill = `rgb(${pic[r * N + c].join(',')})`;
        else { const v = pic[r * N + c][+ch]; fill = ch === '0' ? `rgb(${v},0,0)` : ch === '1' ? `rgb(0,${v},0)` : `rgb(0,0,${v})`; }
        ctx.fillStyle = fill; ctx.fillRect(c * cs, r * cs, cs, cs);
      }
      if (hov) { // highlight the 3x3 neighbourhood a filter would look at
        const { r, c } = hov; ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 2.5;
        ctx.strokeRect(Math.max(0, c - 1) * cs + 1, Math.max(0, r - 1) * cs + 1, (Math.min(n - 1, c + 1) - Math.max(0, c - 1) + 1) * cs - 2, (Math.min(n - 1, r + 1) - Math.max(0, r - 1) + 1) * cs - 2);
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.strokeRect(c * cs + 3, r * cs + 3, cs - 6, cs - 6);
      }
      $('#im-n1').textContent = src === 'pic' ? '16 × 16 × 3 = 768' : '8 × 8 × 1 = 64';
      $('#im-l1').textContent = src === 'pic' ? 'numbers in this picture' : 'numbers in this digit';
      $('#im-cap').textContent = src === 'pic' ? 'A 16×16 colour picture: 3 numbers (R, G, B, each 0–255) per pixel' : 'An 8×8 greyscale digit: 1 number per pixel (0 = black, 255 = white)';
      $('#im-ch').style.display = src === 'pic' ? 'inline-flex' : 'none';
      info();
    }
    function patch(k, r, c) { // 3x3 numbers around the pixel for channel k
      let h = '<table class="t" style="font-size:.78rem;width:auto;display:inline-table"><tbody>';
      for (let dr = -1; dr <= 1; dr++) { h += '<tr>'; for (let dc = -1; dc <= 1; dc++) { const rr = r + dr, cc = c + dc, ok = rr >= 0 && cc >= 0 && rr < size() && cc < size(); h += `<td style="padding:.15rem .4rem;text-align:right;font-family:var(--mono);${dr === 0 && dc === 0 ? 'font-weight:800;background:#fff3b8' : ''}">${ok ? val(rr, cc, k) : '·'}</td>`; } h += '</tr>'; }
      return h + '</tbody></table>';
    }
    function info() {
      const box = $('#im-info');
      if (!hov) { box.innerHTML = '<div class="small" style="color:var(--muted)">Hover over the image: the network never “sees” the picture — it receives these numbers.</div>'; return; }
      const { r, c } = hov;
      if (src === 'dig') { box.innerHTML = `<div class="tiny" style="font-weight:800">PIXEL (row ${r + 1}, column ${c + 1}) = <b style="font-size:1rem">${digit[r * 8 + c]}</b></div><div class="tiny mt" style="font-weight:800">THE 3×3 NEIGHBOURHOOD (what a filter will look at)</div>${patch(0, r, c)}`; return; }
      const p = pic[r * N + c];
      if (ch === 'rgb') box.innerHTML = `<div class="row"><span style="display:inline-block;width:2.2rem;height:2.2rem;border-radius:7px;border:1px solid #0003;background:rgb(${p.join(',')})"></span><div class="tiny" style="font-weight:800">PIXEL (row ${r + 1}, column ${c + 1}) = (R ${p[0]}, G ${p[1]}, B ${p[2]})</div></div>
        <div class="row top mt" style="gap:.8rem"><div><div class="tiny" style="font-weight:800;color:#c0392b">R patch</div>${patch(0, r, c)}</div><div><div class="tiny" style="font-weight:800;color:#1f8a4c">G patch</div>${patch(1, r, c)}</div><div><div class="tiny" style="font-weight:800;color:#2859e0">B patch</div>${patch(2, r, c)}</div></div>`;
      else box.innerHTML = `<div class="tiny" style="font-weight:800">PIXEL (row ${r + 1}, column ${c + 1}), channel ${'RGB'[+ch]} = <b style="font-size:1rem">${p[+ch]}</b></div><div class="tiny mt" style="font-weight:800">THE 3×3 NEIGHBOURHOOD (what a filter will look at)</div>${patch(+ch, r, c)}`;
    }
    cv.addEventListener('pointermove', (e) => { const p = U.pos(cv, e), n = size(), cs = cell(); const r = U.clamp(Math.floor(p.y / cs), 0, n - 1), c = U.clamp(Math.floor(p.x / cs), 0, n - 1); if (!hov || hov.r !== r || hov.c !== c) { hov = { r, c }; draw(); } });
    cv.addEventListener('pointerleave', () => { hov = null; draw(); });
    root.querySelectorAll('#im-src button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#im-src button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); src = b.dataset.s; hov = null; draw(); }));
    root.querySelectorAll('#im-ch button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#im-ch button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); ch = b.dataset.c; draw(); }));
    Deck.on('images', draw);
  })();

  /* =================================================================
     3. BREAK COUNTDOWN
     ================================================================= */
  (function brk() {
    const root = document.getElementById('w-break'); if (!root) return;
    root.innerHTML = `<div id="bk-t" style="font-size:7rem;font-weight:800;letter-spacing:-.03em;font-variant-numeric:tabular-nums;line-height:1">08:00</div>
      <div class="row mt" style="justify-content:center;gap:.5rem"><button class="btn primary" id="bk-go">▶ Start</button><button class="btn" id="bk-reset">Reset</button><span class="spacer" style="flex:0 0 1rem"></span><button class="btn small" id="bk-m">− 1 min</button><button class="btn small" id="bk-p">+ 1 min</button></div>`;
    const $ = (s) => root.querySelector(s); let total = 480, left = 480, timer = null;
    const fmt = (s) => String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
    function draw() { $('#bk-t').textContent = left > 0 ? fmt(left) : 'Time’s up'; $('#bk-t').style.color = left === 0 ? 'var(--gen)' : ''; $('#bk-go').textContent = timer ? '❚❚ Pause' : (left === 0 ? '▶ Restart' : '▶ Start'); }
    function stop() { clearInterval(timer); timer = null; }
    $('#bk-go').addEventListener('click', () => { if (timer) { stop(); } else { if (left === 0) left = total; timer = setInterval(() => { left = Math.max(0, left - 1); if (left === 0) stop(); draw(); }, 1000); } draw(); });
    $('#bk-reset').addEventListener('click', () => { stop(); left = total; draw(); });
    $('#bk-p').addEventListener('click', () => { total += 60; left += 60; draw(); });
    $('#bk-m').addEventListener('click', () => { total = Math.max(60, total - 60); left = Math.max(0, left - 60); draw(); });
    draw();
  })();

  /* =================================================================
     4. POOLING: max / mean, and stability to small shifts
     ================================================================= */
  (function pooling() {
    const root = document.getElementById('w-pool'); if (!root) return;
    const IN = 14, OUT = 12, P = 6, CS = 15;
    const SEVEN = ['..............', '..............', '..##########..', '..##########..', '.........##...', '........##....', '.......##.....', '......##......', '.....##.......', '.....##.......', '....##........', '....##........', '..............', '..............'];
    const base = new Float32Array(IN * IN); SEVEN.forEach((row, r) => [...row].forEach((ch, c) => { base[r * IN + c] = ch === '#' ? 1 : 0; }));
    const K = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    let type = 'max', dx = 1;
    root.innerHTML = `
      <div class="row" style="margin-bottom:.6rem;gap:.8rem">
        <label class="ctl" style="margin:0">Pooling<div class="seg" id="pl-type"><button data-t="max" class="on">Max</button><button data-t="mean">Mean</button></div></label>
        <label class="ctl" style="margin:0">Move the digit by<div class="seg" id="pl-dx"><button data-d="0">0 px</button><button data-d="1" class="on">1 px →</button><button data-d="2">2 px →</button></div></label>
        <span class="small">Window 2×2, stride 2 · filter: vertical edges + ReLU</span>
      </div>
      <div class="row top" style="gap:1.6rem;align-items:flex-start">
        <div><div class="tiny" style="font-weight:800;margin-bottom:.2rem">ORIGINAL IMAGE</div><canvas id="pl-i0" class="cv"></canvas></div>
        <div><div class="tiny" style="font-weight:800;margin-bottom:.2rem">FEATURE MAP</div><canvas id="pl-f0" class="cv"></canvas></div>
        <div><div class="tiny" style="font-weight:800;margin-bottom:.2rem">AFTER POOLING</div><canvas id="pl-p0" class="cv"></canvas></div>
        <div class="stat hot" style="min-width:10rem;align-self:center"><div class="l">feature map still matches the original by</div><div class="n" id="pl-df">–</div><div class="l mt">pooled map still matches the original by</div><div class="n" id="pl-dp" style="color:var(--enc)">–</div></div>
      </div>
      <div class="row top mt" style="gap:1.6rem;align-items:flex-start">
        <div><div class="tiny" style="font-weight:800;margin-bottom:.2rem">SHIFTED IMAGE</div><canvas id="pl-i1" class="cv"></canvas></div>
        <div><div class="tiny" style="font-weight:800;margin-bottom:.2rem">FEATURE MAP</div><canvas id="pl-f1" class="cv"></canvas></div>
        <div><div class="tiny" style="font-weight:800;margin-bottom:.2rem">AFTER POOLING</div><canvas id="pl-p1" class="cv"></canvas></div>
        <div class="tiny" style="align-self:center;max-width:15rem">The map is 12×12 before pooling and 6×6 after: <b>4× fewer numbers</b> to process.</div>
      </div>`;
    const $ = (q) => root.querySelector(q);
    const shift = (img, d) => { const o = new Float32Array(IN * IN); for (let r = 0; r < IN; r++) for (let c = 0; c < IN; c++) { const sc = c - d; o[r * IN + c] = sc >= 0 && sc < IN ? img[r * IN + sc] : 0; } return o; };
    function fmap(img) { const o = new Float32Array(OUT * OUT); for (let r = 0; r < OUT; r++) for (let c = 0; c < OUT; c++) { let s = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += img[(r + i) * IN + c + j] * K[i * 3 + j]; o[r * OUT + c] = Math.max(0, s); } return o; }
    function pool(m) { const o = new Float32Array(P * P); for (let r = 0; r < P; r++) for (let c = 0; c < P; c++) { const v = [m[2 * r * OUT + 2 * c], m[2 * r * OUT + 2 * c + 1], m[(2 * r + 1) * OUT + 2 * c], m[(2 * r + 1) * OUT + 2 * c + 1]]; o[r * P + c] = type === 'max' ? Math.max(...v) : (v[0] + v[1] + v[2] + v[3]) / 4; } return o; }
    const diff = (a, b) => { let ab = 0, aa = 0, bb = 0; for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; } return aa && bb ? ab / Math.sqrt(aa * bb) : 0; };   // cosine similarity: 1 = identical maps
    function paint(cv, data, n, cs, scale, grey) { const ctx = U.fit(cv, n * cs, n * cs); for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { ctx.fillStyle = grey ? U.ink(data[r * n + c]) : U.div(data[r * n + c] / scale); ctx.fillRect(c * cs, r * cs, cs + .4, cs + .4); } ctx.strokeStyle = 'rgba(23,32,44,.08)'; for (let i = 1; i < n; i++) { ctx.beginPath(); ctx.moveTo(i * cs + .5, 0); ctx.lineTo(i * cs + .5, n * cs); ctx.moveTo(0, i * cs + .5); ctx.lineTo(n * cs, i * cs + .5); ctx.stroke(); } }
    function draw() {
      const i0 = base, i1 = shift(base, dx), f0 = fmap(i0), f1 = fmap(i1), p0 = pool(f0), p1 = pool(f1), sc = Math.max(1e-6, ...f0, ...f1);
      paint($('#pl-i0'), i0, IN, CS, 1, true); paint($('#pl-i1'), i1, IN, CS, 1, true);
      paint($('#pl-f0'), f0, OUT, CS, sc); paint($('#pl-f1'), f1, OUT, CS, sc);
      paint($('#pl-p0'), p0, P, CS * 2, sc); paint($('#pl-p1'), p1, P, CS * 2, sc);
      $('#pl-df').textContent = Math.round(diff(f0, f1) * 100) + '%'; $('#pl-dp').textContent = Math.round(diff(p0, p1) * 100) + '%';
    }
    root.querySelectorAll('#pl-type button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#pl-type button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); type = b.dataset.t; draw(); }));
    root.querySelectorAll('#pl-dx button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#pl-dx button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); dx = +b.dataset.d; draw(); }));
    window.__POOL = { draw, test(t, d) { type = t; dx = d; const f0 = fmap(base), f1 = fmap(shift(base, d)); return { fm: diff(f0, f1), pooled: diff(pool(f0), pool(f1)) }; } };
    Deck.on('pooling', draw);
  })();
})();
