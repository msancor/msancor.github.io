/* Widgets that introduce CNNs and the break
   1. images : what a network receives when it handles an image (grid of numbers per colour channel)   (slide "images")
   2. layers : the layer types of a CNN, with a clickable example network                                (slide "layers")
   3. break  : countdown timer                                                                           (slide "break")  */
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
        <div class="seg" id="im-src"><button data-s="pic" class="on">Colour picture</button><button data-s="dig">Handwritten digit (greyscale)</button></div>
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
    let src = 'pic', ch = 'rgb', hov = null;
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
     2. LAYER TYPES
     ================================================================= */
  (function layers() {
    const root = document.getElementById('w-layers'); if (!root) return;
    const T = {
      conv: { name: 'Convolutional layer', col: 'var(--enc)', soft: 'var(--enc-soft)',
        what: 'Slides a set of small <b>filters</b> over its input; each filter produces one <b>feature map</b>. An activation (ReLU) follows.',
        learn: '<b>Yes</b>: F × (k·k·C + 1) numbers (the filters)', choose: 'number of filters, kernel size, stride, padding',
        icon: '<svg viewBox="0 0 110 70" width="100%" height="100%" preserveAspectRatio="xMidYMid meet"><g stroke="#0d8f7f" fill="#dff3ef"><rect x="6" y="8" width="52" height="52"/></g><g stroke="#9bb" stroke-width=".6">' + [1, 2, 3, 4, 5].map((i) => `<line x1="${6 + i * 8.67}" y1="8" x2="${6 + i * 8.67}" y2="60"/><line x1="6" y1="${8 + i * 8.67}" x2="58" y2="${8 + i * 8.67}"/>`).join('') + '</g><rect x="23" y="25" width="26" height="26" fill="rgba(232,113,45,.35)" stroke="#e8712d" stroke-width="2"/><path d="M64 34 L80 34" stroke="#667180" stroke-width="2"/><path d="M76 29 L82 34 L76 39" fill="none" stroke="#667180" stroke-width="2"/><rect x="86" y="14" width="20" height="20" fill="#fdeadd" stroke="#e8712d"/><rect x="90" y="20" width="20" height="20" fill="#fdeadd" stroke="#e8712d" opacity=".7" transform="translate(-4 8)"/></svg>' },
      pool: { name: 'Pooling layer', col: '#2b7fa6', soft: '#d8ecf5',
        what: 'Summarises each small window (usually its <b>maximum</b>) → smaller maps, a little tolerance to shifts. Works on every map separately.',
        learn: '<b>No</b> parameters', choose: 'window size, stride',
        icon: '<svg viewBox="0 0 110 70" width="100%" height="100%" preserveAspectRatio="xMidYMid meet"><g stroke="#2b7fa6" fill="#d8ecf5"><rect x="6" y="8" width="52" height="52"/></g><g stroke="#2b7fa6" stroke-width="1.6" fill="none"><line x1="32" y1="8" x2="32" y2="60"/><line x1="6" y1="34" x2="58" y2="34"/></g><g font-size="9" font-weight="700" fill="#17202c" text-anchor="middle"><text x="19" y="25">1</text><text x="45" y="25">7</text><text x="19" y="51">3</text><text x="45" y="51">2</text></g><path d="M64 34 L80 34" stroke="#667180" stroke-width="2"/><path d="M76 29 L82 34 L76 39" fill="none" stroke="#667180" stroke-width="2"/><rect x="86" y="22" width="22" height="22" fill="#d8ecf5" stroke="#2b7fa6"/><text x="97" y="38" font-size="10" font-weight="800" text-anchor="middle" fill="#17202c">7</text></svg>' },
      dense: { name: 'Fully connected (dense) layer', col: 'var(--accent)', soft: 'var(--accent-soft)',
        what: 'After <b>flattening</b> the maps into a vector, every unit sees every feature — the MLP layer from Class 5. The last one outputs the class scores (softmax).',
        learn: '<b>Yes</b>: inputs × units + units', choose: 'number of units',
        icon: '<svg viewBox="0 0 110 70" width="100%" height="100%" preserveAspectRatio="xMidYMid meet"><g stroke="#8a93a3" stroke-width=".9">' + [14, 30, 46, 58].map((y) => [18, 34, 50].map((y2) => `<line x1="22" y1="${y}" x2="88" y2="${y2 + 3}"/>`).join('')).join('') + '</g><g fill="#fff" stroke="#2859e0" stroke-width="2">' + [14, 30, 46, 58].map((y) => `<circle cx="22" cy="${y}" r="5"/>`).join('') + [21, 37, 53].map((y) => `<circle cx="88" cy="${y}" r="5"/>`).join('') + '</g></svg>' },
    };
    const NET = [['Input', 'in', ''], ['Conv', 'conv', 'conv'], ['ReLU', 'relu', 'conv'], ['Pool', 'pool', 'pool'], ['Conv', 'conv', 'conv'], ['ReLU', 'relu', 'conv'], ['Pool', 'pool', 'pool'], ['Flatten', 'flat', 'dense'], ['Dense', 'dense', 'dense'], ['Softmax', 'out', 'dense']];
    root.innerHTML = `
      <div class="grid g3" id="ly-cards">${Object.keys(T).map((k) => `
        <div class="card" data-k="${k}" style="border-top:4px solid ${T[k].col};cursor:pointer">
          <div style="height:4.2rem;margin-bottom:.5rem;overflow:hidden">${T[k].icon}</div>
          <h3>${T[k].name}</h3>
          <p class="small" style="color:var(--ink-2);margin:.2rem 0 .5rem">${T[k].what}</p>
          <div class="tiny"><b>Learnable?</b> ${T[k].learn}</div>
          <div class="tiny"><b>You choose:</b> ${T[k].choose}</div>
        </div>`).join('')}</div>
      <div class="tiny mt" style="font-weight:800;letter-spacing:.06em">AN EXAMPLE CNN — click a layer type to see where it appears</div>
      <div class="pipe" id="ly-net" style="padding-top:.3rem"></div>`;
    const net = root.querySelector('#ly-net'); let sel = null;
    function draw() {
      net.innerHTML = '';
      NET.forEach(([label, cls, grp], i) => {
        const e = document.createElement('div'); e.className = 'blk ' + cls; e.textContent = label;
        e.style.opacity = sel && grp !== sel ? .28 : 1; e.style.minWidth = '4rem';
        if (sel && grp === sel) e.style.borderColor = T[sel].col;
        net.appendChild(e);
        if (i < NET.length - 1) { const a = document.createElement('div'); a.className = 'arrow'; a.textContent = '›'; net.appendChild(a); }
      });
      root.querySelectorAll('#ly-cards .card').forEach((c) => { c.style.background = sel === c.dataset.k ? T[c.dataset.k].soft : ''; });
    }
    root.querySelectorAll('#ly-cards .card').forEach((c) => c.addEventListener('click', () => { sel = sel === c.dataset.k ? null : c.dataset.k; draw(); }));
    draw();
  })();

  /* =================================================================
     3. BREAK COUNTDOWN
     ================================================================= */
  (function brk() {
    const root = document.getElementById('w-break'); if (!root) return;
    root.innerHTML = `<div id="bk-t" style="font-size:7rem;font-weight:800;letter-spacing:-.03em;font-variant-numeric:tabular-nums;line-height:1">10:00</div>
      <div class="row mt" style="justify-content:center;gap:.5rem"><button class="btn primary" id="bk-go">▶ Start</button><button class="btn" id="bk-reset">Reset</button><span class="spacer" style="flex:0 0 1rem"></span><button class="btn small" id="bk-m">− 1 min</button><button class="btn small" id="bk-p">+ 1 min</button></div>`;
    const $ = (s) => root.querySelector(s); let total = 600, left = 600, timer = null;
    const fmt = (s) => String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
    function draw() { $('#bk-t').textContent = left > 0 ? fmt(left) : 'Time’s up'; $('#bk-t').style.color = left === 0 ? 'var(--gen)' : ''; $('#bk-go').textContent = timer ? '❚❚ Pause' : (left === 0 ? '▶ Restart' : '▶ Start'); }
    function stop() { clearInterval(timer); timer = null; }
    $('#bk-go').addEventListener('click', () => { if (timer) { stop(); } else { if (left === 0) left = total; timer = setInterval(() => { left = Math.max(0, left - 1); if (left === 0) stop(); draw(); }, 1000); } draw(); });
    $('#bk-reset').addEventListener('click', () => { stop(); left = total; draw(); });
    $('#bk-p').addEventListener('click', () => { total += 60; left += 60; draw(); });
    $('#bk-m').addEventListener('click', () => { total = Math.max(60, total - 60); left = Math.max(0, left - 60); draw(); });
    draw();
  })();
})();
