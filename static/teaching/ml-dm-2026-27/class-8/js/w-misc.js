/* Smaller interactives: hierarchy ladder, CNN anatomy, GAN stepper. */
(function () {
  'use strict';

  /* =================================================================
     1. HIERARCHY LADDER
     ================================================================= */
  (function ladder() {
    const root = document.getElementById('w-ladder'); if (!root) return;
    const INK = '#17202c', PAPER = '#f0efe8';
    const face = ['..####..', '.#....#.', '#.#..#.#', '#......#', '#.#..#.#', '#..##..#', '.#....#.', '..####..'];
    const svgs = {
      img: [
        () => { let s = ''; face.forEach((row, r) => [...row].forEach((ch, c) => { s += `<rect x="${56 + c * 11}" y="${8 + r * 11}" width="10.5" height="10.5" fill="${ch === '#' ? INK : PAPER}" stroke="#ddd" stroke-width=".4"/>`; })); return s; },
        () => `<g stroke="${INK}" stroke-width="5" stroke-linecap="round" fill="none">
          ${[[18, 14], [72, 14], [126, 14]].map(([x, y]) => `<rect x="${x}" y="${y}" width="44" height="44" rx="6" fill="#fff" stroke="#ccc" stroke-width="1.2"/>`).join('')}
          <line x1="40" y1="20" x2="40" y2="52"/><line x1="78" y1="36" x2="106" y2="36"/><line x1="132" y1="52" x2="160" y2="20"/>
          <rect x="18" y="68" width="44" height="34" rx="6" fill="#fff" stroke="#ccc" stroke-width="1.2"/><path d="M26 98 Q 40 62 56 98" />
          <rect x="72" y="68" width="44" height="34" rx="6" fill="#fff" stroke="#ccc" stroke-width="1.2"/><path d="M80 92 L 94 76 L 108 92" />
          <rect x="126" y="68" width="44" height="34" rx="6" fill="#fff" stroke="#ccc" stroke-width="1.2"/><circle cx="148" cy="85" r="11" /></g>`,
        () => `<g fill="none" stroke="${INK}" stroke-width="3.5" stroke-linecap="round">
          <ellipse cx="46" cy="32" rx="22" ry="12"/><circle cx="46" cy="32" r="6" fill="${INK}"/>
          <path d="M112 12 Q 104 40 98 52 Q 108 58 122 52"/>
          <path d="M44 84 Q 100 112 156 84"/></g>`,
        () => `<g fill="none" stroke="${INK}" stroke-width="4" stroke-linecap="round"><circle cx="100" cy="56" r="48" fill="#fff"/><circle cx="82" cy="44" r="5" fill="${INK}"/><circle cx="118" cy="44" r="5" fill="${INK}"/><path d="M100 50 L 96 66 L 106 66"/><path d="M78 78 Q 100 96 122 78"/></g>`,
      ],
      txt: [
        () => ['g', 'o', 'o', 'd', ' ', 'n', 'o', 't'].map((ch, i) => `<rect x="${10 + i * 22}" y="40" width="20" height="28" rx="4" fill="#fff" stroke="#bbb"/><text x="${20 + i * 22}" y="60" text-anchor="middle" font-size="16" font-family="ui-monospace,monospace" font-weight="700">${ch}</text>`).join(''),
        () => `<rect x="12" y="38" width="72" height="32" rx="8" fill="#e6edfd" stroke="#2859e0"/><text x="48" y="60" text-anchor="middle" font-size="17" font-weight="700">good</text>
               <rect x="108" y="38" width="64" height="32" rx="8" fill="#e6edfd" stroke="#2859e0"/><text x="140" y="60" text-anchor="middle" font-size="17" font-weight="700">not</text>`,
        () => `<rect x="10" y="38" width="160" height="34" rx="9" fill="#dff3ef" stroke="#0d8f7f"/><text x="90" y="61" text-anchor="middle" font-size="17" font-weight="700">not good</text><path d="M44 28 Q 90 8 136 28" fill="none" stroke="#0d8f7f" stroke-width="3"/><text x="90" y="95" text-anchor="middle" font-size="11" fill="#555">negation flips meaning</text>`,
        () => `<rect x="14" y="30" width="152" height="48" rx="24" fill="#fdeadd" stroke="#e8712d" stroke-width="2"/><text x="90" y="60" text-anchor="middle" font-size="16" font-weight="800" fill="#a14a12">negative tone</text>`,
      ],
    };
    const data = {
      img: [
        ['Pixels', 'Raw numbers', 'Brightness values: to the network a face is just a big grid of numbers. Nothing is an “edge” or an “eye” yet.'],
        ['Layer 1', 'Edges & colour blobs', 'Units here respond to tiny, local patterns: short oriented edges and spots of colour, a few pixels wide. First-layer filters look alike across many different image networks.'],
        ['Middle layers', 'Textures & parts', 'Combine edges into corners, curves, textures — then into object parts: eyes, wheels, fur, windows.'],
        ['Last layers', 'Whole objects', 'Units respond to entire concepts (a face, a dog, a bicycle) — the features a final dense layer, the classifier from Class 5, can use directly.'],
      ],
      txt: [
        ['Input', 'Characters / tokens', 'Raw symbols. “not” and “good” are just sequences of letters.'],
        ['Layer 1', 'Words & word parts', 'Detect words, prefixes, suffixes, simple word identity.'],
        ['Middle layers', 'Phrases & syntax', 'Combine words: “not good” is not “good”. Context and negation now matter.'],
        ['Last layers', 'Meaning', 'Topic, stance, tone, intent — the features a classifier can use (e.g. sentiment in social-media data).'],
      ],
    };
    root.innerHTML = `<div class="row" style="margin-bottom:.7rem"><div class="seg" id="ld-mode"><button data-m="img" class="on">Images</button><button data-m="txt">Text</button></div><span class="small">Click a level.</span></div>
      <div class="ladder" id="ld-row"></div>
      <div class="callout mt" id="ld-detail" style="font-size:1.05rem"></div>`;
    let mode = 'img', sel = 0;
    function render() {
      const row = root.querySelector('#ld-row'); row.innerHTML = '';
      data[mode].forEach((d, i) => {
        const el = document.createElement('div'); el.className = 'rung' + (i === sel ? ' on' : '');
        el.innerHTML = `<svg viewBox="0 0 180 112" role="img" aria-label="${d[1]}">${svgs[mode][i]()}</svg><div class="lvl">${d[0]}</div><h4>${d[1]}</h4>`;
        el.addEventListener('click', () => { sel = i; render(); }); row.appendChild(el);
      });
      root.querySelector('#ld-detail').innerHTML = '<b>' + data[mode][sel][1] + '.</b> ' + data[mode][sel][2];
    }
    root.querySelectorAll('#ld-mode button').forEach((b) => b.addEventListener('click', () => { root.querySelectorAll('#ld-mode button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); mode = b.dataset.m; render(); }));
    render();
  })();

  /* =================================================================
     2. CNN ANATOMY
     ================================================================= */
  (function anatomy() {
    const root = document.getElementById('w-anatomy'); if (!root) return;
    const B = [
      { c: 'in',    t: 'Input',            s: '28×28×1',  d: ['28×28×1', '0', 'A greyscale image: 784 numbers, but <b>kept as a grid</b> (not flattened!).'] },
      { c: 'conv',  t: 'Conv + ReLU',      s: '8 filters 3×3', d: ['26×26×8', '8 × (3×3×1 + 1) = 80', '8 filters slide over the image → <b>8 feature maps</b>; ReLU keeps only positive evidence. Detects the simplest patterns (edges, strokes).'] },
      { c: 'pool',  t: 'Max-pool',         s: '2×2',      d: ['13×13×8', '0', 'Halves width and height, keeps the strongest response per block. <b>No parameters</b>, small shift-tolerance, and it makes the next layer’s receptive field grow faster.'] },
      { c: 'conv',  t: 'Conv + ReLU',      s: '16 filters 3×3', d: ['11×11×16', '16 × (3×3×8 + 1) = 1,168', 'Each filter now looks across <b>all 8 incoming maps</b> (that is the “×8”): it combines strokes into corners, curves, loops.'] },
      { c: 'pool',  t: 'Max-pool',         s: '2×2',      d: ['5×5×16', '0', 'Again half the resolution (11 → 5, rounding down). Now each unit “sees” a large part of the digit.'] },
      { c: 'flat',  t: 'Flatten',          s: '→ 400',    d: ['400', '0', 'Unroll the 5×5×16 feature maps into a vector: <b>this is where the grid ends</b> and the MLP you know begins.'] },
      { c: 'out',   t: 'Dense + softmax',  s: '10 classes', d: ['10', '400 × 10 + 10 = 4,010', 'A normal fully-connected output layer with softmax over 10 digits — exactly the multiclass classifier from Class 5, but fed with <b>learned features</b> instead of raw pixels.'] },
    ];
    root.innerHTML = `<div class="pipe" id="an-pipe"></div>
      <div class="brace"><div style="flex:5;border-color:var(--enc);color:var(--enc)">FEATURE EXTRACTOR — learned filters (convolution + pooling)</div><div style="flex:2;border-color:var(--accent);color:var(--accent)">CLASSIFIER — the MLP from Class 5</div></div>
      <div class="grid g3 mt" id="an-info"></div>`;
    const pipe = root.querySelector('#an-pipe'); let sel = 1;
    function render() {
      pipe.innerHTML = '';
      B.forEach((b, i) => {
        const e = document.createElement('div'); e.className = 'blk ' + b.c + (i === sel ? ' sel' : ''); e.innerHTML = `${b.t}<small>${b.s}</small>`;
        e.addEventListener('click', () => { sel = i; render(); }); pipe.appendChild(e);
        if (i < B.length - 1) { const a = document.createElement('div'); a.className = 'arrow'; a.textContent = '›'; pipe.appendChild(a); }
      });
      const d = B[sel].d;
      root.querySelector('#an-info').innerHTML =
        `<div class="stat"><div class="l">output shape</div><div class="n" style="font-size:1.25rem">${d[0]}</div></div>
         <div class="stat"><div class="l">learnable parameters</div><div class="n" style="font-size:1.25rem">${d[1]}</div></div>
         <div class="card flat" style="padding:.6rem .8rem"><b>${B[sel].t}.</b> ${d[2]}</div>`;
    }
    render();
  })();

  /* =================================================================
     3. GAN STEPPER
     ================================================================= */
  (function gan() {
    const svg = document.getElementById('gan-svg'); if (!svg) return;
    const items = Array.from(document.querySelectorAll('#gan-steps li'));
    function set(n) { svg.dataset.step = n; items.forEach((li, i) => li.classList.toggle('on', i + 1 === n)); }
    items.forEach((li, i) => li.addEventListener('click', () => set(i + 1)));
    document.getElementById('gan-next').addEventListener('click', () => set((+svg.dataset.step % 4) + 1));
    document.getElementById('gan-all').addEventListener('click', () => set(0));
    Deck.on('ganconcept', () => set(0));
  })();
})();
