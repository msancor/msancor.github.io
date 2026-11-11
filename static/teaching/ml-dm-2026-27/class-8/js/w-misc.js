/* Smaller interactives: CNN anatomy. */
(function () {
  'use strict';

  /* =================================================================
     CNN ANATOMY
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
})();
