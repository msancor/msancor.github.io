/* Shared helpers for the interactive widgets. */
(function () {
  'use strict';

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // Box-Muller
  function gauss(rnd) { let u = 0, v = 0; while (!u) u = rnd(); while (!v) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  /** Size a canvas for crisp rendering on HiDPI screens. Returns a 2D context in CSS-pixel units. */
  function fit(canvas, w, h) {
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  /** Position of a pointer event inside a canvas in CSS pixels. */
  function pos(canvas, e) {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (canvas.clientWidth / r.width), y: (e.clientY - r.top) * (canvas.clientHeight / r.height) };
  }

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  function mix(c1, c2, t) { return [Math.round(lerp(c1[0], c2[0], t)), Math.round(lerp(c1[1], c2[1], t)), Math.round(lerp(c1[2], c2[2], t))]; }
  const WHITE = [255, 255, 255], POS = [217, 72, 15], NEG = [28, 100, 242], INK = [23, 32, 44];

  /** Diverging colour: negative = blue, zero = white, positive = orange. v is already scaled to [-1, 1]. */
  function div(v) { v = clamp(v, -1, 1); const c = v >= 0 ? mix(WHITE, POS, v) : mix(WHITE, NEG, -v); return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')'; }
  /** Greyscale "ink on paper": 0 = white, 1 = dark. */
  function ink(v) { v = clamp(v, 0, 1); const c = mix(WHITE, INK, v); return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')'; }

  function commas(n) { return Math.round(n).toLocaleString('en-US'); }
  function human(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + ' billion';
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + ' million';
    if (n >= 1e4) return (n / 1e3).toFixed(0) + ' thousand';
    return commas(n);
  }

  /* ---- digits ---- */
  let _digits = null;
  function digits() {
    if (_digits) return _digits;
    const D = window.DIGITS, n = D.n, X = new Float32Array(n * 64), y = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      y[i] = D.labels.charCodeAt(i) - 48;
      for (let j = 0; j < 64; j++) X[i * 64 + j] = (D.pix.charCodeAt(i * 64 + j) - 97) / 16;
    }
    _digits = { n, X, y };
    return _digits;
  }

  /** Draw an 8x8 (or s x s) image given as array-like of 0..1 values. */
  function drawGrid(ctx, data, s, x0, y0, cell, opts) {
    opts = opts || {};
    for (let r = 0; r < s; r++) for (let c = 0; c < s; c++) {
      ctx.fillStyle = (opts.colorFn || ink)(data[r * s + c]);
      ctx.fillRect(x0 + c * cell, y0 + r * cell, cell + (opts.gap ? -opts.gap : 0.5), cell + (opts.gap ? -opts.gap : 0.5));
    }
  }

  window.U = { mulberry32, gauss, fit, pos, clamp, lerp, mix, div, ink, commas, human, digits, drawGrid };
})();
