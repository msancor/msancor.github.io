/* Minimal slide engine: keyboard / click navigation, step reveals, lazy iframes,
   section pills, overview grid, presenter timer. No dependencies. */
(function () {
  'use strict';

  // Planned minutes per section: 2 + 5 + 6 + 42 = 55 min before the 8-minute break, then 21 + 16 + 3 = 40 min: 103 min in total.
  // The slides marked "optional" in the teaching guide (about 14 min) can be skipped to fit a 90-minute slot.
  const SECTIONS = [
    { id: 'start', label: 'Plan',                    min: 2 },
    { id: 'mlp',   label: 'MLP recap',               min: 5 },
    { id: 'why',   label: 'Why CNNs',                min: 6 },
    { id: 'cnn',   label: 'How CNNs work',           min: 42 },
    { id: 'break', label: 'Break',                   min: 8 },
    { id: 'rep',   label: 'Representation learning', min: 21 },
    { id: 'gan',   label: 'GANs',                    min: 16 },
    { id: 'wrap',  label: 'Wrap-up',                 min: 3 },
  ];

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const slides = $$('.slide');
  const handlers = {};            // slide id -> [fn]
  const revealed = new Map();     // slide -> number of steps shown
  let cur = -1;

  /* ---------- public API for widgets ---------- */
  window.Deck = {
    on(id, fn) { (handlers[id] = handlers[id] || []).push(fn); },
    current: () => slides[cur],
  };

  /* ---------- section pills ---------- */
  const secBox = $('#sections');
  const firstOfSection = {};
  slides.forEach((s, i) => { const k = s.dataset.sec; if (!(k in firstOfSection)) firstOfSection[k] = i; });
  SECTIONS.forEach((sec) => {
    if (sec.hidden || !(sec.id in firstOfSection)) return;
    const b = document.createElement('button');
    b.textContent = sec.label; b.dataset.sec = sec.id;
    b.title = sec.min + ' min';
    b.addEventListener('click', () => go(firstOfSection[sec.id]));
    secBox.appendChild(b);
  });

  /* ---------- navigation ---------- */
  function stepsOf(s) { return $$('.step', s); }
  function applySteps(s) {
    const n = revealed.get(s) || 0;
    stepsOf(s).forEach((el, i) => el.classList.toggle('step-hidden', i >= n));
  }
  function go(i, fromPrev) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    if (i === cur) return;
    const s = slides[i];
    revealed.set(s, fromPrev ? stepsOf(s).length : 0);
    slides.forEach((x) => x.classList.remove('active'));
    s.classList.add('active');
    s.scrollTop = 0;
    cur = i;
    applySteps(s);
    // lazy-load embeds only when their slide is first shown
    $$('iframe[data-src]', s).forEach((f) => {
      if (!f.src) { f.src = f.dataset.src; const l = f.parentElement.querySelector('.loading'); if (l) f.addEventListener('load', () => l.remove()); }
    });
    $('#counter').textContent = (i + 1) + ' / ' + slides.length;
    $('#progress').style.width = ((i + 1) / slides.length * 100) + '%';
    $$('#sections button').forEach((b) => b.classList.toggle('active', b.dataset.sec === s.dataset.sec));
    $('#prev').style.visibility = i === 0 ? 'hidden' : 'visible';
    $('#next').style.visibility = i === slides.length - 1 ? 'hidden' : 'visible';
    const id = s.id || ('s' + (i + 1));
    try { history.replaceState(null, '', '#' + id); } catch (e) { /* file:// etc. */ }
    document.title = (s.dataset.title || 'Class') + ' · Deep Learning: Beyond MLPs';
    (handlers[s.id] || []).forEach((fn) => fn(s));
    updateTimer();
  }
  function next() {
    const s = slides[cur], n = revealed.get(s) || 0, all = stepsOf(s);
    if (n < all.length) { revealed.set(s, n + 1); applySteps(s); return; }
    go(cur + 1, false);
  }
  function prev() {
    const s = slides[cur], n = revealed.get(s) || 0;
    if (n > 0) { revealed.set(s, n - 1); applySteps(s); return; }
    go(cur - 1, true);
  }
  $('#next').addEventListener('click', next);
  $('#prev').addEventListener('click', prev);

  function typing(t) { return t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable); }
  document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (typing(e.target)) return;
    const k = e.key;
    if (k === 'ArrowRight' || k === 'PageDown' || (k === ' ' && !e.shiftKey)) { e.preventDefault(); next(); }
    else if (k === 'ArrowLeft' || k === 'PageUp' || (k === ' ' && e.shiftKey)) { e.preventDefault(); prev(); }
    else if (k === 'Home') go(0);
    else if (k === 'End') go(slides.length - 1);
    else if (k === 'o' || k === 'O' || k === 'Escape') toggleOverlay('overview', k === 'Escape' ? false : undefined);
    else if (k === '?' || k === 'h' || k === 'H') toggleOverlay('help');
    else if (k === 't' || k === 'T') toggleTimer();
    else if (k === 'a' || k === 'A') toggleAll();
    else if (k === 'r' || k === 'R') { if (timerOn) { t0 = Date.now(); updateTimer(); } }
    else if (k === 'f' || k === 'F') { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen && document.documentElement.requestFullscreen(); }
  });
  // keep focus off buttons/sliders after a mouse interaction so Space / arrows keep navigating
  document.addEventListener('pointerup', () => {
    const a = document.activeElement;
    if (a && (a.tagName === 'BUTTON' || (a.tagName === 'INPUT' && a.type === 'range'))) setTimeout(() => a.blur(), 0);
  });

  /* ---------- "show all steps" (self-paced mode) ---------- */
  function toggleAll() { const on = document.body.classList.toggle('showall'); $('#allbtn').classList.toggle('on', on); $('#allbtn').textContent = on ? 'Steps: all shown' : 'Show all steps'; }
  $('#allbtn').addEventListener('click', toggleAll);

  /* ---------- reveal buttons ---------- */
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-reveal]');
    if (!b) return;
    const box = b.dataset.reveal ? document.getElementById(b.dataset.reveal) : b.nextElementSibling;
    if (box) { box.classList.toggle('open'); b.textContent = box.classList.contains('open') ? (b.dataset.hide || 'Hide') : (b.dataset.show || b.textContent); }
  });
  $$('[data-reveal]').forEach((b) => { b.dataset.show = b.textContent; });

  /* ---------- "full width" toggle on hands-on slides with embedded tools ---------- */
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-wide]'); if (!b) return;
    const s = b.closest('.slide'), on = s.classList.toggle('fullw');
    $('.hands-on', s).classList.toggle('wide', on);
    b.textContent = on ? '⤡ Show tasks' : '⤢ Full width';
  });

  /* ---------- overlays ---------- */
  function toggleOverlay(id, force) {
    const el = document.getElementById(id);
    const on = force === undefined ? !el.classList.contains('show') : force;
    if (id === 'help' && on) $('#overview').classList.remove('show');
    el.classList.toggle('show', on);
  }
  const ovGrid = $('#ovgrid');
  slides.forEach((s, i) => {
    const d = document.createElement('div');
    d.className = 'ov-item';
    const sec = SECTIONS.find((x) => x.id === s.dataset.sec);
    d.innerHTML = '<div class="n">' + (i + 1) + ' · ' + (sec ? sec.label : '') + '</div><div class="t">' + (s.dataset.title || '') + '</div>';
    d.addEventListener('click', () => { toggleOverlay('overview', false); go(i); });
    ovGrid.appendChild(d);
  });
  $('#ovbtn').addEventListener('click', () => { $$('.ov-item', ovGrid).forEach((x, j) => x.classList.toggle('cur', j === cur)); toggleOverlay('overview'); });
  $('#helpbtn').addEventListener('click', () => toggleOverlay('help'));
  ['overview', 'help'].forEach((id) => document.getElementById(id).addEventListener('click', (e) => { if (e.target.id === id) toggleOverlay(id, false); }));

  /* ---------- presenter timer ---------- */
  let timerOn = false, t0 = 0, tick = null;
  function fmt(sec) { sec = Math.max(0, Math.round(sec)); return String(Math.floor(sec / 60)).padStart(2, '0') + ':' + String(sec % 60).padStart(2, '0'); }
  function toggleTimer() {
    timerOn = !timerOn;
    $('#timer').classList.toggle('show', timerOn);
    if (timerOn) { if (!t0) t0 = Date.now(); tick = setInterval(updateTimer, 1000); updateTimer(); }
    else clearInterval(tick);
  }
  function updateTimer() {
    if (!timerOn || cur < 0) return;
    const el = (Date.now() - t0) / 1000;
    let start = 0, sec = null;
    for (const s of SECTIONS) { if (s.id === slides[cur].dataset.sec) { sec = s; break; } start += s.min; }
    const total = SECTIONS.reduce((a, s) => a + s.min, 0);
    let msg = '<b>' + fmt(el) + '</b> elapsed &nbsp;/&nbsp; plan ' + total + ' min';
    if (sec && sec.min) {
      const end = (start + sec.min) * 60, over = el - end;
      msg += '<br>' + sec.label + ': planned ' + fmt(start * 60) + '–' + fmt(end);
      msg += over > 0 ? '<br><span class="behind">' + fmt(over) + ' past planned end of this section</span>'
                      : '<br><span class="ahead">within plan</span>';
    }
    msg += '<br><span style="opacity:.6">R = reset · T = hide</span>';
    $('#timer').innerHTML = msg;
  }

  /* ---------- resize notifications for canvas widgets ---------- */
  let rz = null;
  window.addEventListener('resize', () => {
    clearTimeout(rz);
    rz = setTimeout(() => { const s = slides[cur]; (handlers[s.id] || []).forEach((fn) => fn(s, true)); }, 120);
  });

  /* ---------- start ---------- */
  const want = (location.hash || '').replace('#', '');
  let start = slides.findIndex((s) => s.id === want);
  if (start < 0 && /^s?\d+$/.test(want)) start = parseInt(want.replace('s', ''), 10) - 1;
  window.addEventListener('hashchange', () => {
    const h = location.hash.replace('#', ''); const j = slides.findIndex((s) => s.id === h);
    if (j >= 0 && j !== cur) go(j);
  });
  // widgets register handlers during their own script load; start after DOMContentLoaded + one tick
  window.addEventListener('load', () => go(start < 0 ? 0 : start));
})();
