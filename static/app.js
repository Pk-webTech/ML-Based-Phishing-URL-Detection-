/* PhishLens front end. Vanilla JS, no build step. All server data is inserted with textContent. */
(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  /* ---------- tiny DOM helper ---------- */
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k === 'style') {            // CSSOM calls are allowed by the strict CSP; style attributes are not
        String(v).split(';').filter(Boolean).forEach(decl => {
          const i = decl.indexOf(':');
          if (i > 0) el.style.setProperty(decl.slice(0, i).trim(), decl.slice(i + 1).trim());
        });
      }
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  function staticSvg(markup) {           // trusted constants only
    const t = document.createElement('template');
    t.innerHTML = markup.trim();
    return t.content.firstChild;
  }

  /* ---------- constants ---------- */
  const VERDICT = {
    phishing:   { label: 'Phishing',   risk: 'bad',  title: 'Likely phishing',
                  sub: 'Do not open this link or enter any details on it.' },
    suspicious: { label: 'Suspicious', risk: 'warn', title: 'Suspicious',
                  sub: 'Some warning signs. Check who sent it and the exact site address before you open it.' },
    safe:       { label: 'Low risk',   risk: 'ok',   title: 'Looks low risk',
                  sub: 'No strong warning signs in the address. That is not a guarantee, so stay careful on sign-in pages.' },
  };
  const GLYPH = {
    phishing: '<svg class="verdict-glyph" viewBox="0 0 56 56" aria-hidden="true"><path d="M28 5 53 49H3L28 5Z" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linejoin="round"/><path d="M28 21v14" stroke="currentColor" stroke-width="3.6" stroke-linecap="round"/><circle cx="28" cy="42" r="2.4" fill="currentColor"/></svg>',
    suspicious: '<svg class="verdict-glyph" viewBox="0 0 56 56" aria-hidden="true"><path d="M28 4 52 28 28 52 4 28 28 4Z" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linejoin="round"/><path d="M22.5 23a5.5 5.5 0 1 1 8.2 4.8c-1.8 1-2.7 2.1-2.7 4.2" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round"/><circle cx="28" cy="38.5" r="2.3" fill="currentColor"/></svg>',
    safe: '<svg class="verdict-glyph" viewBox="0 0 56 56" aria-hidden="true"><circle cx="28" cy="28" r="23" fill="none" stroke="currentColor" stroke-width="3.2"/><path d="m17.5 29 7.5 7.5L39 21" fill="none" stroke="currentColor" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  };
  const PART_NAME = {
    scheme: 'Protocol', userinfo: 'Before the @', subdomain: 'Subdomain', domain: 'Main domain',
    port: 'Port', path: 'Path', query: 'Query', fragment: 'Fragment', url: 'Link',
  };
  const MODEL_ORDER = ['random_forest', 'gradient_boosting', 'pytorch_mlp', 'ensemble_mean', 'isolation_forest_anomaly'];
  const HISTORY_KEY = 'phishlens.history';
  const SENS_KEY = 'phishlens.sensitivity';
  const THEME_KEY = 'phishlens.theme';
  const HISTORY_MAX = 200;
  const PAGE_SIZE = 150;

  const state = {
    config: { presets: [{ id: 'balanced', label: 'Balanced', threshold: 0.5, hint: '' }], max_batch: 1000, max_url_length: 2048 },
    sensitivity: 'balanced',
    lastUrl: null,
    bulk: null,
    bulkView: { sort: 'desc', filter: 'all', query: '', shown: PAGE_SIZE },
  };

  /* ---------- helpers ---------- */
  const store = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } },
    remove(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
  };
  function pct(p) {
    const v = p * 100;
    return (v >= 95 || v <= 5) && v !== 0 && v !== 100 ? v.toFixed(1) + '%' : Math.round(v) + '%';
  }
  function threshold() {
    const preset = state.config.presets.find(p => p.id === state.sensitivity) || state.config.presets[0];
    return preset.threshold;
  }
  let toastTimer;
  function toast(message) {
    const el = $('#toast');
    el.textContent = message;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 2400);
  }
  async function copyText(text, okMessage) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = h('textarea', { 'aria-hidden': 'true', class: 'sr-only' });
      ta.value = text;
      document.body.append(ta);
      ta.select();
      try { document.execCommand('copy'); } catch { /* ignore */ }
      ta.remove();
    }
    toast(okMessage);
  }
  function csvCell(value) {
    let s = String(value ?? '');
    if (/^[=+\-@\t\r]/.test(s)) s = "'" + s;      // stop spreadsheet formula injection
    return '"' + s.replace(/"/g, '""') + '"';
  }
  function downloadCsv(filename, header, rows) {
    const body = [header, ...rows].map(r => r.map(csvCell).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob(['﻿' + body], { type: 'text/csv;charset=utf-8' }));
    const a = h('a', { href: url, download: filename });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function humanise(name) { return name.replace(/_/g, ' '); }

  async function api(path, body) {
    let res;
    try {
      res = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
    } catch {
      throw new Error('Cannot reach the PhishLens server. Check that it is still running, then try again.');
    }
    let data = null;
    try { data = await res.json(); } catch { /* not json */ }
    if (!res.ok) {
      let msg = 'Something went wrong. Try again.';
      if (data && typeof data.detail === 'string') msg = data.detail;
      else if (data && Array.isArray(data.detail) && data.detail[0]) msg = String(data.detail[0].msg || msg).replace(/^Value error, /, '');
      throw new Error(msg);
    }
    return data;
  }
  function showError(id, message) {
    const el = $(id);
    el.textContent = message || '';
    el.hidden = !message;
  }

  /* ---------- theme ---------- */
  const SUN = 'M12 4V2m0 20v-2M4 12H2m20 0h-2M5.6 5.6 4.2 4.2m15.6 15.6-1.4-1.4M5.6 18.4l-1.4 1.4M19.8 4.2l-1.4 1.4M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10Z';
  const MOON = 'M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z';
  function effectiveTheme() {
    return document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }
  function paintThemeButton() {
    const dark = effectiveTheme() === 'dark';
    const path = $('#theme-icon');
    path.setAttribute('d', dark ? SUN : MOON);
    if (dark) { path.setAttribute('fill', 'none'); path.setAttribute('stroke', 'currentColor'); path.setAttribute('stroke-width', '2'); path.setAttribute('stroke-linecap', 'round'); }
    else { path.setAttribute('fill', 'currentColor'); path.removeAttribute('stroke'); path.removeAttribute('stroke-width'); }
    $('#theme').setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  }
  function initTheme() {
    const saved = store.get(THEME_KEY, null);
    if (saved === 'dark' || saved === 'light') document.documentElement.dataset.theme = saved;
    paintThemeButton();
    $('#theme').addEventListener('click', () => {
      const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      store.set(THEME_KEY, next);
      paintThemeButton();
    });
  }

  /* ---------- sensitivity control ---------- */
  function buildSensitivity() {
    $$('[data-sensitivity]').forEach((fs, idx) => {
      const group = $('.seg-options', fs);
      group.replaceChildren();
      state.config.presets.forEach(p => {
        const input = h('input', { type: 'radio', name: 'sens-' + idx, value: p.id, checked: p.id === state.sensitivity });
        input.addEventListener('change', () => setSensitivity(p.id));
        group.append(h('label', { title: p.hint + ' (alert at ' + Math.round(p.threshold * 1000) / 10 + '%)' }, input, h('span', { text: p.label })));
      });
    });
  }
  function setSensitivity(id) {
    state.sensitivity = id;
    store.set(SENS_KEY, id);
    $$('[data-sensitivity] input').forEach(i => { i.checked = i.value === id; });
    if (state.lastUrl && location.hash === '#check') runCheck(state.lastUrl, { quiet: true });
    if (state.bulk && location.hash === '#bulk') runBulk(state.bulk.urls);
  }

  /* ---------- Check view ---------- */
  function setBusy(btn, busy, idleText, busyText) {
    btn.disabled = busy;
    btn.replaceChildren(...(busy ? [h('span', { class: 'loading', 'aria-hidden': 'true' }), busyText] : [idleText]));
  }

  async function runCheck(rawUrl, opts = {}) {
    const url = (rawUrl || '').trim();
    showError('#url-error', '');
    if (!url) return showError('#url-error', 'Enter a link to check.');
    if (/\s/.test(url)) return showError('#url-error', 'A link cannot contain spaces. Remove them and try again.');
    if (url.length > state.config.max_url_length) return showError('#url-error', 'That link is too long. The limit is ' + state.config.max_url_length + ' characters.');
    const btn = $('#check-btn');
    if (!opts.quiet) setBusy(btn, true, 'Check link', 'Checking');
    try {
      const result = await api('/api/check', { url, threshold: threshold() });
      state.lastUrl = url;
      renderResult(result, opts);
      saveHistory(result);
    } catch (e) {
      showError('#url-error', e.message);
    } finally {
      setBusy(btn, false, 'Check link', '');
    }
  }

  function renderResult(r, opts = {}) {
    const meta = VERDICT[r.verdict];
    const root = $('#result');
    const animate = !opts.quiet;

    /* verdict + meter */
    const fill = h('div', { class: 'meter-fill' });
    const verdict = h('div', { class: 'verdict', 'data-verdict': r.verdict, role: 'group', 'aria-label': 'Result' },
      staticSvg(GLYPH[r.verdict]),
      h('div', {},
        h('h2', { class: 'verdict-title', text: meta.title }),
        h('p', { class: 'verdict-sub', text: meta.sub })),
      h('div', { class: 'meter' },
        h('div', { class: 'meter-track', role: 'img', 'aria-label': 'Phishing score ' + pct(r.probability) + '. Alert threshold ' + pct(r.threshold) + '.' },
          fill,
          h('div', { class: 'meter-tick', style: 'left:' + (r.threshold * 100) + '%' }),
          h('span', { class: 'meter-tick-label', style: 'left:' + Math.min(Math.max(r.threshold * 100, 12), 88) + '%', text: 'Alert at ' + pct(r.threshold) })),
        h('div', { class: 'meter-scale' }, h('span', { text: '0%' }), h('span', { text: 'Phishing score ' + pct(r.probability) }), h('span', { text: '100%' }))));

    /* anatomy */
    const specimen = h('p', { class: 'url-specimen', 'aria-label': 'The link, split into parts' });
    const rows = h('div', { class: 'parts', role: 'list' });
    const link = (i, on) => {
      $$('[data-idx="' + i + '"]').forEach(n => n.classList.toggle('active', on));
    };
    r.anatomy.forEach((seg, i) => {
      const span = h('span', { class: 'seg' + (animate ? ' draw' : ''), 'data-risk': seg.risk, 'data-idx': i, style: '--i:' + i, text: seg.text });
      span.addEventListener('mouseenter', () => link(i, true));
      span.addEventListener('mouseleave', () => link(i, false));
      specimen.append(span);
      const row = h('div', { class: 'part', role: 'listitem', 'data-idx': i },
        h('div', { class: 'part-name' }, h('span', { class: 'dot', 'data-risk': seg.risk, 'aria-hidden': 'true' }),
          h('span', { text: PART_NAME[seg.part] || seg.part }),
          h('span', { class: 'sr-only', text: seg.risk === 'bad' ? ' (high risk)' : seg.risk === 'warn' ? ' (caution)' : ' (normal)' })),
        h('div', { class: 'part-text', text: seg.text }),
        h('div', { class: 'part-note', 'data-risk': seg.risk, text: seg.note || 'Nothing unusual.' }));
      row.addEventListener('mouseenter', () => link(i, true));
      row.addEventListener('mouseleave', () => link(i, false));
      rows.append(row);
    });
    const anatomy = h('section', { class: 'panel' },
      h('h2', { text: 'Anatomy of the link' }),
      h('p', { class: 'hint', text: 'Each part is underlined by how risky it looks. The main domain is what decides who really owns the site.' }),
      specimen, rows);

    /* why */
    const why = h('section', {},
      h('h2', { text: 'Why this result' }));
    if (r.signals.length) {
      why.append(h('ul', { class: 'signal-list' }, r.signals.map(s =>
        h('li', {},
          h('span', { class: 'dot', 'data-risk': s.severity, 'aria-hidden': 'true' }),
          h('div', {}, h('div', { class: 'signal-title', text: s.title }), h('div', { class: 'signal-detail', text: s.detail }))))));
    } else {
      why.append(h('p', { class: 'empty-note', text: 'No warning signs found in the address itself. That does not guarantee a link is safe, because attackers can also use clean-looking addresses.' }));
    }

    /* votes */
    const votes = h('div', { class: 'votes' });
    const addVote = (label, p, cls) => {
      const f = h('div', { class: 'vote-fill' });
      votes.append(h('div', { class: 'vote ' + (cls || '') },
        h('div', { class: 'vote-head' }, h('span', { text: label }), h('b', { text: pct(p) })),
        h('div', { class: 'vote-track' }, f, h('div', { class: 'vote-tick', style: 'left:' + (r.threshold * 100) + '%' }))));
      return [f, p];
    };
    const fills = [addVote('Combined score', r.probability, 'ensemble'), ...r.models.map(m => addVote(m.label, m.probability))];
    const how = h('section', {}, h('h2', { text: 'How each model voted' }), votes);
    if (r.unusual_vs_legitimate !== null && r.unusual_vs_legitimate !== undefined) {
      how.append(h('p', { class: 'anomaly' }, 'Compared with the legitimate links it was trained on, the anomaly detector finds this link ',
        h('b', { text: r.unusual_vs_legitimate ? 'unusual' : 'ordinary' }), '.'));
    }

    /* actions + measurements */
    const actions = h('div', { class: 'actions' },
      h('button', { type: 'button', class: 'btn', onclick: () => copyText(r.defanged, 'Copied the safe, non-clickable version') }, 'Copy safe version'),
      h('button', { type: 'button', class: 'btn quiet', onclick: () => { $('#url-input').value = ''; $('#url-input').focus(); root.replaceChildren(); state.lastUrl = null; } }, 'Check another link'));
    const grid = h('div', { class: 'measure-grid' }, Object.entries(r.features).map(([k, v]) =>
      h('div', {}, h('span', { text: humanise(k) }), h('span', { text: Number.isInteger(v) ? String(v) : v.toFixed(3) }))));
    const measures = h('details', { class: 'measures' }, h('summary', { text: 'All ' + Object.keys(r.features).length + ' measurements the models used' }), grid);

    root.replaceChildren(verdict, anatomy, h('div', { class: 'two-col panel' }, why, how), actions, measures);

    // color the combined bar like the verdict, then animate bars in
    $('.vote.ensemble', root).style.setProperty('--vc', 'var(--' + (meta.risk === 'bad' ? 'bad' : meta.risk === 'warn' ? 'warn' : 'safe') + ')');
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.width = (r.probability * 100) + '%';
      fills.forEach(([f, p]) => { f.style.width = (p * 100) + '%'; });
    }));
    if (animate) verdict.scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  }

  /* ---------- History ---------- */
  function loadHistory() { const v = store.get(HISTORY_KEY, []); return Array.isArray(v) ? v : []; }
  function saveHistory(r) {
    const list = loadHistory().filter(x => x.url !== r.url);
    list.unshift({ url: r.url, verdict: r.verdict, probability: r.probability, t: Date.now() });
    store.set(HISTORY_KEY, list.slice(0, HISTORY_MAX));
  }
  function inspect(url) {
    location.hash = '#check';
    $('#url-input').value = url;
    runCheck(url);
  }
  function badge(verdict) {
    return h('span', { class: 'badge', 'data-verdict': verdict },
      h('span', { class: 'dot', 'data-risk': VERDICT[verdict].risk, 'aria-hidden': 'true' }), VERDICT[verdict].label);
  }
  function renderHistory() {
    const body = $('#history-body');
    const list = loadHistory();
    if (!list.length) {
      body.replaceChildren(h('div', { class: 'empty' }, h('h2', { text: 'No links checked yet' }),
        h('p', { text: 'Links you check on the Check tab will show up here, so you can look at them again.' }),
        h('p', {}, h('a', { href: '#check', text: 'Check a link' }))));
      return;
    }
    let confirming = false;
    const clearBtn = h('button', { type: 'button', class: 'btn small' }, 'Clear history');
    clearBtn.addEventListener('click', () => {
      if (!confirming) { confirming = true; clearBtn.textContent = 'Click again to clear everything'; clearBtn.classList.add('danger'); setTimeout(() => { confirming = false; clearBtn.textContent = 'Clear history'; clearBtn.classList.remove('danger'); }, 4000); return; }
      store.remove(HISTORY_KEY);
      renderHistory();
      toast('History cleared');
    });
    const exportBtn = h('button', { type: 'button', class: 'btn small', onclick: () => downloadCsv('phishlens-history.csv', ['checked_at', 'url', 'verdict', 'phishing_score'],
      list.map(x => [new Date(x.t).toISOString(), x.url, x.verdict, x.probability.toFixed(4)])) }, 'Export CSV');
    const rows = list.map(x => h('tr', {},
      h('td', { 'data-label': 'Result' }, badge(x.verdict)),
      h('td', { class: 'cell-score', text: pct(x.probability) }),
      h('td', { class: 'cell-url', text: x.url }),
      h('td', { class: 'cell-reasons', text: new Date(x.t).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) }),
      h('td', {}, h('button', { type: 'button', class: 'btn small', onclick: () => inspect(x.url) }, 'Inspect'))));
    body.replaceChildren(
      h('div', { class: 'history-actions' }, exportBtn, clearBtn),
      h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, ...['Result', 'Score', 'Link', 'Checked', 'Details'].map(t => h('th', { scope: 'col', text: t })))),
        h('tbody', {}, rows))));
  }

  /* ---------- Bulk ---------- */
  function parseCsv(text) {
    const rows = []; let row = [], cell = '', quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else quoted = false; } else cell += c; }
      else if (c === '"') quoted = true;
      else if (c === ',') { row.push(cell); cell = ''; }
      else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(cell); rows.push(row); row = []; cell = ''; }
      else cell += c;
    }
    if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
    return rows;
  }
  function extractUrls(text, filename) {
    let urls;
    if (/\.csv$/i.test(filename || '')) {
      const rows = parseCsv(text);
      const header = (rows[0] || []).map(c => c.trim().toLowerCase());
      const idx = header.findIndex(c => ['url', 'urls', 'link', 'links', 'address'].includes(c));
      urls = idx >= 0 ? rows.slice(1).map(r => r[idx]) : rows.map(r => r[0]);
    } else {
      urls = text.split(/\r?\n/);
    }
    return urls.map(u => (u || '').trim()).filter(u => u && !u.startsWith('#'));
  }
  function readBulkInput() {
    const urls = [...new Set($('#bulk-text').value.split(/\r?\n/).map(s => s.trim()).filter(s => s && !s.startsWith('#')))];
    return urls;
  }

  async function runBulk(urls) {
    showError('#bulk-error', '');
    if (!urls.length) return showError('#bulk-error', 'Add at least one link, one per line.');
    if (urls.length > state.config.max_batch) return showError('#bulk-error', 'Found ' + urls.length.toLocaleString() + ' links. Scan up to ' + state.config.max_batch.toLocaleString() + ' at a time.');
    const btn = $('#bulk-btn');
    setBusy(btn, true, 'Scan links', 'Scanning');
    try {
      const data = await api('/api/batch', { urls, threshold: threshold() });
      state.bulk = { urls, data };
      state.bulkView.shown = PAGE_SIZE;
      renderBulk();
    } catch (e) {
      showError('#bulk-error', e.message);
    } finally {
      setBusy(btn, false, 'Scan links', '');
    }
  }

  function renderBulk() {
    const root = $('#bulk-result');
    if (!state.bulk) { root.replaceChildren(); return; }
    const { data } = state.bulk;
    const v = state.bulkView;
    const total = data.total, c = data.counts;
    const share = n => (total ? (n / total) * 100 : 0) + '%';

    const tally = h('div', { class: 'tally', role: 'img', 'aria-label': c.phishing + ' phishing, ' + c.suspicious + ' suspicious, ' + c.safe + ' low risk' },
      h('span', { class: 't-bad', style: 'width:' + share(c.phishing) }),
      h('span', { class: 't-warn', style: 'width:' + share(c.suspicious) }),
      h('span', { class: 't-safe', style: 'width:' + share(c.safe) }));
    const legend = h('div', { class: 'tally-legend' },
      h('span', {}, badge('phishing'), ' ', h('b', { text: c.phishing })),
      h('span', {}, badge('suspicious'), ' ', h('b', { text: c.suspicious })),
      h('span', {}, badge('safe'), ' ', h('b', { text: c.safe })));

    const search = h('input', { type: 'text', 'aria-label': 'Search links', placeholder: 'Search links', value: v.query, autocomplete: 'off' });
    search.addEventListener('input', () => { v.query = search.value; v.shown = PAGE_SIZE; renderBulkTable(); });
    const filter = h('select', { 'aria-label': 'Filter by result' },
      ...[['all', 'All results'], ['phishing', 'Phishing'], ['suspicious', 'Suspicious'], ['safe', 'Low risk']].map(([val, label]) => h('option', { value: val, selected: v.filter === val, text: label })));
    filter.addEventListener('change', () => { v.filter = filter.value; v.shown = PAGE_SIZE; renderBulkTable(); });
    const flagged = data.results.filter(r => r.verdict !== 'safe');
    const exportRows = list => list.map(r => [r.url, r.verdict, r.probability.toFixed(4), r.reasons.join('; ')]);
    const toolbar = h('div', { class: 'toolbar' }, search, filter, h('span', { class: 'spacer' }),
      h('button', { type: 'button', class: 'btn small', disabled: !flagged.length, onclick: () => downloadCsv('phishlens-flagged.csv', ['url', 'verdict', 'phishing_score', 'reasons'], exportRows(flagged)) }, 'Export flagged'),
      h('button', { type: 'button', class: 'btn small', onclick: () => downloadCsv('phishlens-all-results.csv', ['url', 'verdict', 'phishing_score', 'reasons'], exportRows(data.results)) }, 'Export all'));

    const tableHolder = h('div', { id: 'bulk-table' });
    root.replaceChildren(
      h('h2', { text: 'Scanned ' + total.toLocaleString() + (total === 1 ? ' link' : ' links') }), tally, legend, toolbar, tableHolder);
    renderBulkTable();

    function renderBulkTable() {
      let list = data.results.filter(r => (v.filter === 'all' || r.verdict === v.filter) && (!v.query || r.url.toLowerCase().includes(v.query.toLowerCase())));
      list = list.sort((a, b) => v.sort === 'desc' ? b.probability - a.probability : a.probability - b.probability);
      const shown = list.slice(0, v.shown);
      if (!list.length) {
        tableHolder.replaceChildren(h('div', { class: 'table-wrap' }, h('div', { class: 'empty' }, h('h2', { text: 'No links match' }), h('p', { text: 'Change the filter or the search to see more results.' }))));
        return;
      }
      const sortBtn = h('button', { type: 'button', 'aria-label': 'Sort by score, currently ' + (v.sort === 'desc' ? 'highest first' : 'lowest first') }, 'Score ' + (v.sort === 'desc' ? '↓' : '↑'));
      sortBtn.addEventListener('click', () => { v.sort = v.sort === 'desc' ? 'asc' : 'desc'; renderBulkTable(); });
      const body = shown.map(r => h('tr', {},
        h('td', {}, badge(r.verdict)),
        h('td', { class: 'cell-score', text: pct(r.probability) }),
        h('td', { class: 'cell-url', text: r.url }),
        h('td', { class: 'cell-reasons', text: r.reasons.length ? r.reasons.join(', ') : 'No warning signs' }),
        h('td', {}, h('button', { type: 'button', class: 'btn small', onclick: () => inspect(r.url) }, 'Inspect'))));
      const parts = [h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', { scope: 'col', text: 'Result' }), h('th', { scope: 'col', 'aria-sort': v.sort === 'desc' ? 'descending' : 'ascending' }, sortBtn),
          h('th', { scope: 'col', text: 'Link' }), h('th', { scope: 'col', text: 'Why' }), h('th', { scope: 'col', text: 'Details' }))),
        h('tbody', {}, body)))];
      if (list.length > shown.length) {
        parts.push(h('div', { class: 'more' }, h('button', { type: 'button', class: 'btn', onclick: () => { v.shown += PAGE_SIZE; renderBulkTable(); } },
          'Show ' + Math.min(PAGE_SIZE, list.length - shown.length) + ' more (' + (list.length - shown.length).toLocaleString() + ' left)')));
      }
      tableHolder.replaceChildren(...parts);
    }
  }

  /* ---------- Model ---------- */
  let reportCache = null;
  async function renderModel() {
    const body = $('#model-body');
    if (!reportCache) {
      body.replaceChildren(h('p', { class: 'empty-note', text: 'Loading' }));
      try { reportCache = await api('/api/report'); }
      catch (e) { body.replaceChildren(h('p', { class: 'field-error', text: e.message })); return; }
    }
    const rep = reportCache;
    if (!rep.available) {
      body.replaceChildren(h('div', { class: 'empty' }, h('h2', { text: 'No evaluation report yet' }),
        h('p', { text: 'Run python train.py in the project folder, then reload this page.' })));
      return;
    }
    const parts = [];
    if (String(rep.data_source).toUpperCase().startsWith('SYNTHETIC')) {
      parts.push(h('div', { class: 'banner' }, h('b', { text: 'Practice data. ' }),
        'These scores come from a generated dataset, so they show the pipeline works, not how it performs on real links. Add data/urls.csv with real phishing and legitimate links and run python train.py again.'));
    }
    const names = rep.model_labels || {};
    const metricCell = val => h('td', {}, h('div', { class: 'metric-cell' }, h('span', { text: val.toFixed(3) }),
      h('span', { class: 'mini', 'aria-hidden': 'true' }, h('i', { style: 'width:' + (val * 100) + '%' }))));
    const rowsData = MODEL_ORDER.filter(k => rep.test_metrics[k]).map(k => {
      const m = rep.test_metrics[k];
      return h('tr', {}, h('th', { scope: 'row', text: names[k] || k }), metricCell(m.precision), metricCell(m.recall), metricCell(m.f1), metricCell(m.roc_auc));
    });
    parts.push(h('div', { class: 'table-wrap' }, h('table', { class: 'metric-table' },
      h('thead', {}, h('tr', {}, ['Model', 'Precision', 'Recall', 'F1', 'ROC-AUC'].map(t => h('th', { scope: 'col', text: t })))),
      h('tbody', {}, rowsData))));
    const facts = h('div', { class: 'facts' },
      h('p', {}, 'Trained on ', h('b', { text: Number(rep.n_urls).toLocaleString() + ' links' }), ' from ', String(rep.data_source), '.'));
    if (rep.cv_f1) facts.append(h('p', {}, '5-fold cross-validation F1: ', h('b', { text: Object.entries(rep.cv_f1).map(([k, v]) => (names[k] || k) + ' ' + v.toFixed(3)).join(', ') }), '.'));
    const strict = rep.ensemble_threshold_98_precision;
    if (strict) facts.append(h('p', {}, 'The Strict sensitivity alerts at ', h('b', { text: pct(strict) }), ', the lowest score that kept precision at 98% or higher on the test set.'));
    facts.append(h('p', { text: 'Precision is how many alerts were real phishing. Recall is how much of the phishing was caught. F1 balances the two.' }));
    parts.push(facts);

    const captions = {
      'confusion_matrices.png': 'How many test links each model labelled correctly and incorrectly.',
      'precision_recall.png': 'The trade-off between false alarms (precision) and missed phishing (recall) as the alert score changes.',
      'feature_importance.png': 'The 15 measurements the Random Forest relies on most.',
    };
    const alts = {
      'confusion_matrices.png': 'Confusion matrices for the three models',
      'precision_recall.png': 'Precision-recall curves for each model and the ensemble',
      'feature_importance.png': 'Bar chart of the top 15 Random Forest features',
    };
    if (rep.images && rep.images.length) {
      parts.push(h('div', { class: 'figures' }, rep.images.map(n =>
        h('figure', {}, h('img', { src: '/api/report-image/' + encodeURIComponent(n), alt: alts[n] || n, loading: 'lazy' }), h('figcaption', { text: captions[n] || '' })))));
    }
    body.replaceChildren(...parts);
  }

  /* ---------- router ---------- */
  const VIEWS = ['check', 'bulk', 'history', 'model'];
  let firstRoute = true;
  function route() {
    const name = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'check';
    VIEWS.forEach(v => { $('#view-' + v).hidden = v !== name; });
    $$('.tabs a').forEach(a => { if (a.dataset.tab === name) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    document.title = $('#view-' + name).dataset.title + ' | PhishLens';
    if (name === 'history') renderHistory();
    if (name === 'model') renderModel();
    if (name === 'bulk' && state.bulk) renderBulk();
    if (!firstRoute) { const hd = $('#view-' + name + ' h1'); if (hd) hd.focus({ preventScroll: true }); window.scrollTo(0, 0); }
    firstRoute = false;
  }

  /* ---------- wiring ---------- */
  async function init() {
    initTheme();
    try {
      state.config = await api('/api/config');
    } catch (e) {
      toast(e.message);
    }
    const saved = store.get(SENS_KEY, 'balanced');
    state.sensitivity = state.config.presets.some(p => p.id === saved) ? saved : 'balanced';
    $$('[data-max-batch]').forEach(n => { n.textContent = state.config.max_batch.toLocaleString(); });
    buildSensitivity();

    $('#check-form').addEventListener('submit', e => { e.preventDefault(); runCheck($('#url-input').value); });
    $$('[data-example]').forEach(b => b.addEventListener('click', () => { $('#url-input').value = b.dataset.example; runCheck(b.dataset.example); }));

    $('#bulk-form').addEventListener('submit', e => { e.preventDefault(); runBulk(readBulkInput()); });
    $('#bulk-sample').addEventListener('click', () => {
      $('#bulk-text').value = [
        'https://www.google.com/search?q=machine+learning', 'https://github.com/Pk-webTech', 'https://en.wikipedia.org/wiki/Phishing',
        'http://192.168.44.12:8080/login/paypal/a8x2k1.php', 'http://paypal.verify.k3j29dkx1a.xyz/login?id=9ak2l3jd0x',
        'https://secure-microsoft-billing-center.top/x8s7d9a2', 'http://apple.com@zz81jd02k.tk/signin', 'https://bit.ly/3xYz9Qa',
        'https://www.amazon.com/dp/B09XYZ123', 'http://netflix-update-account-help.click/k2j39dl1'].join('\n');
      showError('#bulk-error', '');
    });
    $('#bulk-file').addEventListener('change', async e => {
      const file = e.target.files[0];
      e.target.value = '';
      if (!file) return;
      if (file.size > 2 * 1024 * 1024) return showError('#bulk-error', 'That file is larger than 2 MB. Split it into smaller files.');
      try {
        const urls = extractUrls(await file.text(), file.name);
        if (!urls.length) return showError('#bulk-error', 'No links found in that file. Use one link per line, or a CSV with a url column.');
        $('#bulk-text').value = urls.join('\n');
        showError('#bulk-error', '');
        toast('Loaded ' + urls.length.toLocaleString() + ' links from ' + file.name);
      } catch { showError('#bulk-error', 'That file could not be read. Try a plain .txt or .csv file.'); }
    });

    window.addEventListener('hashchange', route);
    route();
    if (location.hash === '#check' || !location.hash) $('#url-input').focus({ preventScroll: true });
  }
  init();
})();
