(function () {
  'use strict';

  /* ---------- Helpers ---------- */
  // Backend base URL comes from js/config.js (falls back to the local Flask API).
  // '' means "same server that served this page".
  const cfg = window.MEDPREDICT_CONFIG || {};
  const API = String(cfg.API_BASE_URL != null ? cfg.API_BASE_URL : 'http://localhost:5000').replace(/\/+$/, '');
  const API_LABEL = API || location.origin;
  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = (s) => { const d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; };
  const pretty = (s) => String(s || '').replace(/_/g, ' ');
  const icon = (id, cls = 'i i-sm') => `<svg class="${cls}" aria-hidden="true"><use href="#${id}"/></svg>`;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  };
  async function postJSON(url, body) {
    const res = await fetch(API + url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }
  function confidenceBadge(pct) {
    if (pct >= 70) return '<span class="badge badge-high">High confidence</span>';
    if (pct >= 40) return '<span class="badge badge-mid">Moderate confidence</span>';
    return '<span class="badge badge-low">Low confidence</span>';
  }

  /* ---------- Routing ---------- */
  const VIEWS = {
    checker:   { title: 'Symptom Checker', sub: 'Select your symptoms and run a machine-learning prediction.' },
    assistant: { title: 'AI Assistant',    sub: 'Describe symptoms in your own words for causes and advice.' },
    vitals:    { title: 'Vitals Check',    sub: 'Screen basic vital signs against common risk patterns.' },
    about:     { title: 'About the Model', sub: 'Dataset, methodology and technical details.' }
  };
  function route(focus) {
    const key = (location.hash || '#checker').slice(1);
    const view = VIEWS[key] ? key : 'checker';
    document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + view));
    document.querySelectorAll('[data-view]').forEach(a => {
      if (a.dataset.view === view) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    $('#page-title').textContent = VIEWS[view].title;
    $('#page-sub').textContent = VIEWS[view].sub;
    document.title = `${VIEWS[view].title} — MedPredict`;
    if (focus) { $('#main').focus({ preventScroll: true }); window.scrollTo(0, 0); }
  }
  window.addEventListener('hashchange', () => route(true));
  route(false);

  /* ---------- Theme ---------- */
  const themeBtn = $('#theme-toggle');
  function syncThemeUI() {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    $('#theme-icon').setAttribute('href', dark ? '#i-sun' : '#i-moon');
    themeBtn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    document.querySelector('meta[name="theme-color"]').setAttribute('content', dark ? '#101D24' : '#0E7490');
  }
  themeBtn.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    store.set('mp-theme', next);
    syncThemeUI();
  });
  syncThemeUI();

  /* ---------- Model status ---------- */
  fetch(API + '/health').then(r => r.json()).then(h => {
    const ok = h.status === 'healthy' && h.ml_model_loaded;
    $('#status-dot').className = 'status-dot ' + (ok ? 'ok' : 'err');
    $('#status-text').textContent = ok ? 'Model online' : 'Model unavailable';
    if (h.diseases_count) {
      $('#status-meta').textContent = `Logistic Regression · ${h.diseases_count} conditions`;
      $('#stat-diseases').textContent = h.diseases_count;
    }
  }).catch(() => {
    $('#status-dot').className = 'status-dot err';
    $('#status-text').textContent = 'Server unreachable';
  });

  /* ---------- Symptom checker ---------- */
  const chipGrid = $('#chip-grid');
  const searchInput = $('#symptom-search');
  const selectedList = $('#selected-list');
  const analyzeBtn = $('#analyze-btn');
  const clearBtn = $('#clear-btn');
  const results = $('#results');
  const selected = new Set();
  let allSymptoms = [];

  function renderChips() {
    const q = searchInput.value.trim().toLowerCase().replace(/_/g, ' ');
    const list = allSymptoms.filter(s => pretty(s).toLowerCase().includes(q));
    if (!list.length) {
      chipGrid.innerHTML = `<div class="empty" style="width:100%">${icon('i-search', 'i')}<p>No symptoms match “${esc(searchInput.value)}”.</p></div>`;
      return;
    }
    chipGrid.innerHTML = list.map(s => {
      const on = selected.has(s);
      return `<button type="button" class="chip" data-s="${esc(s)}" aria-pressed="${on}"><span class="check">${icon('i-check')}</span>${esc(pretty(s))}</button>`;
    }).join('');
  }

  function renderSelected() {
    const n = selected.size;
    $('#selected-count').textContent = n;
    clearBtn.disabled = n === 0;
    analyzeBtn.disabled = n < 2;
    $('#analyze-hint').textContent = n < 2 ? `Select ${2 - n} more symptom${n === 1 ? '' : 's'} to continue.` : 'Ready to analyze.';
    if (!n) {
      selectedList.innerHTML = '<p class="helper" style="text-align:left">No symptoms selected yet.</p>';
      return;
    }
    selectedList.innerHTML = Array.from(selected).map(s =>
      `<span class="sel-chip">${esc(pretty(s))}<button type="button" data-remove="${esc(s)}" aria-label="Remove ${esc(pretty(s))}">${icon('i-x')}</button></span>`
    ).join('');
  }

  function toggleSymptom(s) {
    if (selected.has(s)) selected.delete(s); else selected.add(s);
    const chip = chipGrid.querySelector(`.chip[data-s="${CSS.escape(s)}"]`);
    if (chip) chip.setAttribute('aria-pressed', selected.has(s));
    renderSelected();
  }

  chipGrid.addEventListener('click', e => {
    const chip = e.target.closest('.chip');
    if (chip) toggleSymptom(chip.dataset.s);
  });
  selectedList.addEventListener('click', e => {
    const btn = e.target.closest('[data-remove]');
    if (btn) toggleSymptom(btn.dataset.remove);
  });
  clearBtn.addEventListener('click', () => { selected.clear(); renderChips(); renderSelected(); results.innerHTML = ''; });
  let searchTimer;
  searchInput.addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(renderChips, 80); });

  fetch(API + '/symptoms').then(r => r.json()).then(d => {
    allSymptoms = (d.symptoms || []).slice().sort((a, b) => pretty(a).localeCompare(pretty(b)));
    $('#symptom-total').textContent = `${allSymptoms.length} symptoms`;
    $('#stat-symptoms').textContent = allSymptoms.length;
    renderChips();
  }).catch(() => {
    chipGrid.innerHTML = `<div class="notice notice-danger" style="width:100%">${icon('i-alert', 'i')}<div>Could not reach the backend at <code>${esc(API_LABEL)}</code>. Double-click <code>run.bat</code> in the project folder (or run <code>.venv\\Scripts\\python.exe backend\\app.py</code>), then open <code>http://localhost:5000</code>.</div></div>`;
  });
  renderSelected();

  function insightCard(item, extraClass = '') {
    const name = pretty(item.symptom || item.normalized_symptom);
    const diseases = (item.top_diseases || []).slice(0, 4);
    const flags = (item.red_flags || []).slice(0, 4);
    const advice = (item.advice || []).slice(0, 4);
    return `
      <article class="insight ${extraClass}">
        <div class="insight-head">
          <h4 class="insight-title">${esc(name)}</h4>
          ${item.prevalence_pct != null ? `<span class="badge num" title="Share of dataset cases with this symptom">${esc(item.prevalence_pct)}% of cases</span>` : ''}
        </div>
        ${item.cause ? `<p>${esc(item.cause)}</p>` : ''}
        ${diseases.length ? `<div><div class="sub-label">Commonly linked to</div><div class="tag-row">${diseases.map(d => `<span class="tag num">${esc(d.name)} · ${esc(d.percent)}%</span>`).join('')}</div></div>` : ''}
        ${flags.length ? `<div><div class="sub-label">Seek care promptly if</div><ul class="flag-list">${flags.map(f => `<li>${esc(f)}</li>`).join('')}</ul></div>` : ''}
        ${advice.length ? `<div><div class="sub-label">Self-care</div><ul class="advice-list">${advice.map(a => `<li>${esc(a)}</li>`).join('')}</ul></div>` : ''}
      </article>`;
  }

  function setBtnLoading(btn, loading, label) {
    btn.disabled = loading;
    btn.setAttribute('aria-busy', loading);
    btn.innerHTML = loading ? `<span class="spinner" aria-hidden="true"></span><span>${label}</span>` : label;
  }

  analyzeBtn.addEventListener('click', async () => {
    if (selected.size < 2) return;
    const syms = Array.from(selected);
    const idleLabel = `${icon('i-sparkle')}<span>Analyze symptoms</span>`;
    setBtnLoading(analyzeBtn, true, 'Analyzing…');
    results.innerHTML = `<div class="card card-body" aria-hidden="true"><div class="skeleton" style="height:18px;width:30%;margin-bottom:14px"></div><div class="skeleton" style="height:32px;width:55%;margin-bottom:18px"></div><div class="skeleton" style="height:8px;width:100%"></div></div>`;

    try {
      const [pred, infos] = await Promise.all([
        postJSON('/predict_by_symptoms_ml', { symptoms: syms }),
        Promise.all(syms.slice(0, 6).map(s =>
          fetch(API + '/symptom_info?symptom=' + encodeURIComponent(s)).then(r => r.json()).catch(() => null)))
      ]);

      const probs = Object.entries(pred.probabilities || {}).sort((a, b) => b[1] - a[1]).filter(([n]) => n !== pred.disease).slice(0, 5);
      const conf = Number(pred.confidence) || 0;

      results.innerHTML = `
        <section class="card result-hero" aria-labelledby="res-h">
          <div class="result-main">
            <span class="eyebrow">Most likely condition</span>
            <h3 class="result-name" id="res-h">${esc(pred.disease || 'Unknown')}</h3>
            <div class="conf-line">
              <span class="conf-val num">${conf.toFixed(1)}%</span>
              ${confidenceBadge(conf)}
            </div>
            <div class="meter" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${conf}" aria-label="Model confidence"><span data-w="${Math.min(conf, 100)}"></span></div>
            <div class="result-meta">
              <span class="badge">${icon('i-list')}${syms.length} symptoms analysed</span>
              <span class="badge">Logistic Regression</span>
            </div>
          </div>
          <div class="result-side">
            <span class="eyebrow">Other possibilities</span>
            <div class="diff-list">
              ${probs.map(([n, v]) => `
                <div class="diff-row">
                  <span class="name">${esc(n)}</span><span class="val num">${Number(v).toFixed(1)}%</span>
                  <div class="meter" aria-hidden="true"><span data-w="${Math.min(v, 100)}"></span></div>
                </div>`).join('')}
            </div>
          </div>
        </section>
        ${infos.filter(Boolean).length ? `
        <section aria-labelledby="ins-h">
          <h3 class="section-title" id="ins-h" style="margin-top:0">About your symptoms</h3>
          <div class="insights-grid">${infos.filter(Boolean).map(i => `<div class="card">${insightCard(i)}</div>`).join('')}</div>
        </section>` : ''}`;

      requestAnimationFrame(() => results.querySelectorAll('[data-w]').forEach(el => { el.style.width = el.dataset.w + '%'; }));
      results.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    } catch (err) {
      results.innerHTML = `<div class="notice notice-danger" role="alert">${icon('i-alert', 'i')}<div><strong>Prediction failed.</strong> ${esc(err.message)}. Check that the server is running and try again.</div></div>`;
    } finally {
      setBtnLoading(analyzeBtn, false, idleLabel);
      analyzeBtn.disabled = selected.size < 2;
    }
  });

  /* ---------- Assistant ---------- */
  const chatScroll = $('#chat-scroll');
  const composer = $('#composer');
  const input = $('#composer-input');
  const sendBtn = $('#send-btn');

  function scrollChat() { chatScroll.scrollTop = chatScroll.scrollHeight; }
  function addMsg(role, html) {
    const row = document.createElement('div');
    row.className = 'msg ' + role;
    row.innerHTML = `<div class="avatar ${role}">${icon(role === 'bot' ? 'i-bot' : 'i-user', 'i')}</div><div class="bubble">${html}</div>`;
    chatScroll.appendChild(row);
    scrollChat();
    return row;
  }

  function botReply(data) {
    const insights = data.insights || [];
    let html = '';
    if (insights.length) {
      html += `<p>I recognised <strong>${insights.length}</strong> symptom${insights.length > 1 ? 's' : ''}. Here's what each may indicate:</p>`;
      html += insights.map(i => insightCard(i)).join('');
      const keys = insights.map(i => i.normalized_symptom || i.symptom).filter(k => allSymptoms.includes(k));
      if (keys.length) {
        html += `<div class="bubble-actions"><button class="btn btn-secondary" type="button" data-checker="${esc(keys.join(','))}">Run ML prediction on these ${icon('i-arrow')}</button></div>`;
      }
    } else {
      html += `<p>${esc(data.reply || 'Sorry, I could not process that.')}</p>`;
    }
    if (data.suggestions && data.suggestions.length) {
      html += `<div class="bubble-actions">${data.suggestions.map(s => `<button class="suggest" type="button" data-prompt="${esc(s)}">${esc(s)}</button>`).join('')}</div>`;
    }
    addMsg('bot', html);
  }

  async function send(text) {
    text = text.trim();
    if (!text) return;
    input.value = ''; autoResize();
    sendBtn.disabled = true;
    addMsg('user', `<p>${esc(text)}</p>`);
    const typing = addMsg('bot', '<span class="typing" aria-label="Assistant is typing"><span></span><span></span><span></span></span>');
    try {
      const data = await postJSON('/chat', { message: text });
      typing.remove();
      botReply(data);
    } catch (err) {
      typing.remove();
      addMsg('bot', `<div class="notice notice-danger" role="alert">${icon('i-alert', 'i')}<div>Couldn't reach the backend at <code>${esc(API_LABEL)}</code>. Make sure it is running, then try again.</div></div>`);
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function autoResize() { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 140) + 'px'; }
  input.addEventListener('input', autoResize);
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input.value); } });
  composer.addEventListener('submit', e => { e.preventDefault(); send(input.value); });
  chatScroll.addEventListener('click', e => {
    const p = e.target.closest('[data-prompt]');
    if (p) { send(p.dataset.prompt); return; }
    const c = e.target.closest('[data-checker]');
    if (c) {
      c.dataset.checker.split(',').forEach(s => selected.add(s));
      renderChips(); renderSelected();
      location.hash = '#checker';
    }
  });

  /* ---------- Vitals ---------- */
  const vForm = $('#vitals-form');
  const vResult = $('#vitals-result');
  const painIn = $('#v-pain'), painOut = $('#v-pain-out');
  painIn.addEventListener('input', () => { painOut.textContent = painIn.value; });
  vForm.addEventListener('reset', () => setTimeout(() => {
    painOut.textContent = painIn.value;
    vForm.querySelectorAll('.field').forEach(f => f.classList.remove('invalid'));
  }));

  function validateField(inp) {
    const field = inp.closest('.field');
    const v = parseFloat(inp.value);
    const bad = inp.value === '' || isNaN(v) || v < +inp.min || v > +inp.max;
    field.classList.toggle('invalid', bad);
    inp.setAttribute('aria-invalid', bad);
    return !bad;
  }
  vForm.querySelectorAll('input[type="number"]').forEach(inp => inp.addEventListener('blur', () => validateField(inp)));

  const RANGES = {
    temperature: { label: 'Temperature', unit: '°C', lo: 36.1, hi: 37.2 },
    heart_rate: { label: 'Heart rate', unit: 'bpm', lo: 60, hi: 100 },
    blood_pressure: { label: 'Systolic BP', unit: 'mmHg', lo: 90, hi: 120 },
    oxygen_level: { label: 'SpO₂', unit: '%', lo: 95, hi: 100 }
  };

  vForm.addEventListener('submit', async e => {
    e.preventDefault();
    const inputs = Array.from(vForm.querySelectorAll('input[type="number"]'));
    const invalid = inputs.filter(i => !validateField(i));
    if (invalid.length) { invalid[0].focus(); return; }

    const payload = {};
    inputs.forEach(i => { payload[i.name] = parseFloat(i.value); });
    payload.pain_level = parseInt(painIn.value, 10);

    const btn = $('#vitals-submit');
    const idle = btn.innerHTML;
    setBtnLoading(btn, true, 'Checking…');
    try {
      const d = await postJSON('/predict', payload);
      const conf = Number(d.confidence) || 0;
      const healthy = String(d.disease).toLowerCase() === 'healthy';
      const rows = Object.entries(RANGES).map(([k, r]) => {
        const v = payload[k];
        const ok = v >= r.lo && v <= r.hi;
        const state = ok ? `<span class="state ok">${icon('i-check')}Normal</span>`
                         : `<span class="state warn">${icon('i-alert')}${v < r.lo ? 'Low' : 'High'}</span>`;
        return `<tr><td>${r.label}</td><td class="num">${v} ${r.unit}</td><td>${state}</td></tr>`;
      }).join('');
      vResult.innerHTML = `
        <div class="card">
          <div class="card-body">
            <span class="eyebrow">Screening result</span>
            <h3 class="result-name">${esc(healthy ? 'No risk pattern detected' : d.disease)}</h3>
            <div class="conf-line"><span class="conf-val num">${conf.toFixed(0)}%</span>${confidenceBadge(conf)}</div>
            <div class="meter" aria-hidden="true"><span data-w="${Math.min(conf, 100)}"></span></div>
            <table class="reading-table">
              <thead><tr><th scope="col">Reading</th><th scope="col">Value</th><th scope="col">Status</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
            <p class="hint" style="margin-top:16px;font-size:0.8125rem;color:var(--text-subtle)">Rule-based screen only. Recheck abnormal readings and speak to a clinician if they persist.</p>
          </div>
        </div>`;
      requestAnimationFrame(() => vResult.querySelectorAll('[data-w]').forEach(el => { el.style.width = el.dataset.w + '%'; }));
    } catch (err) {
      vResult.innerHTML = `<div class="notice notice-danger" role="alert">${icon('i-alert', 'i')}<div><strong>Check failed.</strong> ${esc(err.message)}</div></div>`;
    } finally {
      setBtnLoading(btn, false, idle);
    }
  });
})();
