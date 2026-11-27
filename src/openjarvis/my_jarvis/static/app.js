/* J.A.R.V.I.S. Command Center Frontend Logic (Vanilla JS) */

(() => {
  'use strict';

  const state = {
    view: 'command',
    activeBrain: null,
    brains: [],
    providers: [],
    memories: [],
    coreMode: 'standby', // standby | listening | thinking | speaking
    wakeMode: false,
    recognition: null,
    audioLevel: 0,
    isSending: false,
  };

  // --- DOM Helpers ---
  const $ = (id) => document.getElementById(id);

  function toast(msg) {
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = msg;
    $('toast-container').appendChild(el);
    setTimeout(() => el.remove(), 3400);
  }

  function escapeHtml(str) {
    return String(str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // Lightweight safe Markdown renderer for assistant replies
  function renderMarkdown(raw) {
    let text = escapeHtml(raw);
    // Code blocks
    text = text.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Headings
    text = text.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    // Bold & italic
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // Bullet lists
    const lines = text.split('\n');
    let html = '';
    let inList = false;
    for (const line of lines) {
      if (line.startsWith('- ')) {
        if (!inList) { html += '<ul>'; inList = true; }
        html += `<li>${line.slice(2)}</li>`;
      } else {
        if (inList) { html += '</ul>'; inList = false; }
        if (line.startsWith('<h3>') || line.startsWith('<pre>')) {
          html += line;
        } else if (line.trim() === '') {
          html += '';
        } else {
          html += `<p>${line}</p>`;
        }
      }
    }
    if (inList) html += '</ul>';
    return html;
  }

  async function api(path, options = {}) {
    const resp = await fetch(path, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  // --- Acoustic Arc Core Canvas Visualizer ---
  const canvas = $('arc-canvas');
  const ctx = canvas ? canvas.getContext('2d') : null;
  let animPhase = 0;

  function drawArcCore() {
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const size = 192;
    if (canvas.width !== size * dpr) {
      canvas.width = size * dpr;
      canvas.height = size * dpr;
    }
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);

    const cx = size / 2;
    const cy = size / 2;
    const isUncensored = state.activeBrain && state.activeBrain.tier === 'uncensored';
    const primary = isUncensored ? '#f59e0b' : '#38bdf8';
    const primaryRgb = isUncensored ? '245, 158, 11' : '56, 189, 248';

    const speed =
      state.coreMode === 'thinking' ? 0.055 :
      state.coreMode === 'listening' ? 0.035 :
      state.coreMode === 'speaking' ? 0.045 : 0.012;
    animPhase += speed;

    // Soft ambient core glow
    const pulse =
      state.coreMode === 'speaking' ? 0.24 + Math.sin(animPhase * 4) * 0.08 :
      state.coreMode === 'listening' ? 0.22 + Math.sin(animPhase * 3) * 0.07 :
      state.coreMode === 'thinking' ? 0.20 + Math.sin(animPhase * 5) * 0.06 : 0.11;

    const grad = ctx.createRadialGradient(cx, cy, 8, cx, cy, 78);
    grad.addColorStop(0, `rgba(${primaryRgb}, ${pulse})`);
    grad.addColorStop(0.6, `rgba(${primaryRgb}, ${pulse * 0.25})`);
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, 78, 0, Math.PI * 2);
    ctx.fill();

    // Outer segmented telemetry ring
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(animPhase * 0.55);
    ctx.strokeStyle = `rgba(${primaryRgb}, 0.35)`;
    ctx.lineWidth = 1.5;
    for (let i = 0; i < 6; i++) {
      ctx.beginPath();
      ctx.arc(0, 0, 74, i * (Math.PI / 3) + 0.08, (i + 1) * (Math.PI / 3) - 0.18);
      ctx.stroke();
    }
    ctx.restore();

    // Counter-rotating inner precision ring
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-animPhase * 0.85);
    ctx.strokeStyle = `rgba(${primaryRgb}, 0.65)`;
    ctx.lineWidth = 2;
    ctx.setLineDash([12, 8]);
    ctx.beginPath();
    ctx.arc(0, 0, 56, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    // Radial acoustic bars
    const bars = 36;
    for (let i = 0; i < bars; i++) {
      const angle = (i / bars) * Math.PI * 2;
      const activeAmp =
        state.coreMode === 'listening' ? 12 + Math.abs(Math.sin(animPhase * 3 + i * 0.6)) * 14 :
        state.coreMode === 'speaking' ? 10 + Math.abs(Math.cos(animPhase * 4.5 + i * 0.8)) * 13 :
        state.coreMode === 'thinking' ? 7 + Math.abs(Math.sin(animPhase * 5 + i * 0.4)) * 9 :
        4 + Math.abs(Math.sin(animPhase + i * 0.5)) * 3;
      const r0 = 33;
      const r1 = r0 + activeAmp;
      ctx.strokeStyle = `rgba(${primaryRgb}, ${state.coreMode === 'standby' ? 0.32 : 0.78})`;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(angle) * r0, cy + Math.sin(angle) * r0);
      ctx.lineTo(cx + Math.cos(angle) * r1, cy + Math.sin(angle) * r1);
      ctx.stroke();
    }

    // Center core disc
    ctx.fillStyle = primary;
    ctx.beginPath();
    ctx.arc(cx, cy, 13 + (state.coreMode === 'standby' ? 0 : Math.sin(animPhase * 4) * 1.6), 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#080b10';
    ctx.beginPath();
    ctx.arc(cx, cy, 6.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
    requestAnimationFrame(drawArcCore);
  }

  function setCoreMode(mode) {
    state.coreMode = mode;
    const labels = {
      standby: 'STANDBY',
      listening: 'LISTENING',
      thinking: 'PROCESSING',
      speaking: 'SPEAKING',
    };
    const el = $('core-state-label');
    if (el) el.textContent = labels[mode] || 'STANDBY';
  }

  // --- Navigation ---
  const VIEW_TITLES = {
    command: ['Command Center', 'Voice, Tools & Hot-Swappable Multi-Brain Core'],
    brains: ['Brains & Provider Keys', 'Local Ollama · Venice AI · Anthropic · DeepSeek · OpenAI · Gemini'],
    memory: ['Persistent Memory Vault', 'Shared across all brains via OpenJarvis LocalFactStore'],
    hardware: ['Hardware & Hugging Face', 'Live VRAM/RAM fit scoring & GGUF explorer'],
    audit: ['Defensive Self-Audit', 'Authorized passive header/TLS check & local repo scan'],
    doctor: ['Setup & Doctor', 'OpenJarvis health checks & zero-friction setup'],
  };

  function switchView(view) {
    state.view = view;
    document.querySelectorAll('.nav-item').forEach((btn) => {
      const isMatch = btn.dataset.view === view;
      btn.classList.toggle('active', isMatch);
      btn.setAttribute('aria-selected', String(isMatch));
    });
    document.querySelectorAll('.view-panel').forEach((panel) => {
      panel.classList.toggle('active', panel.id === `view-${view}`);
    });
    const [title, meta] = VIEW_TITLES[view] || ['Command Center', ''];
    $('topbar-title').textContent = title;
    $('topbar-meta').textContent = meta;

    if (view === 'brains') loadBrains();
    if (view === 'memory') loadMemories();
    if (view === 'hardware') { loadHardware(); searchHF(); }
    if (view === 'doctor') loadDoctor();
  }

  // --- Active Brain Rendering ---
  function updateActiveBrainUI(brain, ready = true) {
    if (!brain) return;
    state.activeBrain = brain;
    $('rail-brain-name').textContent = brain.name;
    $('rail-brain-model').textContent = `${brain.provider} · ${brain.model}`;
    const dot = $('rail-brain-dot');
    dot.className = 'status-dot' + (ready ? '' : ' warn');
    const tag = $('core-brain-tag');
    if (tag) tag.textContent = `${brain.id} (${brain.model})`;
  }

  function renderQuickBrains() {
    const container = $('quick-brains-list');
    if (!container) return;
    const featuredIds = [
      'local-qwen',
      'venice-deepseek',
      'claude-opus',
      'claude-sonnet',
      'deepseek-cloud',
      'gpt-5',
      'gemini-flash',
      'builtin-offline',
    ];
    const items = state.brains.filter((b) => featuredIds.includes(b.id) || b.active);
    container.innerHTML = items.map((b) => `
      <button class="quick-brain-item ${b.active ? 'active' : ''}" data-switch-brain="${escapeHtml(b.id)}">
        <div class="qbi-left">
          <span class="qbi-name">${escapeHtml(b.name)}</span>
          <span class="qbi-sub">${escapeHtml(b.model)}</span>
        </div>
        <span class="tier-tag ${escapeHtml(b.tier)}">${escapeHtml(b.ready ? b.tier : 'setup')}</span>
      </button>
    `).join('');

    container.querySelectorAll('[data-switch-brain]').forEach((btn) => {
      btn.addEventListener('click', () => handleSwitchBrain(btn.dataset.switchBrain));
    });
  }

  async function handleSwitchBrain(targetId) {
    try {
      const res = await api('/api/jarvis/brains/switch', {
        method: 'POST',
        body: JSON.stringify({ target: targetId }),
      });
      updateActiveBrainUI(res.active_brain, res.ready);
      await loadBrains();
      toast(`Switched brain to ${res.active_brain.name}`);
    } catch (err) {
      toast(`Switch failed: ${err.message}`);
    }
  }

  // --- Chat & History ---
  function renderMessage(turn) {
    $('empty-state').classList.add('hidden');
    const wrap = document.createElement('div');
    wrap.className = `msg ${turn.role}`;

    const isUncensored = turn.provider === 'venice';
    const metaParts = [];
    if (turn.role === 'assistant') {
      metaParts.push(`<span class="msg-brain-badge ${isUncensored ? 'uncensored' : ''}">${escapeHtml(turn.brain_id || turn.model)}</span>`);
      if (turn.latency_ms) metaParts.push(`<span>${turn.latency_ms} ms</span>`);
      if (turn.estimated_cost_usd > 0) metaParts.push(`<span>$${turn.estimated_cost_usd.toFixed(5)}</span>`);
      else metaParts.push(`<span>$0.00</span>`);
    } else {
      metaParts.push('<span>YOU</span>');
    }

    let toolsHtml = '';
    if (turn.tools_used && turn.tools_used.length > 0) {
      toolsHtml = `<div class="tool-trace-row">${turn.tools_used.map((t) =>
        `<span class="tool-pill">⚙ ${escapeHtml(t.tool || 'tool')}</span>`
      ).join('')}</div>`;
    }

    wrap.innerHTML = `
      <div class="msg-meta">${metaParts.join('')}</div>
      ${toolsHtml}
      <div class="msg-bubble">${turn.role === 'assistant' ? renderMarkdown(turn.content) : escapeHtml(turn.content)}</div>
    `;

    const box = $('chat-messages');
    box.appendChild(wrap);
    box.scrollTop = box.scrollHeight;
  }

  async function sendChat(text) {
    const clean = (text || '').trim();
    if (!clean || state.isSending) return;
    state.isSending = true;
    $('btn-send').disabled = true;
    setCoreMode('thinking');

    renderMessage({ role: 'user', content: clean });
    $('chat-input').value = '';
    $('chat-input').style.height = 'auto';

    try {
      const data = await api('/api/jarvis/chat', {
        method: 'POST',
        body: JSON.stringify({
          message: clean,
          use_tools: $('toggle-tools').checked,
        }),
      });
      const rep = data.reply;
      renderMessage(rep);
      if (data.active_brain) {
        updateActiveBrainUI(data.active_brain, true);
      }
      if (data.brain_switched) {
        loadBrains();
        toast(`Brain hot-swapped → ${data.active_brain.name}`);
      }
      $('stat-latency').textContent = `${rep.latency_ms || 0} ms`;
      $('stat-cost').textContent = `$${(rep.estimated_cost_usd || 0).toFixed(4)}`;

      // Update memory badge in case user saved a fact
      loadStatusQuiet();

      if ($('toggle-tts').checked && rep.content) {
        speakText(rep.content);
      } else {
        setCoreMode('standby');
      }
    } catch (err) {
      toast(`Error: ${err.message}`);
      setCoreMode('standby');
    } finally {
      state.isSending = false;
      $('btn-send').disabled = false;
    }
  }

  // --- Voice STT & TTS (Browser Web Speech API) ---
  function initTTSVoices() {
    if (!('speechSynthesis' in window)) return;
    const populate = () => {
      const select = $('tts-voice-select');
      const voices = window.speechSynthesis.getVoices();
      if (!voices.length) return;
      const current = select.value;
      select.innerHTML = '<option value="">System Default Voice</option>' +
        voices.map((v) => `<option value="${escapeHtml(v.name)}">${escapeHtml(v.name)} (${escapeHtml(v.lang)})</option>`).join('');
      if (current) {
        select.value = current;
      } else {
        // Prefer a crisp British or English voice for JARVIS feel
        const preferred = voices.find((v) => /daniel|arthur|en-GB|george|google uk english male/i.test(v.name + ' ' + v.lang));
        if (preferred) select.value = preferred.name;
      }
    };
    populate();
    window.speechSynthesis.onvoiceschanged = populate;
  }

  function speakText(markdownText) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    // Strip markdown formatting for natural speech
    const plain = markdownText
      .replace(/```[\s\S]*?```/g, ' Code block omitted. ')
      .replace(/[*#`_~]/g, '')
      .replace(/https?:\/\/\S+/g, 'link')
      .slice(0, 700);
    const utt = new SpeechSynthesisUtterance(plain);
    const chosen = $('tts-voice-select').value;
    if (chosen) {
      const voice = window.speechSynthesis.getVoices().find((v) => v.name === chosen);
      if (voice) utt.voice = voice;
    }
    utt.rate = 1.04;
    utt.onstart = () => {
      setCoreMode('speaking');
      $('btn-stop-tts').classList.remove('hidden');
    };
    utt.onend = utt.onerror = () => {
      setCoreMode('standby');
      $('btn-stop-tts').classList.add('hidden');
    };
    window.speechSynthesis.speak(utt);
  }

  function startSpeechRecognition(continuousWake = false) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      toast('Browser Speech Recognition is not supported here — use Chrome/Edge or type directly.');
      return;
    }
    stopSpeechRecognition();
    const rec = new SR();
    rec.lang = 'en-US';
    rec.interimResults = true;
    rec.continuous = continuousWake;
    state.recognition = rec;

    rec.onstart = () => {
      setCoreMode('listening');
      $('btn-mic').classList.add('recording');
      $('voice-status-bar').classList.remove('hidden');
      $('voice-status-text').textContent = continuousWake
        ? 'Hands-free active — say "Jarvis, ..."'
        : 'Listening — speak your command...';
    };

    rec.onresult = (event) => {
      let interim = '';
      let finalStr = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalStr += transcript;
        else interim += transcript;
      }
      if (interim) $('voice-status-text').textContent = `Hearing: "${interim}"`;
      if (finalStr.trim()) {
        const spoken = finalStr.trim();
        if (continuousWake) {
          const m = spoken.match(/(?:hey\s+)?jarvis[, ]*(.*)/i);
          if (m && m[1].trim()) {
            sendChat(m[1].trim());
          }
        } else {
          stopSpeechRecognition();
          sendChat(spoken);
        }
      }
    };

    rec.onerror = (e) => {
      if (e.error !== 'no-speech' && e.error !== 'aborted') {
        toast(`Voice input: ${e.error}`);
      }
    };

    rec.onend = () => {
      if (state.wakeMode && continuousWake) {
        try { rec.start(); } catch (_) {}
      } else {
        stopSpeechRecognition();
      }
    };

    rec.start();
  }

  function stopSpeechRecognition() {
    if (state.recognition) {
      const r = state.recognition;
      state.recognition = null;
      try { r.stop(); } catch (_) {}
    }
    $('btn-mic').classList.remove('recording');
    $('voice-status-bar').classList.add('hidden');
    if (state.coreMode === 'listening') setCoreMode('standby');
  }

  // --- Brains & Providers View ---
  async function loadBrains() {
    try {
      const data = await api('/api/jarvis/brains');
      state.brains = data.brains || [];
      state.providers = data.providers || [];
      const active = state.brains.find((b) => b.active);
      if (active) updateActiveBrainUI(active, active.ready);

      renderQuickBrains();
      renderBrainCards();
      renderProviders();
    } catch (err) {
      toast(`Failed to load brains: ${err.message}`);
    }
  }

  function renderBrainCards() {
    const grid = $('brain-cards-grid');
    if (!grid) return;
    grid.innerHTML = state.brains.map((b) => `
      <div class="brain-card ${b.active ? 'active' : ''}" data-brain-id="${escapeHtml(b.id)}">
        <div class="brain-card-head">
          <span class="brain-card-title">${escapeHtml(b.name)}</span>
          <span class="tier-tag ${escapeHtml(b.tier)}">${escapeHtml(b.tier)}</span>
        </div>
        <div class="brain-card-model">${escapeHtml(b.provider)} · ${escapeHtml(b.model)}</div>
        <p class="brain-card-desc">${escapeHtml(b.description)}</p>
        <div class="brain-card-foot">
          <span>${escapeHtml(b.status_note)}</span>
          <span>${b.active ? '● ACTIVE' : 'Click to Activate'}</span>
        </div>
      </div>
    `).join('');

    grid.querySelectorAll('.brain-card').forEach((card) => {
      card.addEventListener('click', () => handleSwitchBrain(card.dataset.brainId));
    });
  }

  function renderProviders() {
    const list = $('provider-list');
    if (!list) return;
    list.innerHTML = state.providers.map((p) => {
      const isOllama = p.provider === 'ollama';
      return `
        <div class="provider-row">
          <div class="provider-row-head">
            <span class="provider-name">${escapeHtml(p.label)}</span>
            <span class="provider-status ${p.configured ? 'ok' : 'missing'}">${escapeHtml(p.status)}</span>
          </div>
          <form class="provider-form" data-provider="${escapeHtml(p.provider)}">
            <input
              type="${isOllama ? 'text' : 'password'}"
              name="secret"
              value="${isOllama ? escapeHtml(p.host || 'http://localhost:11434') : ''}"
              placeholder="${isOllama ? 'http://localhost:11434' : (p.masked ? `Saved (${p.masked}) — paste new key to update` : `Paste ${p.env_var}`)}"
            />
            <button type="submit" class="btn-secondary">Save</button>
          </form>
        </div>
      `;
    }).join('');

    list.querySelectorAll('.provider-form').forEach((form) => {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const provider = form.dataset.provider;
        const val = form.querySelector('input[name="secret"]').value.trim();
        try {
          const payload = provider === 'ollama'
            ? { provider, host: val }
            : { provider, api_key: val };
          await api('/api/jarvis/providers', {
            method: 'POST',
            body: JSON.stringify(payload),
          });
          toast(`Saved ${provider} configuration`);
          await loadBrains();
        } catch (err) {
          toast(`Save failed: ${err.message}`);
        }
      });
    });
  }

  // --- Memory Vault View ---
  async function loadMemories() {
    try {
      const data = await api('/api/jarvis/memory');
      state.memories = data.facts || [];
      $('nav-memory-count').textContent = String(state.memories.length);
      const list = $('memory-list');
      if (!state.memories.length) {
        list.innerHTML = '<div class="memory-card"><span class="memory-meta">No facts stored yet. Add one above or say "Remember that..." in Command Center.</span></div>';
        return;
      }
      list.innerHTML = state.memories.map((m) => `
        <div class="memory-card">
          <div>
            <div class="memory-text">${escapeHtml(m.text)}</div>
            <div class="memory-meta">Source: ${escapeHtml(m.source)} · Trust: ${escapeHtml(m.trust)}</div>
          </div>
          <button class="btn-ghost-sm" data-del-mem="${m.index}">Delete</button>
        </div>
      `).join('');

      list.querySelectorAll('[data-del-mem]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          await api(`/api/jarvis/memory/${btn.dataset.delMem}`, { method: 'DELETE' });
          toast('Memory entry removed');
          loadMemories();
        });
      });
    } catch (err) {
      toast(`Failed to load memory: ${err.message}`);
    }
  }

  // --- Hardware & Hugging Face View ---
  async function loadHardware() {
    try {
      const data = await api('/api/jarvis/hardware');
      const hw = data.hardware;
      const gpuLabel = hw.gpus && hw.gpus.length
        ? hw.gpus.map((g) => `${g.name} (${g.vram_gb} GB)`).join(', ')
        : 'CPU Only (No GPU detected)';

      $('hw-metrics-grid').innerHTML = `
        <div class="hw-metric-card">
          <div class="hw-label">MEMORY (RAM)</div>
          <div class="hw-value">${hw.ram_total_gb} GB</div>
          <div class="hw-sub">${hw.ram_available_gb} GB available</div>
        </div>
        <div class="hw-metric-card">
          <div class="hw-label">GPU / VRAM</div>
          <div class="hw-value">${hw.total_vram_gb > 0 ? hw.total_vram_gb + ' GB' : 'CPU'}</div>
          <div class="hw-sub">${escapeHtml(gpuLabel)}</div>
        </div>
        <div class="hw-metric-card">
          <div class="hw-label">PROCESSOR</div>
          <div class="hw-value">${hw.cpu_cores_logical} Threads</div>
          <div class="hw-sub">${hw.cpu_cores_physical} physical · ${escapeHtml(hw.arch)}</div>
        </div>
        <div class="hw-metric-card">
          <div class="hw-label">OLLAMA RUNTIME</div>
          <div class="hw-value">${hw.ollama_running ? 'Online' : 'Offline'}</div>
          <div class="hw-sub">${hw.ollama_models.length} local models pulled</div>
        </div>
      `;

      $('model-fit-list').innerHTML = (data.recommendations || []).map((m) => `
        <div class="fit-row">
          <div>
            <div><strong>${escapeHtml(m.name)}</strong> <span class="hw-sub">(${escapeHtml(m.params)} · ${escapeHtml(m.quant)})</span></div>
            <div class="memory-meta">${escapeHtml(m.fit_reason)}</div>
            <div class="memory-meta">${escapeHtml(m.notes)}</div>
          </div>
          <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
            <span class="fit-badge ${escapeHtml(m.fit)}">${escapeHtml(m.fit.replace('_', ' '))}</span>
            <button class="btn-ghost-sm" data-copy-cmd="${escapeHtml(m.ollama_cmd)}">Copy Pull</button>
          </div>
        </div>
      `).join('');

      $('model-fit-list').querySelectorAll('[data-copy-cmd]').forEach((btn) => {
        btn.addEventListener('click', () => {
          navigator.clipboard?.writeText(btn.dataset.copyCmd);
          toast(`Copied: ${btn.dataset.copyCmd}`);
        });
      });
    } catch (err) {
      toast(`Hardware probe failed: ${err.message}`);
    }
  }

  async function searchHF() {
    const q = ($('hf-query')?.value || 'abliterated').trim();
    const list = $('hf-results-list');
    list.innerHTML = '<div class="hf-row"><span class="memory-meta">Searching Hugging Face Hub...</span></div>';
    try {
      const data = await api(`/api/jarvis/hf-models?q=${encodeURIComponent(q)}&limit=8`);
      list.innerHTML = (data.models || []).map((m) => `
        <div class="hf-row">
          <div style="min-width:0;">
            <div><strong>${escapeHtml(m.repo_id)}</strong></div>
            <div class="memory-meta">${escapeHtml(m.notes || `Source: ${data.source} · Downloads: ${m.downloads || 0}`)}</div>
            <div style="margin-top:6px;"><span class="cmd-box">${escapeHtml(m.ollama_run_cmd)}</span></div>
          </div>
          <div style="display:flex;gap:6px;flex-shrink:0;">
            <button class="btn-ghost-sm" data-copy-hf="${escapeHtml(m.ollama_run_cmd)}">Copy</button>
            <button class="btn-secondary" data-reg-hf="${escapeHtml(m.repo_id)}">Add Brain</button>
          </div>
        </div>
      `).join('');

      list.querySelectorAll('[data-copy-hf]').forEach((btn) => {
        btn.addEventListener('click', () => {
          navigator.clipboard?.writeText(btn.dataset.copyHf);
          toast(`Copied: ${btn.dataset.copyHf}`);
        });
      });
      list.querySelectorAll('[data-reg-hf]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          const repo = btn.dataset.regHf;
          const shortId = 'hf-' + repo.split('/').pop().toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 28);
          await api('/api/jarvis/brains/custom', {
            method: 'POST',
            body: JSON.stringify({
              brain_id: shortId,
              name: repo.split('/').pop(),
              provider: 'ollama',
              model: `hf.co/${repo}`,
              description: `Hugging Face GGUF model (${repo}) via Ollama`,
            }),
          });
          toast(`Registered ${shortId} in Brains`);
          loadBrains();
        });
      });
    } catch (err) {
      list.innerHTML = `<div class="hf-row"><span class="memory-meta">Search error: ${escapeHtml(err.message)}</span></div>`;
    }
  }

  // --- Defensive Self-Audit ---
  async function runSiteAudit(url) {
    const out = $('site-audit-output');
    out.innerHTML = '<div class="audit-report-card">Inspecting HTTP security headers &amp; TLS posture...</div>';
    try {
      const rep = await api('/api/jarvis/audit/site', {
        method: 'POST',
        body: JSON.stringify({ url }),
      });
      if (rep.error) {
        out.innerHTML = `<div class="audit-report-card"><strong>Error:</strong> ${escapeHtml(rep.error)}</div>`;
        return;
      }
      const findingsHtml = (rep.findings || []).length
        ? rep.findings.map((f) => `
            <div class="finding-row">
              <div><span class="fit-badge ${escapeHtml(f.severity)}">${escapeHtml(f.severity)}</span> <strong>${escapeHtml(f.title)}</strong></div>
              <div class="memory-meta">${escapeHtml(f.detail)}</div>
              <div><code>${escapeHtml(f.remediation)}</code></div>
            </div>
          `).join('')
        : '<div class="finding-row"><strong>All core security headers verified.</strong></div>';

      out.innerHTML = `
        <div class="audit-report-card">
          <div class="audit-summary-row">
            <div class="grade-box">${escapeHtml(rep.grade)}</div>
            <div>
              <div><strong>${escapeHtml(rep.final_url)}</strong> · Score ${rep.score}/100 · HTTP ${rep.status_code}</div>
              <div class="memory-meta">TLS: ${rep.tls.enabled ? `${escapeHtml(rep.tls.protocol)} (${rep.tls.days_remaining ?? '?'} days left)` : 'Plaintext HTTP'}</div>
            </div>
          </div>
          ${findingsHtml}
        </div>
      `;
    } catch (err) {
      out.innerHTML = `<div class="audit-report-card">Audit failed: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function runRepoAudit(path) {
    const out = $('repo-audit-output');
    out.innerHTML = '<div class="audit-report-card">Scanning local codebase...</div>';
    try {
      const rep = await api('/api/jarvis/audit/repo', {
        method: 'POST',
        body: JSON.stringify({ path }),
      });
      if (rep.error) {
        out.innerHTML = `<div class="audit-report-card"><strong>Error:</strong> ${escapeHtml(rep.error)}</div>`;
        return;
      }
      const findingsHtml = (rep.findings || []).length
        ? rep.findings.map((f) => `
            <div class="finding-row">
              <div><span class="fit-badge ${escapeHtml(f.severity)}">${escapeHtml(f.severity)}</span> <strong>${escapeHtml(f.title)}</strong> <span class="memory-meta">${escapeHtml(f.location)}</span></div>
              <div class="memory-meta">${escapeHtml(f.detail)}</div>
              <div class="memory-meta">Fix: ${escapeHtml(f.remediation)}</div>
            </div>
          `).join('')
        : '<div class="finding-row"><strong>Clean scan — no hardcoded secrets or risky patterns detected.</strong></div>';

      out.innerHTML = `
        <div class="audit-report-card">
          <div><strong>${escapeHtml(rep.path)}</strong> · ${rep.files_scanned} files scanned · Risk: <span class="fit-badge ${escapeHtml(rep.risk_level.toLowerCase())}">${escapeHtml(rep.risk_level)}</span></div>
          ${findingsHtml}
        </div>
      `;
    } catch (err) {
      out.innerHTML = `<div class="audit-report-card">Scan failed: ${escapeHtml(err.message)}</div>`;
    }
  }

  // --- Doctor View ---
  async function loadDoctor() {
    const list = $('doctor-list');
    try {
      const data = await api('/api/jarvis/doctor');
      list.innerHTML = (data.checks || []).map((c) => `
        <div class="doctor-row">
          <div>
            <div><strong>${escapeHtml(c.name)}</strong></div>
            <div class="memory-meta">${escapeHtml(c.detail)}</div>
            ${c.fix ? `<div class="memory-meta" style="color:#f59e0b;margin-top:3px;">Fix: ${escapeHtml(c.fix)}</div>` : ''}
          </div>
          <span class="fit-badge ${escapeHtml(c.status)}">${escapeHtml(c.status)}</span>
        </div>
      `).join('');
    } catch (err) {
      list.innerHTML = `<div class="doctor-row">Doctor failed: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function loadStatusQuiet() {
    try {
      const st = await api('/api/jarvis/status');
      $('nav-memory-count').textContent = String(st.memory_count || 0);
      updateActiveBrainUI(st.active_brain, st.active_brain_ready);
    } catch (_) {}
  }

  // --- Wire Events & Boot ---
  function bindEvents() {
    document.querySelectorAll('.nav-item').forEach((btn) => {
      btn.addEventListener('click', () => switchView(btn.dataset.view));
    });
    document.querySelectorAll('[data-jump]').forEach((btn) => {
      btn.addEventListener('click', () => switchView(btn.dataset.jump));
    });
    $('rail-brain-btn').addEventListener('click', () => switchView('brains'));

    // Starter prompts
    document.querySelectorAll('.starter-btn').forEach((btn) => {
      btn.addEventListener('click', () => sendChat(btn.dataset.prompt));
    });

    // Chat submit + auto-grow textarea
    const input = $('chat-input');
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 140) + 'px';
    });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChat(input.value);
      }
    });
    $('chat-form').addEventListener('submit', (e) => {
      e.preventDefault();
      sendChat(input.value);
    });

    $('btn-clear-chat').addEventListener('click', async () => {
      await api('/api/jarvis/history', { method: 'DELETE' });
      const box = $('chat-messages');
      box.querySelectorAll('.msg').forEach((m) => m.remove());
      $('empty-state').classList.remove('hidden');
      toast('Conversation cleared');
    });

    // Voice controls
    $('btn-mic').addEventListener('click', () => {
      if (state.recognition) stopSpeechRecognition();
      else startSpeechRecognition(false);
    });
    $('btn-stop-voice').addEventListener('click', () => {
      state.wakeMode = false;
      $('btn-wake-mode').classList.remove('active-state');
      $('btn-wake-mode').textContent = 'Enable Hands-Free "Jarvis" Mode';
      stopSpeechRecognition();
    });
    $('btn-wake-mode').addEventListener('click', () => {
      state.wakeMode = !state.wakeMode;
      const btn = $('btn-wake-mode');
      if (state.wakeMode) {
        btn.classList.add('active-state');
        btn.textContent = 'Listening for "Hey Jarvis" (Click to Stop)';
        $('toggle-tts').checked = true;
        startSpeechRecognition(true);
      } else {
        btn.classList.remove('active-state');
        btn.textContent = 'Enable Hands-Free "Jarvis" Mode';
        stopSpeechRecognition();
      }
    });
    $('btn-stop-tts').addEventListener('click', () => {
      window.speechSynthesis?.cancel();
      setCoreMode('standby');
      $('btn-stop-tts').classList.add('hidden');
    });

    // Brains
    $('btn-refresh-brains').addEventListener('click', loadBrains);
    $('custom-brain-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      try {
        await api('/api/jarvis/brains/custom', {
          method: 'POST',
          body: JSON.stringify({
            brain_id: $('cb-id').value,
            name: $('cb-name').value,
            provider: $('cb-provider').value,
            model: $('cb-model').value,
            host: $('cb-host').value,
          }),
        });
        toast('Custom brain registered!');
        $('custom-brain-form').reset();
        loadBrains();
      } catch (err) {
        toast(`Could not add brain: ${err.message}`);
      }
    });

    // Memory
    $('memory-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const val = $('memory-input').value.trim();
      if (!val) return;
      await api('/api/jarvis/memory', {
        method: 'POST',
        body: JSON.stringify({ text: val, source: 'hud' }),
      });
      $('memory-input').value = '';
      toast('Fact saved to persistent OpenJarvis memory');
      loadMemories();
    });

    // Hardware & HF
    $('btn-refresh-hardware').addEventListener('click', loadHardware);
    $('hf-search-form').addEventListener('submit', (e) => {
      e.preventDefault();
      searchHF();
    });

    // Audit
    $('site-audit-form').addEventListener('submit', (e) => {
      e.preventDefault();
      runSiteAudit($('audit-url-input').value);
    });
    $('repo-audit-form').addEventListener('submit', (e) => {
      e.preventDefault();
      runRepoAudit($('audit-path-input').value);
    });

    // Doctor
    $('btn-refresh-doctor').addEventListener('click', loadDoctor);
  }

  async function boot() {
    bindEvents();
    initTTSVoices();
    requestAnimationFrame(drawArcCore);

    await Promise.all([loadStatusQuiet(), loadBrains()]);
    try {
      const hist = await api('/api/jarvis/history');
      if (hist.messages && hist.messages.length) {
        hist.messages.forEach(renderMessage);
      }
    } catch (_) {}
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
