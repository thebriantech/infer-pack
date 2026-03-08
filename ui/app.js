/**
 * InferPack Dashboard — minimal UI client.
 *
 * This is a pure REST API client.  No business logic lives here.
 * The UI is optional and replaceable.
 */

const API = '/api/v1';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let events = [];

function logEvent(level, message) {
    events.unshift({ timestamp: Date.now() / 1000, level, message });
    if (events.length > 100) events.pop();
    renderEvents();
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function api(method, path, body) {
    const opts = { method, headers: {} };
    if (body instanceof FormData) {
        opts.body = body;
    } else if (body) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
    }
    const res = await fetch(API + path, opts);
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status}: ${text}`);
    }
    return res.json();
}

// ---------------------------------------------------------------------------
// Refresh
// ---------------------------------------------------------------------------

async function refresh() {
    try {
        const health = await api('GET', '/health');
        const versionEl = document.getElementById('version');
        versionEl.textContent = `v${health.version}`;
        versionEl.className = 'badge version';

        const tritonEl = document.getElementById('triton-status');
        tritonEl.textContent = health.triton_ready ? 'Triton Ready' : 'Triton Not Ready';
        tritonEl.className = `badge ${health.triton_ready ? 'ok' : 'degraded'}`;

        const status = await api('GET', '/status');
        renderPipelines(status.pipelines);
        renderModels(status.models);
        populatePipelineSelect(status.pipelines);
    } catch (err) {
        logEvent('error', `Refresh failed: ${err.message}`);
    }
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

function renderPipelines(pipelines) {
    const container = document.getElementById('pipelines-list');
    container.innerHTML = '';

    for (const p of pipelines) {
        const card = document.createElement('div');
        card.className = 'pipeline-card';

        const stateColor = p.state === 'active' ? 'ok'
            : p.state === 'error' ? 'error'
            : 'version';

        card.innerHTML = `
            <h3>${p.name} <span class="badge ${stateColor}">${p.state}</span>
                ${p.builtin ? '<span class="badge version">built-in</span>' : ''}
            </h3>
            <p>${p.description}</p>
            <div class="models">Models: ${p.models.join(', ')}</div>
            <div class="models">Stages: ${p.stages.join(' → ')}</div>
        `;

        if (p.state === 'inactive' || p.state === 'error') {
            const btn = document.createElement('button');
            btn.className = 'btn-activate';
            btn.textContent = 'Activate';
            btn.onclick = () => activatePipeline(p.name);
            card.appendChild(btn);
        } else if (p.state === 'active') {
            const btn = document.createElement('button');
            btn.className = 'btn-deactivate';
            btn.textContent = 'Deactivate';
            btn.onclick = () => deactivatePipeline(p.name);
            card.appendChild(btn);
        }

        container.appendChild(card);
    }
}

function renderModels(models) {
    const tbody = document.querySelector('#models-table tbody');
    tbody.innerHTML = '';
    for (const m of models) {
        const tr = document.createElement('tr');
        const lastUsed = m.last_used > 0
            ? new Date(m.last_used * 1000).toLocaleTimeString()
            : '—';
        tr.innerHTML = `
            <td>${m.name}</td>
            <td><span class="badge ${modelStateClass(m.state)}">${m.state}</span></td>
            <td>${m.ref_count}</td>
            <td>${m.active_inferences}</td>
            <td>${lastUsed}</td>
        `;
        tbody.appendChild(tr);
    }
}

function modelStateClass(state) {
    switch (state) {
        case 'ready': return 'ok';
        case 'loading': case 'warming': return 'degraded';
        case 'idle': return 'version';
        case 'unloaded': case 'unloading': return 'version';
        default: return 'error';
    }
}

function renderEvents() {
    const el = document.getElementById('events-log');
    el.innerHTML = events.map(e => {
        const t = new Date(e.timestamp * 1000).toLocaleTimeString();
        return `<div>[${t}] [${e.level}] ${e.message}</div>`;
    }).join('');
}

function populatePipelineSelect(pipelines) {
    const select = document.getElementById('exec-pipeline');
    const current = select.value;
    select.innerHTML = '';
    for (const p of pipelines.filter(p => p.state === 'active')) {
        const opt = document.createElement('option');
        opt.value = p.name;
        opt.textContent = p.name;
        select.appendChild(opt);
    }
    if (current) select.value = current;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async function activatePipeline(name) {
    try {
        await api('POST', `/pipelines/${name}/activate`);
        logEvent('info', `Pipeline ${name} activated`);
        await refresh();
    } catch (err) {
        logEvent('error', `Activate ${name}: ${err.message}`);
    }
}

async function deactivatePipeline(name) {
    try {
        await api('POST', `/pipelines/${name}/deactivate`);
        logEvent('info', `Pipeline ${name} deactivated`);
        await refresh();
    } catch (err) {
        logEvent('error', `Deactivate ${name}: ${err.message}`);
    }
}

document.getElementById('execute-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('exec-pipeline').value;
    if (!name) return;

    const resultEl = document.getElementById('exec-result');
    resultEl.style.display = 'block';
    resultEl.textContent = 'Running…';

    const formData = new FormData();
    const img = document.getElementById('exec-image').files[0];
    const img1 = document.getElementById('exec-image-1').files[0];
    const img2 = document.getElementById('exec-image-2').files[0];

    if (img) formData.append('image', img);
    if (img1) formData.append('image_1', img1);
    if (img2) formData.append('image_2', img2);

    try {
        const result = await api('POST', `/pipelines/${name}/execute`, formData);
        resultEl.textContent = JSON.stringify(result, null, 2);
        logEvent('info', `Pipeline ${name} executed (success=${result.success})`);
    } catch (err) {
        resultEl.textContent = `Error: ${err.message}`;
        logEvent('error', `Execute ${name}: ${err.message}`);
    }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

refresh();
setInterval(refresh, 5000);
