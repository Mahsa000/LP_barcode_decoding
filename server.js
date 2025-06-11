const express = require('express');
const path = require('path');
const fetch = require('node-fetch');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = process.env.PORT || 3006;

// The base FastAPI URL
// const FASTAPI_URL = 'http://127.0.0.1:8000';
const FASTAPI_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';  // default for local runs

// Global logger middleware
app.use((req, res, next) => {
  console.log("Incoming request:", req.method, req.url);
  next();
});

// --------------------------------------------------------------------------
// 1) Proxy all /plots requests to FastAPI so that dynamic plots are served
// --------------------------------------------------------------------------
app.use('/plots', createProxyMiddleware({
  target: `${FASTAPI_URL}/plots`,
  changeOrigin: true,
  onProxyReq: (proxyReq, req, res) => {
    console.log(`Proxying ${req.method} ${req.originalUrl} to ${FASTAPI_URL}`);
  }
}));

// 2) Serve JSON
app.use(express.json());

// 3) Serve a local "public" folder for static HTML/JS
app.use(express.static(path.join(__dirname, 'public')));

// Root route => index.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Helper to parse FastAPI response as JSON or show error
async function handleFastApiResponse(resp) {
  const clone = resp.clone();
  let data;
  try {
    data = await resp.json();
  } catch (err) {
    const txt = await clone.text();
    throw new Error(txt);
  }
  if (!resp.ok) {
    throw new Error(data.detail || 'FastAPI error');
  }
  return data;
}

// --------------------------------------------------------------------------
// 4) Pass-through endpoints to FastAPI
// --------------------------------------------------------------------------

// (a) Analyze raw data
app.post('/analyze-raw-data', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/analyze-raw-data`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// (b) Load & Filter Data
app.post('/load-and-filter-data', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/load-and-filter-data`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    // const data = await handleFastApiResponse(resp);
    // res.json({ status: 'success', data });
    const data = await handleFastApiResponse(resp)
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// (c) Show matched cc
app.post('/show-matched-cc', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/plot-same-cc-across-samples`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', fastApiResponse: data });
  } catch (err) {
    console.error('[show-matched-cc] error:', err);
    res.status(500).json({ error: err.message });
  }
});

// (d) get-samples
app.get('/get-samples', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/get-samples`;
    const resp = await fetch(url);
    const data = await handleFastApiResponse(resp);
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// (e) plot-multi-sample-iter => pass HTML
app.get('/plot-multi-sample-iter', async (req, res) => {
  try {
    const qp = new URLSearchParams(req.query).toString();
    const url = `${FASTAPI_URL}/plot-multi-sample-iter?${qp}`;
    const resp = await fetch(url);
    const text = await resp.text();
    res.status(resp.status).send(text);
  } catch (err) {
    res.status(500).send(err.message);
  }
});

// (f) get-clickable-areas
app.get('/get-clickable-areas', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/get-clickable-areas?plot_filename=${encodeURIComponent(req.query.plot_filename)}`;
    const resp = await fetch(url);
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', clickable_areas: data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// (g) line-peaks-view => pass HTML
app.get('/line-peaks-view', async (req, res) => {
  try {
    const qp = new URLSearchParams(req.query).toString();
    const url = `${FASTAPI_URL}/line-peaks-view?${qp}`;
    const resp = await fetch(url);
    const html = await resp.text();
    res.status(resp.status).send(html);
  } catch (err) {
    res.status(500).send(err.message);
  }
});

// (h) LID editing endpoints
app.post('/delete-lid', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/delete-lid`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/undo-delete-lid', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/undo-delete-lid`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/maybe-lid', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/maybe-lid`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/split-lid-dbscan', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/split-lid-dbscan`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/undo-split-lid', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/undo-split-lid`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// (i) assign-submid
app.post('/assign-submid', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/assign-submid`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// (j) undo-submid
app.post('/undo-submid', async (req, res) => {
  try {
    const payload = JSON.stringify(req.body);
    const url = `${FASTAPI_URL}/undo-submid`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', data });
  } catch (error) {
    console.error('Error in /undo-submid:', error);
    res.status(500).json({ error: error.message });
  }
});

// (k) Label-Level changes => apply-label-changes, discard-label-changes
app.post('/apply-label-changes', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/apply-label-changes`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    // data may contain {status, message}
    res.json({ status: 'success', ...data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/discard-label-changes', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/discard-label-changes`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', ...data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// (l) merge-lids
app.post('/merge-lids', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/merge-lids`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// (m) Load Label Data
app.post('/load-label-data', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/load-label-data`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', ...data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/load-label-data', async (req, res) => {
  try {
    const url = `${FASTAPI_URL}/load-label-data`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await handleFastApiResponse(resp);
    res.json({ status: 'success', ...data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

function loadLabelData(sampleName, labelVal) {
  const payload = {
    sample_name: sampleName,
    label: labelVal
  };
  fetch('/load-label-data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(r => r.json())
  .then(d => {
     alert('Label data loaded: ' + JSON.stringify(d));
     // Optionally reload the page
     // location.reload();
  })
  .catch(e => alert('Error loading label data: ' + e));
}


// --------------------------------------------------------------------------
// Start the server
// --------------------------------------------------------------------------
app.listen(PORT, () => {
  console.log(`Node server listening on http://localhost:${PORT}`);
});
