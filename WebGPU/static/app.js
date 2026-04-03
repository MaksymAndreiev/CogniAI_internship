// =============================================================
//  MNIST CNN Inference  —  app.js
//  PyTorch CNN exported to ONNX → runs in browser via
//  ONNX Runtime Web (WebGPU execution provider)
//  Input normalisation: (pixel/255 - 0.1307) / 0.3081
// =============================================================

// Load ONNX Runtime via script tag (avoids CORS issues)
let ort = null;

async function loadORT() {
  try {
    // Wait for window.ort to be available (loaded via script tag in HTML)
    for (let i = 0; i < 50; i++) {
      if (typeof window.ort !== 'undefined') {
        ort = window.ort;
        ort.env.wasm.numThreads = 1;
        ort.env.logLevel = 'error';
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
        return true;
      }
      await new Promise(r => setTimeout(r, 100));
    }
    throw new Error('ONNX Runtime did not load within 5 seconds');
  } catch (err) {
    throw new Error('Failed to load ONNX Runtime: ' + err.message);
  }
}

// UI handles (initialized when DOM is ready)
let statusDot, statusText, predDigit, predConf, gpuInfo, barsWrap;
let btnPredict, btnClear, drawCanvas, ctx2d;

function initDOM() {
  statusDot  = document.getElementById('status-dot');
  statusText = document.getElementById('status-text');
  predDigit  = document.getElementById('pred-digit');
  predConf   = document.getElementById('pred-conf');
  gpuInfo    = document.getElementById('gpu-info');
  barsWrap   = document.getElementById('bars-container');
  btnPredict = document.getElementById('btn-predict');
  btnClear   = document.getElementById('btn-clear');
  drawCanvas = document.getElementById('draw-canvas');
  ctx2d      = drawCanvas.getContext('2d');
  return !!statusDot;
}

let barFills = [];
const CANVAS_SIZE = 280;
let painting = false;

function setupCanvas() {
  drawCanvas.width  = CANVAS_SIZE;
  drawCanvas.height = CANVAS_SIZE;
  ctx2d.fillStyle = '#000';
  ctx2d.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
}

function setupBarChart() {
  for (let i = 0; i < 10; i++) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML =
      '<span class="bar-digit">' + i + '</span>' +
      '<div class="bar-track"><div class="bar-fill" id="bar-' + i + '"></div></div>' +
      '<span class="bar-pct" id="pct-' + i + '">—</span>';
    barsWrap.appendChild(row);
    barFills.push(document.getElementById('bar-' + i));
  }
}

function getPos(e) {
  const rect   = drawCanvas.getBoundingClientRect();
  const src    = e.touches ? e.touches[0] : e;
  const scaleX = drawCanvas.width  / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  return [
    (src.clientX - rect.left) * scaleX,
    (src.clientY - rect.top)  * scaleY,
  ];
}

function drawDot(x, y) {
  ctx2d.beginPath();
  ctx2d.arc(x, y, 9, 0, Math.PI * 2);
  ctx2d.fillStyle = '#fff';
  ctx2d.fill();
}

function setupEventListeners() {
  drawCanvas.addEventListener('pointerdown', function(e) {
    e.preventDefault();
    painting = true;
    var pos = getPos(e);
    drawDot(pos[0], pos[1]);
    ctx2d.beginPath();
    ctx2d.moveTo(pos[0], pos[1]);
  });

  drawCanvas.addEventListener('pointermove', function(e) {
    e.preventDefault();
    if (!painting) return;
    var pos = getPos(e);
    ctx2d.lineTo(pos[0], pos[1]);
    ctx2d.strokeStyle = '#fff';
    ctx2d.lineWidth   = 18;
    ctx2d.lineCap     = 'round';
    ctx2d.lineJoin    = 'round';
    ctx2d.stroke();
    ctx2d.beginPath();
    ctx2d.moveTo(pos[0], pos[1]);
  });

  drawCanvas.addEventListener('pointerup', function(e) {
    e.preventDefault();
    painting = false;
  });

  drawCanvas.addEventListener('pointerleave', function() {
    painting = false;
  });

  btnClear.addEventListener('click', function() {
    ctx2d.fillStyle = '#000';
    ctx2d.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    resetResults();
  });

  btnPredict.addEventListener('click', handlePredict);
}

function resetResults() {
  predDigit.textContent = '—';
  predConf.textContent  = '';
  for (var i = 0; i < 10; i++) {
    barFills[i].style.width = '0%';
    barFills[i].classList.remove('top');
    document.getElementById('pct-' + i).textContent = '—';
  }
}

const MEAN = 0.1307, STD = 0.3081;

function canvasToTensor() {
  var off = document.createElement('canvas');
  off.width = off.height = 28;
  var oc = off.getContext('2d');
  oc.drawImage(drawCanvas, 0, 0, 28, 28);
  var rgba = oc.getImageData(0, 0, 28, 28).data;
  var data = new Float32Array(784);
  for (var i = 0; i < 784; i++) {
    data[i] = (rgba[i * 4] / 255.0 - MEAN) / STD;
  }
  return new ort.Tensor('float32', data, [1, 1, 28, 28]);
}

function softmax(logits) {
  var max = Math.max.apply(null, logits);
  var exp = logits.map(function(x) { return Math.exp(x - max); });
  var sum = exp.reduce(function(a, b) { return a + b; }, 0);
  return exp.map(function(x) { return x / sum; });
}

let ortSession = null;

async function initORT() {
  setStatus('loading', 'Fetching CNN model...');
  var resp = await fetch('/static/MulticlassCNN.onnx');
  if (!resp.ok) {
    throw new Error('Model fetch failed: ' + resp.status + ' — did you run train.py?');
  }
  var modelBuffer = await resp.arrayBuffer();

  setStatus('loading', 'Initializing inference session...');

  var executionProviders = ['wasm'];

  if (navigator.gpu) {
    try {
      var adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        var device = await adapter.requestDevice();
        ort.env.webgpu.device = device;
        executionProviders = [{ name: 'webgpu', preferredLayout: 'NHWC' }, 'wasm'];
        var info = {};
        try { info = await adapter.requestAdapterInfo(); } catch(e) {}
        gpuInfo.textContent = 'WebGPU: ' + (info.description || info.vendor || 'GPU active');
      }
    } catch (e) {
      console.warn('WebGPU failed, using WASM:', e);
      gpuInfo.textContent = 'WebGPU unavailable — using WASM';
    }
  } else {
    gpuInfo.textContent = 'WebGPU not supported — using WASM';
  }

  ortSession = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: executionProviders,
    graphOptimizationLevel: 'all',
  });

  console.log('Input names:',  ortSession.inputNames);
  console.log('Output names:', ortSession.outputNames);

  setStatus('ok', 'Ready — draw a digit and click Predict');
  btnPredict.disabled = false;
}

async function runInference() {
  var inputName  = ortSession.inputNames[0];
  var outputName = ortSession.outputNames[0];
  var tensor     = canvasToTensor();
  var feeds      = {};
  feeds[inputName] = tensor;
  var t0      = performance.now();
  var results = await ortSession.run(feeds);
  var dt      = (performance.now() - t0).toFixed(1);
  var logits  = Array.from(results[outputName].data);
  return { probs: softmax(logits), dt: dt };
}

async function handlePredict() {
  btnPredict.disabled    = true;
  btnPredict.textContent = '...';
  try {
    var result = await runInference();
    var probs  = result.probs;
    var dt     = result.dt;
    var best   = probs.indexOf(Math.max.apply(null, probs));

    predDigit.textContent = best;
    predConf.textContent  = (probs[best] * 100).toFixed(1) + '% confidence · ' + dt + ' ms';

    for (var i = 0; i < 10; i++) {
      var pct = (probs[i] * 100).toFixed(1);
      barFills[i].style.width = pct + '%';
      barFills[i].classList.toggle('top', i === best);
      document.getElementById('pct-' + i).textContent = pct + '%';
    }
  } catch (err) {
    setStatus('err', 'Inference error: ' + err.message);
    console.error(err);
  }
  btnPredict.disabled    = false;
  btnPredict.textContent = '▶ Predict';
}

function setStatus(state, msg) {
  if (!statusDot) return;
  statusDot.className    = 'dot ' + (state === 'ok' ? 'ok' : state === 'err' ? 'err' : 'loading');
  statusText.textContent = msg;
}

async function boot() {
  try {
    await loadORT();
    initDOM();
    setupCanvas();
    setupBarChart();
    setupEventListeners();
    await initORT();
  } catch (err) {
    setStatus('err', err.message);
    if (gpuInfo) gpuInfo.textContent = err.message;
    console.error('[MNIST boot]', err);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}