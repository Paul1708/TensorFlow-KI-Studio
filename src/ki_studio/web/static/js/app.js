// --- Application State ---
let currentView = 'dashboard';
let currentMode = 'zahlen'; // 'zahlen' or 'smiley'
let drawTool = 'brush';     // 'brush' or 'eraser'
let isDrawing = false;
let activeButton = 0;
let lastX = 0, lastY = 0;

let brainData = null;       // Latest response from /api/analyze
let lockedOutputIdx = null; // null = follow winner automatically, or integer index
let hoveredNode = null;     // { colType, layerIdx, nodeIdx, x, y, ... }
let hitNodes = [];          // Populated every render frame for mouse hover/click detection

// Smooth node brightness interpolation arrays (dynamically sized to any trained architecture!)
let smoothConv = [];
let smoothHiddenLayers = [];
let smoothOutputs = new Float32Array(10);
let pulsePhase = 0;

// KI-Labor State
let labTargetMode = 'zahlen';
let latestStatus = null;
let latestTrainMetrics = [];
let lastTrainedMode = 'zahlen';

// Throttle state for real-time drawing inference
let analyzeInFlight = false;
let analyzePending = false;
let lastAnalyzeTime = 0;
let currentSourceLabel = 'Live-Zeichnung';

// --- Setup Drawing Canvas ---
const drawCanvas = document.getElementById('drawCanvas');
const drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });
resetDrawSurface();

function resetDrawSurface() {
  drawCtx.fillStyle = '#ffffff';
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
}

function setDrawTool(tool) {
  drawTool = tool;
  document.getElementById('btnBrush').classList.toggle('active', tool === 'brush');
  document.getElementById('btnEraser').classList.toggle('active', tool === 'eraser');
}

function getCanvasCoords(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  return [
    (e.clientX - rect.left) * scaleX,
    (e.clientY - rect.top) * scaleY
  ];
}

drawCanvas.addEventListener('contextmenu', e => e.preventDefault());

drawCanvas.addEventListener('pointerdown', e => {
  e.preventDefault();
  drawCanvas.setPointerCapture(e.pointerId);
  isDrawing = true;
  activeButton = e.button;
  const [x, y] = getCanvasCoords(e);
  lastX = x;
  lastY = y;
  paintSegment(x, y, x, y, activeButton === 2 || drawTool === 'eraser');
  currentSourceLabel = 'Live-Zeichnung';
  scheduleLiveAnalyze();
});

drawCanvas.addEventListener('pointermove', e => {
  if (!isDrawing) return;
  const [x, y] = getCanvasCoords(e);
  paintSegment(lastX, lastY, x, y, activeButton === 2 || drawTool === 'eraser');
  lastX = x;
  lastY = y;
  scheduleLiveAnalyze();
});

drawCanvas.addEventListener('pointerup', e => {
  if (!isDrawing) return;
  isDrawing = false;
  triggerAnalyzeNow();
});

function paintSegment(x0, y0, x1, y1, erase) {
  const baseRadius = currentMode === 'zahlen' ? 10 : 6;
  const radius = erase ? 16 : baseRadius;
  drawCtx.save();
  drawCtx.strokeStyle = erase ? '#ffffff' : '#000000';
  drawCtx.fillStyle = erase ? '#ffffff' : '#000000';
  drawCtx.lineWidth = radius * 2;
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';
  drawCtx.beginPath();
  drawCtx.moveTo(x0, y0);
  drawCtx.lineTo(x1, y1);
  drawCtx.stroke();
  drawCtx.restore();
}

function clearCanvas() {
  resetDrawSurface();
  currentSourceLabel = 'Leeres Feld';
  triggerAnalyzeNow();
}

function saveCanvasPNG() {
  const link = document.createElement('a');
  link.download = currentMode === 'zahlen' ? 'meine_zahl.png' : 'mein_smiley.png';
  link.href = drawCanvas.toDataURL('image/png');
  link.click();
}

// --- Navigation ---
function switchView(view) {
  currentView = view;
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.view === view);
  });

  document.getElementById('view-dashboard').classList.toggle('hidden', view !== 'dashboard');
  document.getElementById('view-studio').classList.toggle('hidden', view !== 'zahlen' && view !== 'smiley');
  document.getElementById('view-training').classList.toggle('hidden', view !== 'training');

  if (view === 'zahlen' || view === 'smiley') {
    const modeChanged = currentMode !== view;
    currentMode = view;
    lockedOutputIdx = null;
    updateFocusPill();

    const isZahlen = view === 'zahlen';
    document.getElementById('padTitle').textContent = isZahlen
      ? 'Zeichenfeld (Zahlen 0–9)'
      : 'Zeichenfeld (Smiley)';
    document.getElementById('btnRandom').textContent = isZahlen
      ? '🎲 Zufalls-Zahl'
      : '🎲 Zufalls-Smiley';
    document.getElementById('inputResLabel').textContent = isZahlen ? '28×28 px' : '48×48 px';
    document.getElementById('happinessBox').classList.toggle('hidden', isZahlen);

    if (modeChanged) {
      resetDrawSurface();
      brainData = null;
      hoveredNode = null;
      smoothConv = [];
      smoothHiddenLayers = [];
      smoothOutputs.fill(0);
    }
    resizeBrainCanvas();
    triggerAnalyzeNow();
  } else if (view === 'training') {
    refreshStatus();
    onLabParamChange();
    setTimeout(() => renderTrainingChart(latestTrainMetrics), 30);
  }
}

function openLabForCurrentMode() {
  selectLabTarget(currentMode);
  switchView('training');
  loadActiveModelIntoControls();
}

// --- Real-Time Brain API Communication ---
function scheduleLiveAnalyze() {
  const now = performance.now();
  if (analyzeInFlight || (now - lastAnalyzeTime < 65)) {
    analyzePending = true;
    return;
  }
  runAnalyzeRequest();
}

function triggerAnalyzeNow() {
  if (analyzeInFlight) {
    analyzePending = true;
    return;
  }
  runAnalyzeRequest();
}

async function runAnalyzeRequest() {
  analyzeInFlight = true;
  analyzePending = false;
  lastAnalyzeTime = performance.now();

  try {
    const b64 = drawCanvas.toDataURL('image/png');
    const resp = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: currentMode, image_b64: b64, return_canvas: false })
    });
    if (resp.ok) {
      const data = await resp.json();
      if (!data.error) {
        applyBrainData(data, currentSourceLabel);
      }
    }
  } catch (err) {
    console.error(err);
  } finally {
    analyzeInFlight = false;
    if (analyzePending) {
      setTimeout(scheduleLiveAnalyze, 40);
    }
  }
}

async function loadRandomSample() {
  try {
    const resp = await fetch('/api/random', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: currentMode })
    });
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.canvas_png_b64) {
      await drawBase64ToPad(data.canvas_png_b64);
    }
    currentSourceLabel = data.source_label || 'Zufalls-Beispiel';
    applyBrainData(data, currentSourceLabel);
  } catch (err) {
    console.error(err);
  }
}

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  event.target.value = '';

  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: currentMode,
          image_b64: reader.result,
          return_canvas: true
        })
      });
      const data = await resp.json();
      if (data.error) {
        alert('Fehler beim Bildladen: ' + data.error);
        return;
      }
      if (data.canvas_png_b64) {
        await drawBase64ToPad(data.canvas_png_b64);
      }
      currentSourceLabel = 'Datei: ' + file.name;
      applyBrainData(data, currentSourceLabel);
    } catch (err) {
      alert('Fehler: ' + err);
    }
  };
  reader.readAsDataURL(file);
}

function drawBase64ToPad(b64) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      resetDrawSurface();
      drawCtx.drawImage(img, 0, 0, drawCanvas.width, drawCanvas.height);
      resolve();
    };
    img.src = 'data:image/png;base64,' + b64;
  });
}

function getInspectedOutputIndex() {
  if (!brainData || !brainData.outputs || brainData.outputs.length === 0) return 0;
  if (lockedOutputIdx !== null && lockedOutputIdx < brainData.outputs.length) {
    return lockedOutputIdx;
  }
  if (brainData.prediction) {
    return brainData.prediction.index;
  }
  return 0;
}

function setOutputFocus(idx) {
  lockedOutputIdx = idx;
  updateFocusPill();
  updateMathExplainer();
}

function resetOutputFocus() {
  lockedOutputIdx = null;
  updateFocusPill();
  updateMathExplainer();
}

function updateFocusPill() {
  const pill = document.getElementById('focusResetPill');
  if (lockedOutputIdx === null) {
    pill.textContent = '🎯 Fokus: Automatisch auf Gewinner';
  } else if (brainData && brainData.outputs[lockedOutputIdx]) {
    const o = brainData.outputs[lockedOutputIdx];
    pill.textContent = `📌 Fokus fixiert auf: ${o.label} (Klick zum Zurücksetzen)`;
  }
}

function formatMetaSummary(meta) {
  if (!meta) return 'Standard-Architektur aktiv';
  const convPart = (meta.conv_layers && meta.conv_layers.length > 0)
    ? `${meta.conv_layers.length} Filter-Schicht${meta.conv_layers.length > 1 ? 'en' : ''} (${meta.conv_layers.join('→')})`
    : '0 Filter-Schichten (Reines Pixel-MLP)';
  const hidList = meta.hidden_layers || [64];
  const hidPart = `${hidList.length} Denkschicht${hidList.length > 1 ? 'en' : ''} (${hidList.join('→')} ${(meta.activation || 'relu').toUpperCase()})`;
  const optPart = meta.optimizer ? ` • ${(meta.optimizer).toUpperCase()} (LR ${meta.learning_rate ?? 0.001})` : '';
  const accPart = (meta.test_accuracy !== null && meta.test_accuracy !== undefined)
    ? ` • Test-Genauigkeit: ${Number(meta.test_accuracy).toFixed(2)} %`
    : '';
  return `${meta.arch_type || 'CNN'}: ${convPart} • ${hidPart}${optPart}${accPart}`;
}

// --- Update UI Panels with Brain Data ---
function applyBrainData(data, sourceText) {
  brainData = data;

  if (data.preview_b64) {
    document.getElementById('aiPreviewImg').src = 'data:image/png;base64,' + data.preview_b64;
  }

  // Dynamic Architecture Header Tag & Summary Strip
  const meta = data.model_meta;
  const prefix = currentMode === 'zahlen' ? 'Zahlen' : 'Smiley';
  if (meta) {
    const actUp = (meta.activation || 'relu').toUpperCase();
    const accBadge = (meta.test_accuracy !== null && meta.test_accuracy !== undefined)
      ? ` • ${Number(meta.test_accuracy).toFixed(1)} %`
      : '';
    document.getElementById('brainModelTag').textContent =
      `${prefix}-${meta.arch_type} • ${meta.total_nodes} Knoten • ${actUp}${accBadge}`;
    document.getElementById('studioArchSummary').textContent =
      `Trainiertes Gehirn: ${formatMetaSummary(meta)}`;
  }

  // Top Conv1 Feature Maps (or MLP Direct Pixel Note when conv_layers.length === 0!)
  const fmapBox = document.getElementById('topFiltersBox');
  const eyeTitle = document.getElementById('eyeBoxTitle');
  fmapBox.innerHTML = '';
  const hasConv = data.conv_layers && data.conv_layers.length > 0;

  if (!hasConv && !data.model_missing) {
    eyeTitle.textContent = '👁️ KI-Eingabe (Direktes Pixel-MLP)';
    const note = document.createElement('div');
    note.className = 'mlp-eye-note';
    note.innerHTML = '<b>Reines MLP aktiv (0 Filter):</b><br>Keine Lupen-Filter – die Pixel fließen direkt in Denkschicht 1!';
    fmapBox.appendChild(note);
  } else {
    eyeTitle.textContent = '👁️ KI-Eingabe & aktivste Augen-Filter';
    if (!data.empty && hasConv && data.conv_layers[0].top_maps && data.conv_layers[0].top_maps.length > 0) {
      data.conv_layers[0].top_maps.forEach(fm => {
        const div = document.createElement('div');
        div.className = 'fmap-item';
        div.innerHTML = `<img src="data:image/png;base64,${fm.png_b64}" alt="F${fm.filter_index}"><span>F#${fm.filter_index}</span>`;
        fmapBox.appendChild(div);
      });
    } else {
      const placeholderCount = hasConv ? Math.min(4, data.conv_layers[0].count || 4) : 4;
      for (let i = 0; i < placeholderCount; i++) {
        const div = document.createElement('div');
        div.className = 'fmap-item';
        div.innerHTML = `<img src="data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA=" alt="-"><span>Filter</span>`;
        fmapBox.appendChild(div);
      }
    }
  }

  // Result Banner
  const badge = document.getElementById('resultBadge');
  const title = document.getElementById('resultTitle');
  const sub = document.getElementById('resultSubtitle');

  if (data.model_missing) {
    badge.textContent = '!';
    title.textContent = 'Modell noch nicht trainiert';
    sub.textContent = 'Wechsle oben zu „KI-Training & Labor“, um das Netz zu trainieren.';
    return;
  }

  if (data.empty || !data.prediction) {
    badge.textContent = '–';
    badge.style.color = '#94a3b8';
    title.textContent = currentMode === 'zahlen'
      ? 'Zeichne eine Zahl (0–9) ins Feld!'
      : 'Zeichne einen Smiley ins Feld!';
    sub.textContent = 'Alle Knoten warten auf deine Eingabe...';
    if (currentMode === 'smiley') {
      document.getElementById('happinessValueText').textContent = 'Glücks-Level: –';
      document.getElementById('happinessFill').style.width = '0%';
    }
  } else {
    const p = data.prediction;
    if (currentMode === 'zahlen') {
      badge.textContent = p.label;
      badge.style.color = '#38bdf8';
      title.textContent = `KI erkennt Zahl: ${p.label}`;
      sub.textContent = `Sicherheit: ${p.confidence.toFixed(2)} %  •  Quelle: ${sourceText}`;
    } else {
      badge.textContent = p.symbol;
      const h = p.happiness ?? 50;
      const col = happinessColor(h);
      badge.style.color = col;
      title.textContent = `${p.label} (${p.symbol})`;
      sub.textContent = `Glücks-Level: ${h.toFixed(1)} %  •  Sicherheit: ${p.confidence.toFixed(1)} %  •  ${sourceText}`;
      const hTxt = document.getElementById('happinessValueText');
      hTxt.textContent = `Glücks-Level: ${h.toFixed(1)} %`;
      hTxt.style.color = col;
      document.getElementById('happinessFill').style.width = `${Math.max(2, h)}%`;
    }
  }

  updateFocusPill();
  updateMathExplainer();
}

function happinessColor(h) {
  if (h < 20) return '#fb7185';
  if (h < 40) return '#fb923c';
  if (h < 60) return '#fbbf24';
  if (h < 82) return '#34d399';
  return '#10b981';
}

// --- Update 3-Step Softmax Probability Explainer ---
function updateMathExplainer() {
  const chipsBox = document.getElementById('step1Chips');
  const step1TitleEl = document.getElementById('step1Title');
  const f1 = document.getElementById('step1Formula');
  const f2 = document.getElementById('step2Formula');
  const f3 = document.getElementById('step3Formula');
  const tbody = document.getElementById('probTableBody');
  const headerTitle = document.getElementById('mathHeaderTitle');

  const hidLayers = (brainData && brainData.hidden_layers && brainData.hidden_layers.length)
    ? brainData.hidden_layers
    : (brainData && brainData.hidden_layer ? [brainData.hidden_layer] : []);
  const lastHidden = hidLayers.length ? hidLayers[hidLayers.length - 1] : null;
  const neuronCount = lastHidden ? lastHidden.count : 64;
  const actName = lastHidden && lastHidden.activation ? lastHidden.activation : 'RELU';

  step1TitleEl.textContent = `${neuronCount} Denk-Neuronen (${actName}) vergeben Punkte`;

  if (!brainData || brainData.empty || !brainData.outputs || !lastHidden) {
    headerTitle.textContent = '📐 Wie kommt die Wahrscheinlichkeit zustande? (Live-Rechnung)';
    chipsBox.innerHTML = '';
    f1.textContent = `Zeichne links etwas ins Feld, um die Live-Abstimmung der ${neuronCount} Neuronen zu sehen.`;
    f2.textContent = 'exp(z − z_max) wird live berechnet...';
    f3.textContent = 'Wahrscheinlichkeit = Eigenes Gewicht / Gesamtsumme';
    tbody.innerHTML = '';
    return;
  }

  const focusIdx = getInspectedOutputIndex();
  const targetOut = brainData.outputs[focusIdx];
  const kandidatName = currentMode === 'zahlen'
    ? `Zahl ${targetOut.label}`
    : `${targetOut.label} (${targetOut.symbol})`;

  headerTitle.textContent = `📐 Live-Wahrscheinlichkeits-Rechnung für: ${kandidatName}`;

  // Gather all final hidden layer neuron contributions for focusIdx
  const rawHidden = lastHidden.raw;
  const contribs = brainData.synapses.contributions;
  const weights = brainData.synapses.weights;

  const items = [];
  for (let j = 0; j < rawHidden.length; j++) {
    items.push({
      j,
      act: rawHidden[j],
      w: weights[j][focusIdx],
      c: contribs[j][focusIdx]
    });
  }

  const topPos = [...items].filter(x => x.c > 0.01).sort((a, b) => b.c - a.c).slice(0, 3);
  const topNeg = [...items].filter(x => x.c < -0.01).sort((a, b) => a.c - b.c).slice(0, 2);

  chipsBox.innerHTML = '';
  topPos.forEach(it => {
    const d = document.createElement('div');
    d.className = 'chip-line chip-pos';
    const wStr = `${it.w >= 0 ? '+' : ''}${it.w.toFixed(2)}`;
    d.innerHTML = `<span>Neuron #${it.j}: ${it.act.toFixed(2)} × (${wStr})</span><strong>+${it.c.toFixed(2)} Pkt.</strong>`;
    chipsBox.appendChild(d);
  });
  topNeg.forEach(it => {
    const d = document.createElement('div');
    d.className = 'chip-line chip-neg';
    const wStr = `${it.w >= 0 ? '+' : ''}${it.w.toFixed(2)}`;
    d.innerHTML = `<span>Neuron #${it.j}: ${it.act.toFixed(2)} × (${wStr})</span><strong>${it.c.toFixed(2)} Pkt.</strong>`;
    chipsBox.appendChild(d);
  });

  const signBias = targetOut.bias >= 0 ? '+' : '−';
  f1.innerHTML =
    `Summe aller ${rawHidden.length} Neuronen: <b>${targetOut.neuron_sum >= 0 ? '+' : ''}${targetOut.neuron_sum.toFixed(2)}</b><br>` +
    `${signBias} Grundwert (Bias): <b>${Math.abs(targetOut.bias).toFixed(2)}</b><br>` +
    `──────────────────────────<br>` +
    `= Roh-Punkte (Logit z): <b style="color:#38bdf8">${targetOut.logit >= 0 ? '+' : ''}${targetOut.logit.toFixed(2)}</b>`;

  const zMax = brainData.softmax_meta.z_max;
  const diffZ = targetOut.logit - zMax;
  f2.innerHTML =
    `Höchster Rohwert z<sub>max</sub> = <b>${zMax >= 0 ? '+' : ''}${zMax.toFixed(2)}</b><br>` +
    `Abstand: z − z<sub>max</sub> = ${targetOut.logit.toFixed(2)} − (${zMax.toFixed(2)}) = <b>${diffZ.toFixed(2)}</b><br>` +
    `──────────────────────────<br>` +
    `Softmax-Gewicht e<sup>${diffZ.toFixed(2)}</sup> = <b style="color:#c084fc">${targetOut.exp_rel.toFixed(4)}</b>`;

  const expSum = brainData.softmax_meta.exp_sum;
  f3.innerHTML =
    `Eigenes Gewicht: <b>${targetOut.exp_rel.toFixed(4)}</b><br>` +
    `÷ Summe aller Kandidaten: <b>${expSum.toFixed(4)}</b><br>` +
    `──────────────────────────<br>` +
    `= Wahrscheinlichkeit: <b style="color:#34d399; font-size:13px;">${targetOut.prob.toFixed(2)} %</b>`;

  // Populate complete candidate table
  tbody.innerHTML = '';
  brainData.outputs.forEach(o => {
    const tr = document.createElement('tr');
    tr.className = 'prob-row' + (o.index === focusIdx ? ' focused' : '');
    tr.onclick = () => setOutputFocus(o.index);

    const nameStr = currentMode === 'zahlen'
      ? `Zahl ${o.label}`
      : `${o.label} ${o.symbol}`;
    const bSign = o.bias >= 0 ? '+' : '−';
    const isWinner = brainData.prediction && brainData.prediction.index === o.index;

    tr.innerHTML = `
      <td style="font-family: inherit; font-weight: ${isWinner ? '700' : '500'}; color: ${isWinner ? '#38bdf8' : '#e2e8f0'};">
        ${o.index === focusIdx ? '▶ ' : ''}${nameStr}
      </td>
      <td>${o.neuron_sum >= 0 ? '+' : ''}${o.neuron_sum.toFixed(2)} ${bSign} ${Math.abs(o.bias).toFixed(2)}</td>
      <td style="color:${o.logit >= 0 ? '#7dd3fc' : '#94a3b8'}; font-weight:600;">
        ${o.logit >= 0 ? '+' : ''}${o.logit.toFixed(2)}
      </td>
      <td>${o.exp_rel.toFixed(4)}</td>
      <td class="bar-cell">
        <div class="mini-bar-bg">
          <div class="mini-bar-fill" style="width:${Math.min(100, o.prob).toFixed(1)}%;"></div>
        </div>
      </td>
      <td style="text-align:right; font-weight:700; color:${isWinner ? '#34d399' : '#cbd5e1'};">
        ${o.prob.toFixed(2)} %
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// =========================================================================
// 60-FPS INTERACTIVE NEURAL NETWORK BRAIN RENDERER (HTML5 HIGH-DPI CANVAS)
// =========================================================================
const brainCanvas = document.getElementById('brainCanvas');
const brainCtx = brainCanvas.getContext('2d');
const brainWrap = document.getElementById('brainWrap');
const tooltipEl = document.getElementById('nodeTooltip');
const tooltipHtmlEl = document.getElementById('nodeTooltipHtml');
const tooltipMapCanvas = document.getElementById('tooltipMapCanvas');
const tooltipMapCtx = tooltipMapCanvas.getContext('2d');

function resizeBrainCanvas() {
  const rect = brainWrap.getBoundingClientRect();
  if (rect.width < 10) return;
  const dpr = window.devicePixelRatio || 1;
  brainCanvas.width = Math.round(rect.width * dpr);
  brainCanvas.height = Math.round(rect.height * dpr);
  brainCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resizeBrainCanvas);

brainCanvas.addEventListener('mousemove', e => {
  const rect = brainCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  let best = null;
  let bestDist = 14;
  for (let i = 0; i < hitNodes.length; i++) {
    const n = hitNodes[i];
    const d = Math.hypot(mx - n.x, my - n.y);
    if (d <= Math.max(n.r + 4, bestDist) && d < bestDist + 4) {
      best = n;
      bestDist = d;
    }
  }
  hoveredNode = best;
  updateBrainTooltip(mx, my, rect.width, rect.height);
});

brainCanvas.addEventListener('mouseleave', () => {
  hoveredNode = null;
  tooltipEl.classList.add('hidden');
});

brainCanvas.addEventListener('click', () => {
  if (hoveredNode && hoveredNode.colType === 'output') {
    setOutputFocus(hoveredNode.nodeIdx);
  }
});

function paintMiniHeatmap(grid8, rgbMode) {
  tooltipMapCanvas.classList.remove('hidden');
  const cell = 9; // 8 * 9 = 72px
  tooltipMapCtx.clearRect(0, 0, 72, 72);
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const v = grid8[r][c] / 100;
      const red = rgbMode === 'gold' ? Math.round(20 + v * 231) : Math.round(15 + v * 45);
      const green = rgbMode === 'gold' ? Math.round(24 + v * 167) : Math.round(23 + v * 195);
      const blue = rgbMode === 'gold' ? Math.round(38 + v * 20) : Math.round(42 + v * 213);
      tooltipMapCtx.fillStyle = `rgb(${red},${green},${blue})`;
      tooltipMapCtx.fillRect(c * cell, r * cell, cell, cell);
    }
  }
}

function updateBrainTooltip(mx, my, wrapW, wrapH) {
  if (!hoveredNode || !brainData) {
    tooltipEl.classList.add('hidden');
    return;
  }

  tooltipEl.classList.remove('hidden');
  tooltipMapCanvas.classList.add('hidden');

  const focusIdx = getInspectedOutputIndex();
  const targetOut = brainData.outputs ? brainData.outputs[focusIdx] : null;
  const targetLabel = targetOut ? targetOut.label : '?';
  const hidLayers = (brainData.hidden_layers && brainData.hidden_layers.length)
    ? brainData.hidden_layers
    : (brainData.hidden_layer ? [brainData.hidden_layer] : []);

  if (hoveredNode.colType === 'conv') {
    const layer = brainData.conv_layers[hoveredNode.layerIdx];
    if (!layer) return;
    const actPct = (hoveredNode.val * 100).toFixed(1);
    const rawMean = layer.raw_means ? layer.raw_means[hoveredNode.nodeIdx] : 0;
    tooltipHtmlEl.innerHTML =
      `<strong style="color:#38bdf8">${layer.name} • Knoten #${hoveredNode.nodeIdx}</strong><br>` +
      `Relative Feuerstärke: <b>${actPct} %</b> (Ø ${rawMean})<br>` +
      `<span style="color:#94a3b8">Live-Filterkarte dieses Knotens:</span>`;

    if (layer.mini_maps && layer.mini_maps[hoveredNode.nodeIdx]) {
      paintMiniHeatmap(layer.mini_maps[hoveredNode.nodeIdx], 'cyan');
    }
  } else if (hoveredNode.colType === 'hidden') {
    const hIdx = hoveredNode.layerIdx || 0;
    const hLayer = hidLayers[hIdx];
    if (!hLayer) return;
    const j = hoveredNode.nodeIdx;
    const rawAct = hLayer.raw[j] || 0;
    const actFn = hLayer.activation || 'RELU';
    const isLastHidden = hIdx === hidLayers.length - 1;
    const titlePrefix = hidLayers.length > 1 ? `Denkschicht ${hIdx + 1}` : 'Denk-Neuron';

    if (isLastHidden) {
      const w = brainData.synapses.weights[j][focusIdx];
      const contrib = brainData.synapses.contributions[j][focusIdx];
      const col = contrib >= 0 ? '#34d399' : '#fb7185';
      const verdict = Math.abs(contrib) < 0.005
        ? 'Neutral / inaktiv'
        : (contrib > 0 ? `Spricht FÜR „${targetLabel}“` : `Spricht GEGEN „${targetLabel}“`);

      tooltipHtmlEl.innerHTML =
        `<strong style="color:#fbbf24">${titlePrefix} • #${j} (${actFn})</strong><br>` +
        `Neuron-Aktivierung: <b>${rawAct >= 0 ? '+' : ''}${rawAct.toFixed(3)}</b><br>` +
        `Gewicht zu [${targetLabel}]: <b>${w >= 0 ? '+' : ''}${w.toFixed(3)}</b><br>` +
        `Punkte-Beitrag: <b style="color:${col}">${contrib >= 0 ? '+' : ''}${contrib.toFixed(3)} Pkt.</b><br>` +
        `<span style="color:${col}">${verdict}</span>`;
    } else {
      tooltipHtmlEl.innerHTML =
        `<strong style="color:#f59e0b">${titlePrefix} • #${j} (${actFn})</strong><br>` +
        `Neuron-Aktivierung: <b>${rawAct >= 0 ? '+' : ''}${rawAct.toFixed(3)}</b><br>` +
        `Relative Stärke: <b>${(hoveredNode.val * 100).toFixed(1)} %</b><br>` +
        `<span style="color:#fcd34d">Leitet kombinierte Merkmale weiter an Denkschicht ${hIdx + 2}!</span>`;
    }

    if (hLayer.mini_maps && hLayer.mini_maps[j]) {
      paintMiniHeatmap(hLayer.mini_maps[j], 'gold');
    }
  } else if (hoveredNode.colType === 'output') {
    const o = brainData.outputs[hoveredNode.nodeIdx];
    tooltipHtmlEl.innerHTML =
      `<strong style="color:#34d399">Ausgabeknoten: ${o.label}</strong><br>` +
      `Roh-Punkte (Logit z): <b>${o.logit >= 0 ? '+' : ''}${o.logit.toFixed(2)}</b><br>` +
      `Wahrscheinlichkeit: <b>${o.prob.toFixed(2)} %</b><br>` +
      `<span style="color:#7dd3fc">Klicken, um alle Synapsen für „${o.label}“ zu beleuchten!</span>`;
  }

  const left = Math.min(wrapW - 260, Math.max(10, mx + 16));
  const top = Math.min(wrapH - 165, Math.max(10, my - 20));
  tooltipEl.style.left = left + 'px';
  tooltipEl.style.top = top + 'px';
}

function computeNodeGrid(cnt, colGap) {
  let subCols = 4;
  let baseR = 5.5;
  let baseSpacing = 15.5;

  if (cnt <= 10) {
    subCols = 1; baseR = 8.0; baseSpacing = 18;
  } else if (cnt <= 20) {
    subCols = 2; baseR = 7.0; baseSpacing = 18.5;
  } else if (cnt <= 32) {
    subCols = 2; baseR = 6.0; baseSpacing = 17.0;
  } else if (cnt <= 48) {
    subCols = 3; baseR = 5.7; baseSpacing = 16.0;
  } else if (cnt <= 64) {
    subCols = 4; baseR = 5.5; baseSpacing = 15.5;
  } else if (cnt <= 96) {
    subCols = 6; baseR = 4.6; baseSpacing = 12.2;
  } else {
    subCols = 8; baseR = 4.0; baseSpacing = 10.2;
  }

  const maxInnerW = Math.max(24, colGap * 0.62 - 16);
  const spacingX = subCols > 1 ? Math.min(baseSpacing, maxInnerW / (subCols - 1)) : 0;
  const rNode = subCols > 1 ? Math.min(baseR, spacingX * 0.40) : baseR;
  const rows = Math.ceil(cnt / subCols);
  const capW = (subCols > 1 ? (subCols - 1) * spacingX : 0) + rNode * 2 + 16;
  return { subCols, rows, spacingX, rNode, capW };
}

function renderBrainLoop() {
  requestAnimationFrame(renderBrainLoop);
  if (currentView !== 'zahlen' && currentView !== 'smiley') return;

  const w = brainCanvas.clientWidth;
  const h = brainCanvas.clientHeight;
  if (w < 20 || h < 20) return;

  pulsePhase = (pulsePhase + 0.018) % 1.0;
  const ctx = brainCtx;
  ctx.clearRect(0, 0, w, h);

  if (!brainData) return;

  // 0. Smoothly interpolate activations for ANY number of Conv & Hidden layers
  const convLayers = brainData.conv_layers || [];
  for (let li = 0; li < convLayers.length; li++) {
    const target = convLayers[li].activations;
    if (!smoothConv[li] || smoothConv[li].length !== target.length) {
      smoothConv[li] = new Float32Array(target.length);
    }
    for (let i = 0; i < target.length; i++) {
      smoothConv[li][i] += (target[i] - smoothConv[li][i]) * 0.22;
    }
  }

  const hidLayers = (brainData.hidden_layers && brainData.hidden_layers.length)
    ? brainData.hidden_layers
    : (brainData.hidden_layer ? [brainData.hidden_layer] : []);

  for (let di = 0; di < hidLayers.length; di++) {
    const targetHid = hidLayers[di].norm || [];
    if (!smoothHiddenLayers[di] || smoothHiddenLayers[di].length !== targetHid.length) {
      smoothHiddenLayers[di] = new Float32Array(targetHid.length);
    }
    for (let j = 0; j < targetHid.length; j++) {
      smoothHiddenLayers[di][j] += (targetHid[j] - smoothHiddenLayers[di][j]) * 0.22;
    }
  }

  const outs = brainData.outputs || [];
  if (smoothOutputs.length !== outs.length) {
    smoothOutputs = new Float32Array(outs.length);
  }
  for (let k = 0; k < outs.length; k++) {
    const targetP = (outs[k].prob || 0) / 100.0;
    smoothOutputs[k] += (targetP - smoothOutputs[k]) * 0.22;
  }

  // Dynamic Columns: [InputGrid, ...ConvLayers (0..3), ...HiddenLayers (1..2+), Outputs]
  const numCols = 1 + convLayers.length + hidLayers.length + 1;
  const leftPad = 66;
  const rightPad = currentMode === 'zahlen' ? 98 : 148;
  const usableW = Math.max(240, w - leftPad - rightPad);
  const colGap = usableW / Math.max(1, numCols - 1);
  const colX = [];
  for (let c = 0; c < numCols; c++) {
    colX.push(leftPad + colGap * c);
  }

  const topPad = 52;
  const botPad = 18;
  const usableH = h - topPad - botPad;

  hitNodes = [];

  // 1. Compute Coordinates of Every Layer's Nodes dynamically
  const inputBoxSize = 76;
  const inputCenter = { x: colX[0], y: topPad + usableH / 2 };
  const inputX0 = inputCenter.x - inputBoxSize / 2;
  const inputY0 = inputCenter.y - inputBoxSize / 2;

  const convCoords = [];
  const layerCapsules = [];

  for (let li = 0; li < convLayers.length; li++) {
    const cnt = convLayers[li].count;
    const cx = colX[1 + li];
    const g = computeNodeGrid(cnt, colGap);
    const coords = [];

    for (let i = 0; i < cnt; i++) {
      const sc = i % g.subCols;
      const row = Math.floor(i / g.subCols);
      const nx = cx + (sc - (g.subCols - 1) / 2) * g.spacingX;
      const ny = topPad + ((row + 0.5) / g.rows) * usableH;
      const val = smoothConv[li][i] || 0;
      const nodeObj = { colType: 'conv', layerIdx: li, nodeIdx: i, x: nx, y: ny, r: g.rNode, val };
      coords.push(nodeObj);
      hitNodes.push(nodeObj);
    }
    convCoords.push(coords);
    layerCapsules.push({
      cx,
      w: g.capW,
      borderColor: 'rgba(56, 189, 248, 0.24)'
    });
  }

  // Hidden Dense Layer(s) (1 or 2+ columns!)
  const hidBaseCol = 1 + convLayers.length;
  const hidLayerCoords = [];
  for (let di = 0; di < hidLayers.length; di++) {
    const cnt = hidLayers[di].count;
    const cx = colX[hidBaseCol + di];
    const g = computeNodeGrid(cnt, colGap);
    const coords = [];

    for (let j = 0; j < cnt; j++) {
      const sc = j % g.subCols;
      const row = Math.floor(j / g.subCols);
      const nx = cx + (sc - (g.subCols - 1) / 2) * g.spacingX;
      const ny = topPad + ((row + 0.5) / g.rows) * usableH;
      const val = smoothHiddenLayers[di][j] || 0;
      const nodeObj = { colType: 'hidden', layerIdx: di, nodeIdx: j, x: nx, y: ny, r: g.rNode, val };
      coords.push(nodeObj);
      hitNodes.push(nodeObj);
    }
    hidLayerCoords.push(coords);
    layerCapsules.push({
      cx,
      w: g.capW,
      borderColor: di === 0 ? 'rgba(251, 191, 36, 0.26)' : 'rgba(245, 158, 11, 0.30)'
    });
  }

  const firstHidCoords = hidLayerCoords[0] || [];
  const lastHidCoords = hidLayerCoords[hidLayerCoords.length - 1] || [];

  // Output Layer (10 digits or 5 smiley moods)
  const outColIdx = numCols - 1;
  const outCoords = [];
  {
    const cnt = outs.length;
    const cx = colX[outColIdx];
    const rNode = cnt <= 5 ? 17.5 : 13.5;
    for (let k = 0; k < cnt; k++) {
      const ny = topPad + ((k + 0.5) / cnt) * usableH;
      const val = smoothOutputs[k] || 0;
      const nodeObj = { colType: 'output', layerIdx: 0, nodeIdx: k, x: cx, y: ny, r: rNode, val };
      outCoords.push(nodeObj);
      hitNodes.push(nodeObj);
    }
  }

  const focusIdx = getInspectedOutputIndex();
  const isEmpty = brainData.empty;

  // --- 2. Draw Subtle Layer Capsules & Dynamic Column Headers ---
  for (let c = 0; c < layerCapsules.length; c++) {
    const cap = layerCapsules[c];
    ctx.fillStyle = 'rgba(15, 23, 42, 0.55)';
    ctx.strokeStyle = cap.borderColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(cap.cx - cap.w / 2, topPad - 6, cap.w, usableH + 12, 12);
    ctx.fill();
    ctx.stroke();
  }

  ctx.textAlign = 'center';
  ctx.font = '700 12px "Segoe UI", sans-serif';
  ctx.fillStyle = '#e2e8f0';
  ctx.fillText('Eingabe', colX[0], 19);
  ctx.font = '10.5px Consolas, monospace';
  ctx.fillStyle = '#94a3b8';
  ctx.fillText(currentMode === 'zahlen' ? '28×28 px' : '48×48 px', colX[0], 34);

  for (let li = 0; li < convLayers.length; li++) {
    ctx.font = '700 12px "Segoe UI", sans-serif';
    ctx.fillStyle = '#7dd3fc';
    ctx.fillText(`Filter-Schicht ${li + 1}`, colX[1 + li], 19);
    ctx.font = '10.5px Consolas, monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`${convLayers[li].count} Filter`, colX[1 + li], 34);
  }

  for (let di = 0; di < hidLayers.length; di++) {
    const hInfo = hidLayers[di];
    const hTitle = hidLayers.length === 1 ? 'Denkschicht' : `Denkschicht ${di + 1}`;
    const actTag = hInfo.activation || 'RELU';
    ctx.font = '700 12px "Segoe UI", sans-serif';
    ctx.fillStyle = di === 0 ? '#fcd34d' : '#fb923c';
    ctx.fillText(hTitle, colX[hidBaseCol + di], 19);
    ctx.font = '10.5px Consolas, monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(`${hInfo.count} • ${actTag}`, colX[hidBaseCol + di], 34);
  }

  ctx.font = '700 12px "Segoe UI", sans-serif';
  ctx.fillStyle = '#34d399';
  ctx.fillText('Ausgabe (Softmax)', colX[outColIdx], 19);
  ctx.font = '10.5px Consolas, monospace';
  ctx.fillStyle = '#cbd5e1';
  ctx.fillText(`${outs.length} Klassen`, colX[outColIdx], 34);

  // --- 2b. Subtle Structural Baseline Neural Web across ALL Active Layers ---
  if (convCoords.length > 0) {
    const c1 = convCoords[0];
    const stepC1 = Math.max(1, Math.floor(c1.length / 16));
    for (let i = 0; i < c1.length; i += stepC1) {
      drawBezierSynapse(
        ctx,
        inputCenter.x + inputBoxSize / 2,
        inputCenter.y + ((i / c1.length) - 0.5) * 52,
        c1[i].x,
        c1[i].y,
        'rgba(148, 163, 184, 0.07)',
        0.7,
        -1,
        ''
      );
    }
    for (let li = 0; li < convCoords.length - 1; li++) {
      const sArr = convCoords[li];
      const dArr = convCoords[li + 1];
      const stepD = Math.max(1, Math.floor(dArr.length / 20));
      for (let d = 0; d < dArr.length; d += stepD) {
        const s = sArr[(d * 5) % sArr.length];
        drawBezierSynapse(ctx, s.x, s.y, dArr[d].x, dArr[d].y, 'rgba(148, 163, 184, 0.06)', 0.7, -1, '');
      }
    }
    if (firstHidCoords.length > 0) {
      const lastC = convCoords[convCoords.length - 1];
      const stepH = Math.max(1, Math.floor(firstHidCoords.length / 20));
      for (let j = 0; j < firstHidCoords.length; j += stepH) {
        const s = lastC[(j * 7) % lastC.length];
        drawBezierSynapse(ctx, s.x, s.y, firstHidCoords[j].x, firstHidCoords[j].y, 'rgba(148, 163, 184, 0.06)', 0.7, -1, '');
      }
    }
  } else if (firstHidCoords.length > 0) {
    // Pure MLP structural baseline web: Input Box -> Denkschicht 1
    const stepH = Math.max(1, Math.floor(firstHidCoords.length / 18));
    for (let j = 0; j < firstHidCoords.length; j += stepH) {
      drawBezierSynapse(
        ctx,
        inputCenter.x + inputBoxSize / 2,
        inputCenter.y + ((j / firstHidCoords.length) - 0.5) * 56,
        firstHidCoords[j].x,
        firstHidCoords[j].y,
        'rgba(148, 163, 184, 0.07)',
        0.7,
        -1,
        ''
      );
    }
  }

  for (let di = 0; di < hidLayerCoords.length - 1; di++) {
    const sH = hidLayerCoords[di];
    const dH = hidLayerCoords[di + 1];
    const stepD = Math.max(1, Math.floor(dH.length / 20));
    for (let j = 0; j < dH.length; j += stepD) {
      const s = sH[(j * 5) % sH.length];
      drawBezierSynapse(ctx, s.x, s.y, dH[j].x, dH[j].y, 'rgba(148, 163, 184, 0.06)', 0.7, -1, '');
    }
  }

  if (lastHidCoords.length > 0 && outCoords.length > 0) {
    const stepO = Math.max(1, Math.floor(lastHidCoords.length / 16));
    for (let j = 0; j < lastHidCoords.length; j += stepO) {
      const o = outCoords[j % outCoords.length];
      drawBezierSynapse(ctx, lastHidCoords[j].x, lastHidCoords[j].y, o.x - o.r, o.y, 'rgba(148, 163, 184, 0.06)', 0.7, -1, '');
    }
  }

  // --- 3. Draw Active Live Synapses & Animated Signal Pulses ---
  if (!isEmpty) {
    // A1) CNN Mode: Input Box -> Conv1 active filters
    if (convCoords.length > 0) {
      const c1 = convCoords[0];
      for (let i = 0; i < c1.length; i++) {
        const act = c1[i].val;
        if (act < 0.20) continue;
        const alpha = Math.min(0.65, 0.12 + act * 0.45);
        drawBezierSynapse(
          ctx,
          inputCenter.x + inputBoxSize / 2,
          inputCenter.y + ((i / c1.length) - 0.5) * 52,
          c1[i].x,
          c1[i].y,
          `rgba(56, 189, 248, ${alpha.toFixed(3)})`,
          0.9 + act * 1.1,
          act > 0.48 ? (pulsePhase + i * 0.07) % 1 : -1,
          '#7dd3fc'
        );
      }
    } else if (firstHidCoords.length > 0 && brainData.synapses.input_to_hidden) {
      // A2) Pure MLP Mode: Exact glowing 14x14 drawn pixels -> Denkschicht 1!
      const in2h = brainData.synapses.input_to_hidden;
      const cell14 = inputBoxSize / 14;
      for (let j = 0; j < firstHidCoords.length; j++) {
        const hNode = firstHidCoords[j];
        const isHovH0 = Boolean(
          hoveredNode && hoveredNode.colType === 'hidden' && hoveredNode.layerIdx === 0 && hoveredNode.nodeIdx === j
        );
        if (hNode.val < 0.16 && !isHovH0) continue;
        const incoming = in2h[j] || [];
        const limit = isHovH0 ? incoming.length : 2;
        for (let m = 0; m < Math.min(limit, incoming.length); m++) {
          const [r14, c14, signedFlow] = incoming[m];
          const mag = Math.abs(signedFlow) * Math.max(0.25, hNode.val);
          if (mag < 0.05 && !isHovH0) continue;
          const px = inputX0 + (c14 + 0.5) * cell14;
          const py = inputY0 + (r14 + 0.5) * cell14;
          const alpha = isHovH0 ? 0.88 : Math.min(0.62, 0.14 + mag * 0.85);
          const col = signedFlow >= 0
            ? `rgba(56, 189, 248, ${alpha.toFixed(3)})`
            : `rgba(251, 113, 133, ${alpha.toFixed(3)})`;
          drawBezierSynapse(
            ctx,
            px,
            py,
            hNode.x,
            hNode.y,
            col,
            isHovH0 ? 2.2 : (0.9 + mag * 1.3),
            mag > 0.14 ? (pulsePhase + j * 0.04) % 1 : -1,
            signedFlow >= 0 ? '#fde047' : '#fb7185'
          );
        }
      }
    }

    // B) Conv[li] -> Conv[li+1] using real kernel coupling links
    for (let li = 0; li < convCoords.length - 1; li++) {
      const key = `conv${li + 1}_to_conv${li + 2}`;
      const links = brainData.synapses[key];
      const srcNodes = convCoords[li];
      const dstNodes = convCoords[li + 1];
      if (!links) continue;

      for (let d = 0; d < dstNodes.length; d++) {
        const dstNode = dstNodes[d];
        const dstAct = dstNode.val;
        const isHovered = Boolean(
          hoveredNode &&
          hoveredNode.colType === 'conv' &&
          hoveredNode.layerIdx === li + 1 &&
          hoveredNode.nodeIdx === d
        );
        if (dstAct < 0.22 && !isHovered) continue;

        const incoming = links[d] || [];
        const limit = isHovered ? incoming.length : 2;
        for (let m = 0; m < Math.min(limit, incoming.length); m++) {
          const [sIdx, wRel] = incoming[m];
          const srcNode = srcNodes[sIdx];
          if (!srcNode) continue;
          const flow = srcNode.val * dstAct * wRel;
          if (flow < 0.065 && !isHovered) continue;
          const alpha = isHovered ? 0.85 : Math.min(0.58, 0.12 + flow * 0.65);
          drawBezierSynapse(
            ctx,
            srcNode.x,
            srcNode.y,
            dstNode.x,
            dstNode.y,
            `rgba(56, 189, 248, ${alpha.toFixed(3)})`,
            isHovered ? 2.2 : (0.9 + flow * 1.2),
            flow > 0.22 ? (pulsePhase + d * 0.05) % 1 : -1,
            '#7dd3fc'
          );
        }
      }
    }

    // C) Last Conv Layer -> First Hidden Layer (real spatial-filter flow!)
    if (convCoords.length > 0 && firstHidCoords.length > 0 && brainData.synapses.conv_to_hidden) {
      const lastConv = convCoords[convCoords.length - 1];
      const c2h = brainData.synapses.conv_to_hidden;
      for (let j = 0; j < firstHidCoords.length; j++) {
        const hNode = firstHidCoords[j];
        const isHovHid = Boolean(
          hoveredNode && hoveredNode.colType === 'hidden' && hoveredNode.layerIdx === 0 && hoveredNode.nodeIdx === j
        );
        if (hNode.val < 0.15 && !isHovHid) continue;

        const incoming = c2h[j] || [];
        const limit = isHovHid ? incoming.length : 2;
        for (let m = 0; m < Math.min(limit, incoming.length); m++) {
          const [cIdx, signedFlow] = incoming[m];
          const cNode = lastConv[cIdx];
          if (!cNode) continue;
          const mag = Math.abs(signedFlow) * Math.max(0.25, hNode.val);
          if (mag < 0.045 && !isHovHid) continue;
          const alpha = isHovHid ? 0.88 : Math.min(0.62, 0.12 + mag * 0.85);
          const col = signedFlow >= 0
            ? `rgba(56, 189, 248, ${alpha.toFixed(3)})`
            : `rgba(251, 113, 133, ${alpha.toFixed(3)})`;
          drawBezierSynapse(
            ctx,
            cNode.x,
            cNode.y,
            hNode.x,
            hNode.y,
            col,
            isHovHid ? 2.2 : (0.9 + mag * 1.3),
            mag > 0.14 ? (pulsePhase + j * 0.03) % 1 : -1,
            signedFlow >= 0 ? '#fde047' : '#fb7185'
          );
        }
      }
    }

    // C2) Hidden[di] -> Hidden[di+1] (when 2 Denkschichten are trained!)
    const h2hAll = brainData.synapses.hidden_to_hidden || [];
    for (let di = 0; di < hidLayerCoords.length - 1; di++) {
      const srcHid = hidLayerCoords[di];
      const dstHid = hidLayerCoords[di + 1];
      const links = h2hAll[di];
      if (!links) continue;

      for (let j = 0; j < dstHid.length; j++) {
        const dNode = dstHid[j];
        const isHovDst = Boolean(
          hoveredNode && hoveredNode.colType === 'hidden' && hoveredNode.layerIdx === di + 1 && hoveredNode.nodeIdx === j
        );
        const incoming = links[j] || [];
        const limit = isHovDst ? incoming.length : 2;

        for (let m = 0; m < Math.min(limit, incoming.length); m++) {
          const [sIdx, signedFlow] = incoming[m];
          const sNode = srcHid[sIdx];
          if (!sNode) continue;
          const isHovSrc = Boolean(
            hoveredNode && hoveredNode.colType === 'hidden' && hoveredNode.layerIdx === di && hoveredNode.nodeIdx === sIdx
          );
          if (dNode.val < 0.14 && !isHovDst && !isHovSrc) continue;

          const mag = Math.abs(signedFlow) * Math.max(0.3, dNode.val);
          if (mag < 0.045 && !isHovDst && !isHovSrc) continue;
          const highlight = isHovDst || isHovSrc;
          const alpha = highlight ? 0.90 : Math.min(0.68, 0.15 + mag * 0.85);
          const col = signedFlow >= 0
            ? `rgba(251, 191, 36, ${alpha.toFixed(3)})`
            : `rgba(244, 63, 94, ${alpha.toFixed(3)})`;
          drawBezierSynapse(
            ctx,
            sNode.x,
            sNode.y,
            dNode.x,
            dNode.y,
            col,
            highlight ? 2.3 : (0.95 + mag * 1.4),
            mag > 0.15 ? (pulsePhase + j * 0.04) % 1 : -1,
            signedFlow >= 0 ? '#fde047' : '#fda4af'
          );
        }
      }
    }

    // D) HERO SYNAPSES: Final Denkschicht -> Softmax Output Nodes (dynamic neuron count!)
    const contribs = brainData.synapses.contributions;
    const numLastHid = lastHidCoords.length;
    if (contribs && numLastHid > 0 && contribs.length === numLastHid) {
      let maxAbsC = 0.01;
      for (let j = 0; j < numLastHid; j++) {
        const v = Math.abs(contribs[j][focusIdx] || 0);
        if (v > maxAbsC) maxAbsC = v;
      }

      const lastLayerIdx = hidLayerCoords.length - 1;
      const hovHidIdx = (
        hoveredNode &&
        hoveredNode.colType === 'hidden' &&
        hoveredNode.layerIdx === lastLayerIdx
      ) ? hoveredNode.nodeIdx : null;

      if (hovHidIdx !== null && lastHidCoords[hovHidIdx]) {
        for (let k = 0; k < outCoords.length; k++) {
          const cVal = contribs[hovHidIdx][k];
          const wVal = brainData.synapses.weights[hovHidIdx][k];
          const strength = Math.min(1, Math.abs(cVal) / maxAbsC + Math.abs(wVal) * 0.25);
          if (strength < 0.04) continue;
          const isPos = (Math.abs(cVal) > 1e-4 ? cVal >= 0 : wVal >= 0);
          const rgb = isPos ? '16, 185, 129' : '244, 63, 94';
          drawBezierSynapse(
            ctx,
            lastHidCoords[hovHidIdx].x,
            lastHidCoords[hovHidIdx].y,
            outCoords[k].x - outCoords[k].r,
            outCoords[k].y,
            `rgba(${rgb}, ${(0.3 + strength * 0.68).toFixed(3)})`,
            1.2 + strength * 2.6,
            Math.abs(cVal) > 0.04 ? pulsePhase : -1,
            isPos ? '#6ee7b7' : '#fda4af'
          );
        }
      } else {
        // Subtle background lines to non-focused outputs
        const stride = Math.max(2, Math.floor(numLastHid / 7));
        for (let k = 0; k < outCoords.length; k++) {
          if (k === focusIdx) continue;
          for (let j = k % stride; j < numLastHid; j += stride) {
            const cVal = contribs[j][k];
            if (Math.abs(cVal) > maxAbsC * 0.28) {
              const rgb = cVal >= 0 ? '56, 189, 248' : '244, 63, 94';
              drawBezierSynapse(
                ctx,
                lastHidCoords[j].x,
                lastHidCoords[j].y,
                outCoords[k].x - outCoords[k].r,
                outCoords[k].y,
                `rgba(${rgb}, 0.10)`,
                0.8,
                -1,
                ''
              );
            }
          }
        }

        // Vivid spotlight excitatory (green) & inhibitory (red) synapses into focusIdx!
        const targetNode = outCoords[focusIdx];
        if (targetNode) {
          for (let j = 0; j < numLastHid; j++) {
            const cVal = contribs[j][focusIdx];
            const rel = Math.abs(cVal) / maxAbsC;
            if (rel < 0.06) continue;

            const isPos = cVal >= 0;
            const rgb = isPos ? '16, 185, 129' : '244, 63, 94';
            const alpha = Math.min(0.92, 0.20 + rel * 0.75);
            const width = 0.9 + rel * 2.9;
            const showPulse = rel > 0.26 ? (pulsePhase + j * 0.04) % 1 : -1;

            drawBezierSynapse(
              ctx,
              lastHidCoords[j].x,
              lastHidCoords[j].y,
              targetNode.x - targetNode.r,
              targetNode.y,
              `rgba(${rgb}, ${alpha.toFixed(3)})`,
              width,
              showPulse,
              isPos ? '#6ee7b7' : '#fda4af'
            );
          }
        }
      }
    }
  }

  // --- 4. Draw Input 14x14 Live Pixel Matrix ---
  {
    ctx.fillStyle = '#020617';
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.65)';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.roundRect(inputX0 - 4, inputY0 - 4, inputBoxSize + 8, inputBoxSize + 8, 10);
    ctx.fill();
    ctx.stroke();

    const grid = brainData.input_grid;
    if (grid && grid.length === 14) {
      const cell = inputBoxSize / 14;
      for (let r = 0; r < 14; r++) {
        for (let c = 0; c < 14; c++) {
          const v = grid[r][c] / 100;
          if (v > 0.04) {
            ctx.fillStyle = `rgba(125, 211, 252, ${Math.min(1, 0.25 + v * 0.9).toFixed(2)})`;
            ctx.fillRect(inputX0 + c * cell, inputY0 + r * cell, cell - 0.4, cell - 0.4);
          } else {
            ctx.fillStyle = 'rgba(30, 41, 59, 0.35)';
            ctx.fillRect(inputX0 + c * cell, inputY0 + r * cell, cell - 0.8, cell - 0.8);
          }
        }
      }
    }
  }

  // --- 5. Draw Conv Filter Nodes (High Contrast & Glow!) ---
  for (let li = 0; li < convCoords.length; li++) {
    const nodes = convCoords[li];
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const isHov = hoveredNode === n;
      const v = isEmpty ? 0 : n.val;

      if (v > 0.35 || isHov) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + 4.0, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(56, 189, 248, ${(isHov ? 0.55 : v * 0.38).toFixed(2)})`;
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(n.x, n.y, isHov ? n.r + 2.0 : n.r, 0, Math.PI * 2);
      const rC = Math.round(32 + v * 175);
      const gC = Math.round(48 + v * 192);
      const bC = Math.round(76 + v * 179);
      ctx.fillStyle = `rgb(${rC},${gC},${bC})`;
      ctx.fill();
      ctx.strokeStyle = isHov
        ? '#ffffff'
        : `rgba(125, 211, 252, ${(0.45 + v * 0.55).toFixed(2)})`;
      ctx.lineWidth = isHov ? 2.0 : 1.15;
      ctx.stroke();
    }
  }

  // --- 6. Draw All Hidden Thinking Layer(s) (High Contrast Amber/Gold & Warm Orange!) ---
  for (let di = 0; di < hidLayerCoords.length; di++) {
    const nodes = hidLayerCoords[di];
    for (let j = 0; j < nodes.length; j++) {
      const n = nodes[j];
      const isHov = hoveredNode === n;
      const v = isEmpty ? 0 : n.val;

      if (v > 0.25 || isHov) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + 4.4, 0, Math.PI * 2);
        ctx.fillStyle = di === 0
          ? `rgba(251, 191, 36, ${(isHov ? 0.58 : v * 0.42).toFixed(2)})`
          : `rgba(251, 146, 60, ${(isHov ? 0.58 : v * 0.44).toFixed(2)})`;
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(n.x, n.y, isHov ? n.r + 2.0 : n.r, 0, Math.PI * 2);
      const rC = Math.round(36 + v * 219);
      const gC = di === 0 ? Math.round(46 + v * 192) : Math.round(44 + v * 158);
      const bC = di === 0 ? Math.round(72 + v * 95) : Math.round(74 + v * 70);
      ctx.fillStyle = `rgb(${rC},${gC},${bC})`;
      ctx.fill();
      ctx.strokeStyle = isHov
        ? '#ffffff'
        : (v > 0.05
            ? (di === 0 ? `rgba(253, 224, 71, ${(0.55 + v * 0.45).toFixed(2)})` : `rgba(253, 186, 116, ${(0.55 + v * 0.45).toFixed(2)})`)
            : 'rgba(148, 163, 184, 0.48)');
      ctx.lineWidth = isHov ? 2.0 : 1.2;
      ctx.stroke();
    }
  }

  // --- 7. Draw Output Softmax Nodes with Probability Rings & Labels ---
  for (let k = 0; k < outCoords.length; k++) {
    const n = outCoords[k];
    const o = outs[k];
    const isFocus = k === focusIdx;
    const isWinner = !isEmpty && brainData.prediction && brainData.prediction.index === k;
    const isHov = hoveredNode === n;
    const prob01 = isEmpty ? 0 : n.val;

    if (isWinner || isFocus || isHov) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + 6, 0, Math.PI * 2);
      ctx.fillStyle = isWinner
        ? 'rgba(16, 185, 129, 0.30)'
        : 'rgba(56, 189, 248, 0.25)';
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = isWinner
      ? 'rgba(16, 185, 129, 0.35)'
      : (isFocus ? 'rgba(56, 189, 248, 0.25)' : '#1e293b');
    ctx.fill();
    ctx.strokeStyle = isWinner ? '#34d399' : (isFocus ? '#38bdf8' : 'rgba(148, 163, 184, 0.55)');
    ctx.lineWidth = (isFocus || isWinner) ? 2.2 : 1.4;
    ctx.stroke();

    if (prob01 > 0.005) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * prob01);
      ctx.strokeStyle = isWinner ? '#34d399' : '#38bdf8';
      ctx.lineWidth = 3.4;
      ctx.stroke();
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = currentMode === 'zahlen' ? 'bold 12.5px "Segoe UI", sans-serif' : 'bold 11px Consolas, monospace';
    ctx.fillStyle = isWinner ? '#ffffff' : '#f1f5f9';
    ctx.fillText(o.symbol, n.x, n.y + 0.5);

    ctx.textAlign = 'left';
    const pctText = isEmpty ? '0.0 %' : `${o.prob.toFixed(1)} %`;
    ctx.font = isWinner ? 'bold 12px Consolas, monospace' : '11px Consolas, monospace';
    ctx.fillStyle = isWinner ? '#34d399' : (isFocus ? '#7dd3fc' : '#cbd5e1');

    if (currentMode === 'zahlen') {
      ctx.fillText(pctText, n.x + n.r + 10, n.y + 0.5);
    } else {
      ctx.font = isWinner ? 'bold 11px "Segoe UI", sans-serif' : '10.5px "Segoe UI", sans-serif';
      ctx.fillText(o.label, n.x + n.r + 10, n.y - 6);
      ctx.font = '10.5px Consolas, monospace';
      ctx.fillText(pctText, n.x + n.r + 10, n.y + 7);
    }
    ctx.textBaseline = 'alphabetic';
  }
}

function drawBezierSynapse(ctx, x0, y0, x1, y1, color, width, pulseT, pulseColor) {
  const cpX = (x0 + x1) * 0.5;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.bezierCurveTo(cpX, y0, cpX, y1, x1, y1);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();

  if (pulseT >= 0 && pulseT <= 1) {
    const t = pulseT;
    const mt = 1 - t;
    const px = mt * mt * mt * x0 + 3 * mt * mt * t * cpX + 3 * mt * t * t * cpX + t * t * t * x1;
    const py = mt * mt * mt * y0 + 3 * mt * mt * t * y0 + 3 * mt * t * t * y1 + t * t * t * y1;
    ctx.beginPath();
    ctx.arc(px, py, 3.0, 0, Math.PI * 2);
    ctx.fillStyle = pulseColor;
    ctx.fill();
  }
}

// =========================================================================
// KI-TRAINING & PARAMETER-LABOR (PRESETS, LIVE BLUEPRINT & LEARNING CURVE)
// =========================================================================
let trainPollTimer = null;

function populateDataAmountOptions(mode) {
  const sel = document.getElementById('inpDataAmount');
  if (!sel) return;
  sel.innerHTML = '';
  if (mode === 'zahlen') {
    [
      ['0.05', '5 % (3.000 Bilder – Blitz-Training ~2 Sek.)'],
      ['0.25', '25 % (15.000 Bilder – Schnell)'],
      ['0.50', '50 % (30.000 Bilder – Halber Datensatz)'],
      ['1.0', '100 % (Alle 60.000 MNIST-Bilder – Standard)']
    ].forEach(([val, txt]) => {
      const opt = document.createElement('option');
      opt.value = val;
      opt.textContent = txt;
      if (val === '1.0') opt.selected = true;
      sel.appendChild(opt);
    });
  } else {
    [
      ['300', '300 / Stufe (1.500 Smileys – Blitz-Training)'],
      ['1000', '1.000 / Stufe (5.000 Smileys – Schnell)'],
      ['2400', '2.400 / Stufe (12.000 Smileys – Standard)'],
      ['3200', '3.200 / Stufe (16.000 Smileys – Extra groß)']
    ].forEach(([val, txt]) => {
      const opt = document.createElement('option');
      opt.value = val;
      opt.textContent = txt;
      if (val === '2400') opt.selected = true;
      sel.appendChild(opt);
    });
  }
}

function selectLabTarget(mode) {
  labTargetMode = mode;
  document.getElementById('labTabZahlen').classList.toggle('active', mode === 'zahlen');
  document.getElementById('labTabSmiley').classList.toggle('active', mode === 'smiley');
  populateDataAmountOptions(mode);

  // Load active saved model parameters for this mode if available, otherwise standard defaults
  const meta = latestStatus ? (mode === 'zahlen' ? latestStatus.zahlen_meta : latestStatus.smiley_meta) : null;
  if (meta) {
    applyMetaToControls(meta, mode);
  } else {
    applyLabPreset('standard');
  }
  updateLoadedBrainSummary();
  onLabParamChange();
}

function updateLoadedBrainSummary() {
  const el = document.getElementById('labLoadedBrainInfo');
  if (!el) return;
  const meta = latestStatus ? (labTargetMode === 'zahlen' ? latestStatus.zahlen_meta : latestStatus.smiley_meta) : null;
  if (!meta) {
    el.textContent = 'Noch kein trainiertes Modell gefunden.';
    return;
  }
  const herkunft = meta.is_custom ? ' • 💾 Eigenes Modell (KI_Studio_Daten)' : ' • 🛡️ Werkseinstellung (integriert)';
  const timePart = (meta.is_custom && meta.trained_at) ? ` (${meta.trained_at})` : '';
  el.textContent = formatMetaSummary(meta) + herkunft + timePart;
}

function setSelectClosest(id, valStr) {
  const sel = document.getElementById(id);
  if (!sel) return;
  for (let i = 0; i < sel.options.length; i++) {
    if (String(sel.options[i].value) === String(valStr) || Number(sel.options[i].value) === Number(valStr)) {
      sel.selectedIndex = i;
      return;
    }
  }
}

function applyMetaToControls(meta, mode) {
  if (!meta) return;
  const convCnt = meta.conv_layers ? meta.conv_layers.length : (mode === 'zahlen' ? 2 : 3);
  setSelectClosest('inpConvLayers', convCnt);
  if (meta.base_filters) setSelectClosest('inpBaseFilters', meta.base_filters);
  setSelectClosest('inpDenseLayers', meta.dense_layers || (meta.hidden_layers ? meta.hidden_layers.length : 1));
  setSelectClosest('inpHiddenUnits', meta.hidden_units || (meta.hidden_layers ? meta.hidden_layers[0] : 64));
  setSelectClosest('inpActivation', (meta.activation || 'relu').toLowerCase());
  document.getElementById('inpEpochs').value = Math.min(15, Math.max(1, meta.epochs || (mode === 'zahlen' ? 4 : 5)));
  document.getElementById('inpDropout').value = Math.round((meta.dropout ?? 0.3) * 100 / 5) * 5;
  setSelectClosest('inpLR', meta.learning_rate ?? 0.001);
  setSelectClosest('inpOptimizer', (meta.optimizer || 'adam').toLowerCase());
  setSelectClosest('inpBatchSize', meta.batch_size || 64);
  if (mode === 'zahlen' && meta.data_fraction) {
    setSelectClosest('inpDataAmount', meta.data_fraction);
  } else if (mode === 'smiley' && meta.samples_per_class) {
    setSelectClosest('inpDataAmount', meta.samples_per_class);
  }
}

function loadActiveModelIntoControls() {
  const meta = latestStatus ? (labTargetMode === 'zahlen' ? latestStatus.zahlen_meta : latestStatus.smiley_meta) : null;
  if (meta) {
    applyMetaToControls(meta, labTargetMode);
    onLabParamChange();
  }
}

function applyLabPreset(preset) {
  const isZ = labTargetMode === 'zahlen';
  if (preset === 'standard') {
    setSelectClosest('inpConvLayers', isZ ? 2 : 3);
    setSelectClosest('inpBaseFilters', 32);
    setSelectClosest('inpDenseLayers', 1);
    setSelectClosest('inpHiddenUnits', 64);
    setSelectClosest('inpActivation', 'relu');
    document.getElementById('inpEpochs').value = isZ ? 4 : 5;
    document.getElementById('inpDropout').value = 30;
    setSelectClosest('inpLR', 0.001);
    setSelectClosest('inpOptimizer', 'adam');
    setSelectClosest('inpBatchSize', 64);
    setSelectClosest('inpDataAmount', isZ ? '1.0' : '2400');
  } else if (preset === 'mini') {
    setSelectClosest('inpConvLayers', 1);
    setSelectClosest('inpBaseFilters', 16);
    setSelectClosest('inpDenseLayers', 1);
    setSelectClosest('inpHiddenUnits', 16);
    setSelectClosest('inpActivation', 'relu');
    document.getElementById('inpEpochs').value = 3;
    document.getElementById('inpDropout').value = 15;
    setSelectClosest('inpLR', 0.003);
    setSelectClosest('inpOptimizer', 'adam');
    setSelectClosest('inpBatchSize', 64);
    setSelectClosest('inpDataAmount', isZ ? '0.25' : '1000');
  } else if (preset === 'deep') {
    setSelectClosest('inpConvLayers', 3);
    setSelectClosest('inpBaseFilters', 32);
    setSelectClosest('inpDenseLayers', 2);
    setSelectClosest('inpHiddenUnits', 48);
    setSelectClosest('inpActivation', 'elu');
    document.getElementById('inpEpochs').value = 5;
    document.getElementById('inpDropout').value = 25;
    setSelectClosest('inpLR', 0.001);
    setSelectClosest('inpOptimizer', 'adam');
    setSelectClosest('inpBatchSize', 64);
    setSelectClosest('inpDataAmount', isZ ? '0.50' : '2400');
  } else if (preset === 'mlp') {
    setSelectClosest('inpConvLayers', 0);
    setSelectClosest('inpDenseLayers', 2);
    setSelectClosest('inpHiddenUnits', 48);
    setSelectClosest('inpActivation', 'tanh');
    document.getElementById('inpEpochs').value = 5;
    document.getElementById('inpDropout').value = 20;
    setSelectClosest('inpLR', 0.001);
    setSelectClosest('inpOptimizer', 'adam');
    setSelectClosest('inpBatchSize', 64);
    setSelectClosest('inpDataAmount', isZ ? '0.50' : '1000');
  } else if (preset === 'chaos') {
    setSelectClosest('inpConvLayers', 1);
    setSelectClosest('inpBaseFilters', 16);
    setSelectClosest('inpDenseLayers', 1);
    setSelectClosest('inpHiddenUnits', 32);
    setSelectClosest('inpActivation', 'relu');
    document.getElementById('inpEpochs').value = 4;
    document.getElementById('inpDropout').value = 0;
    setSelectClosest('inpLR', 0.05);
    setSelectClosest('inpOptimizer', 'adam');
    setSelectClosest('inpBatchSize', 32);
    setSelectClosest('inpDataAmount', isZ ? '0.25' : '300');
  }
  onLabParamChange();
}

function readLabParams() {
  const convLayers = parseInt(document.getElementById('inpConvLayers').value, 10) || 0;
  const baseFilters = parseInt(document.getElementById('inpBaseFilters').value, 10) || 32;
  const denseLayers = parseInt(document.getElementById('inpDenseLayers').value, 10) || 1;
  const hiddenUnits = parseInt(document.getElementById('inpHiddenUnits').value, 10) || 64;
  const activation = document.getElementById('inpActivation').value || 'relu';
  const epochs = parseInt(document.getElementById('inpEpochs').value, 10) || 4;
  const dropoutPct = parseInt(document.getElementById('inpDropout').value, 10) || 0;
  const lr = parseFloat(document.getElementById('inpLR').value) || 0.001;
  const optimizer = document.getElementById('inpOptimizer').value || 'adam';
  const batchSize = parseInt(document.getElementById('inpBatchSize').value, 10) || 64;
  const dataRaw = document.getElementById('inpDataAmount').value;

  const filterChain = [];
  if (convLayers > 0) {
    filterChain.push(baseFilters);
    for (let i = 1; i < convLayers; i++) {
      filterChain.push(Math.min(128, baseFilters * 2));
    }
  }

  const params = {
    conv_layers: convLayers,
    base_filters: baseFilters,
    dense_layers: denseLayers,
    hidden_units: hiddenUnits,
    activation,
    epochs,
    dropout: dropoutPct / 100.0,
    learning_rate: lr,
    optimizer,
    batch_size: batchSize,
    filterChain
  };
  if (labTargetMode === 'zahlen') {
    params.data_fraction = parseFloat(dataRaw) || 1.0;
  } else {
    params.samples_per_class = parseInt(dataRaw, 10) || 2400;
  }
  return params;
}

function onLabParamChange() {
  const p = readLabParams();
  document.getElementById('lblConvLayers').textContent = p.conv_layers === 0 ? '0 (MLP)' : `${p.conv_layers}`;
  document.getElementById('boxBaseFilters').classList.toggle('disabled-box', p.conv_layers === 0);
  document.getElementById('lblBaseFilters').textContent = p.conv_layers === 0 ? 'Aus' : p.filterChain.join('→');
  document.getElementById('lblDenseLayers').textContent = `${p.dense_layers} Ebene${p.dense_layers > 1 ? 'n' : ''}`;
  document.getElementById('lblHiddenUnits').textContent = `${p.hidden_units}`;
  document.getElementById('lblActivation').textContent = p.activation.toUpperCase();
  document.getElementById('lblEpochs').textContent = `${p.epochs}`;
  document.getElementById('lblDropout').textContent = `${Math.round(p.dropout * 100)} %`;
  document.getElementById('lblLR').textContent = `${p.learning_rate}`;
  document.getElementById('lblOptimizer').textContent = p.optimizer.toUpperCase();
  document.getElementById('lblBatchSize').textContent = `${p.batch_size}`;

  if (labTargetMode === 'zahlen') {
    document.getElementById('lblDataAmount').textContent = `${Math.round((p.data_fraction || 1) * 100)} %`;
  } else {
    document.getElementById('lblDataAmount').textContent = `${(p.samples_per_class || 2400) * 5}`;
  }

  // Live Visual Blueprint Chain
  const outCount = labTargetMode === 'zahlen' ? 10 : 5;
  const inRes = labTargetMode === 'zahlen' ? '28×28' : '48×48';
  let totalNodes = outCount;
  const parts = [`<span class="bp-node bp-in">Eingabe ${inRes}</span>`];

  p.filterChain.forEach((fc, idx) => {
    totalNodes += fc;
    parts.push(`<span style="color:#64748b">➔</span><span class="bp-node bp-conv">Filter ${idx + 1}: ${fc}</span>`);
  });

  for (let d = 0; d < p.dense_layers; d++) {
    totalNodes += p.hidden_units;
    const dLabel = p.dense_layers === 1 ? 'Denkschicht' : `Denk ${d + 1}`;
    parts.push(`<span style="color:#64748b">➔</span><span class="bp-node bp-hid">${dLabel}: ${p.hidden_units} ${p.activation.toUpperCase()}</span>`);
  }

  parts.push(`<span style="color:#64748b">➔</span><span class="bp-node bp-out">Softmax: ${outCount}</span>`);
  document.getElementById('blueprintChain').innerHTML = parts.join('');
  const archWord = p.conv_layers === 0 ? 'Reines MLP' : 'CNN';
  document.getElementById('bpTotalNodes').textContent = `${archWord} • ${totalNodes} Knoten im Visualisierer`;
}

function updateDashboardCardBadges(meta, badgesElId, isZahlen) {
  const box = document.getElementById(badgesElId);
  if (!box || !meta) return;
  const pills = [];
  if (meta.conv_layers && meta.conv_layers.length > 0) {
    pills.push(`${meta.conv_layers.join(' + ')} Filter`);
  } else {
    pills.push('Reines Pixel-MLP (0 Filter)');
  }
  const hidList = meta.hidden_layers || [64];
  const actUp = (meta.activation || 'relu').toUpperCase();
  pills.push(`${hidList.join('→')} ${actUp}-Neuronen`);
  if (meta.test_accuracy !== null && meta.test_accuracy !== undefined) {
    pills.push(`Genauigkeit: ${Number(meta.test_accuracy).toFixed(1)} %`);
  } else {
    pills.push(isZahlen ? '10 Ausgabeknoten' : '5 Gefühls-Stufen');
  }
  box.innerHTML = pills.map(t => `<span class="pill-badge">${t}</span>`).join('');
}

async function refreshStatus() {
  try {
    const resp = await fetch('/api/status');
    if (!resp.ok) return;
    const s = await resp.json();
    latestStatus = s;

    if (s.booting) {
      const bStep = (s.boot && s.boot.step) ? s.boot.step : 'Lade TensorFlow-Engine...';
      document.getElementById('engineStatusText').textContent = bStep;
      document.getElementById('dashZahlenStatus').textContent = 'Zahlen-Netz: Lädt...';
      document.getElementById('dashSmileyStatus').textContent = 'Smiley-Netz: Lädt...';
      updateDataStorageUI(s.data_info);
      return;
    }

    const zMeta = s.zahlen_meta;
    const sMeta = s.smiley_meta;
    const zTxt = s.zahlen_bereit ? `Bereit (${zMeta ? zMeta.arch_type + ' • ' + zMeta.total_nodes + ' Kn.' : '✓'})` : 'Fehlt';
    const sTxt = s.smiley_bereit ? `Bereit (${sMeta ? sMeta.arch_type + ' • ' + sMeta.total_nodes + ' Kn.' : '✓'})` : 'Fehlt';

    document.getElementById('engineStatusText').textContent =
      `TensorFlow ${s.tf_version} aktiv • Zahlen-KI: ${zTxt} • Smiley-KI: ${sTxt}`;
    document.getElementById('dashZahlenStatus').textContent = `Zahlen-Netz: ${zTxt}`;
    document.getElementById('dashSmileyStatus').textContent = `Smiley-Netz: ${sTxt}`;

    if (zMeta) {
      updateDashboardCardBadges(zMeta, 'dashZahlenBadges', true);
      document.getElementById('labSubZahlen').textContent =
        `Aktiv: ${zMeta.arch_type} • ${zMeta.total_nodes} Knoten${zMeta.test_accuracy != null ? ' • ' + zMeta.test_accuracy.toFixed(1) + '%' : ''}`;
    }
    if (sMeta) {
      updateDashboardCardBadges(sMeta, 'dashSmileyBadges', false);
      document.getElementById('labSubSmiley').textContent =
        `Aktiv: ${sMeta.arch_type} • ${sMeta.total_nodes} Knoten${sMeta.test_accuracy != null ? ' • ' + sMeta.test_accuracy.toFixed(1) + '%' : ''}`;
    }
    updateLoadedBrainSummary();
    updateDataStorageUI(s.data_info);

    // If no live metrics yet, display saved history of selected model in chart!
    if (latestTrainMetrics.length === 0) {
      const activeMeta = labTargetMode === 'zahlen' ? zMeta : sMeta;
      if (activeMeta && activeMeta.history && activeMeta.history.acc && activeMeta.history.acc.length > 0) {
        const hist = activeMeta.history;
        const synth = hist.acc.map((a, idx) => ({
          epoch: idx + 1,
          total_epochs: hist.acc.length,
          acc: a,
          val_acc: hist.val_acc ? hist.val_acc[idx] : a,
          loss: hist.loss ? hist.loss[idx] : 0
        }));
        renderTrainingChart(synth);
      } else {
        renderTrainingChart([]);
      }
    }
  } catch (e) {}
}

async function startCustomTraining() {
  const p = readLabParams();
  const scriptName = labTargetMode === 'zahlen' ? 'train_digits' : 'train_smiley';
  lastTrainedMode = labTargetMode;
  document.getElementById('postTrainActionBox').classList.add('hidden');
  latestTrainMetrics = [];
  renderTrainingChart([]);
  updateTrainProgressUI(true, [], p.epochs);

  const resp = await fetch('/api/train/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ script: scriptName, params: p })
  });
  const d = await resp.json();
  if (!d.ok) {
    alert(d.message);
    return;
  }
  pollTrainingLogs();
}

async function startTraining(scriptName) {
  document.getElementById('postTrainActionBox').classList.add('hidden');
  const resp = await fetch('/api/train/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ script: scriptName })
  });
  const d = await resp.json();
  if (!d.ok) {
    alert(d.message);
    return;
  }
  pollTrainingLogs();
}

async function stopTraining() {
  await fetch('/api/train/stop', { method: 'POST', body: '{}' });
}

function updateTrainProgressUI(active, metrics, fallbackEpochs) {
  const lbl = document.getElementById('trainProgressLabel');
  const pctEl = document.getElementById('trainProgressPct');
  const fill = document.getElementById('trainProgressFill');

  if (metrics && metrics.length > 0) {
    const last = metrics[metrics.length - 1];
    const tot = Math.max(1, last.total_epochs || fallbackEpochs || 1);
    const pct = active
      ? Math.min(96, Math.round((last.epoch / tot) * 100))
      : 100;
    lbl.textContent = active
      ? `Trainiere Epoche ${last.epoch} / ${tot} • Prüf-Genauigkeit: ${last.val_acc.toFixed(2)} % (Loss: ${last.loss.toFixed(3)})`
      : `Training abgeschlossen (${tot} Epochen) • Prüf-Genauigkeit: ${last.val_acc.toFixed(2)} % ✓`;
    pctEl.textContent = `${pct} %`;
    fill.style.width = `${pct}%`;
  } else if (active) {
    lbl.textContent = 'Bereite Datensatz vor & kompiliere TensorFlow-Gehirn...';
    pctEl.textContent = '8 %';
    fill.style.width = '8%';
  }
}

function renderTrainingChart(metrics) {
  const canvas = document.getElementById('trainChartCanvas');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width > 20 ? rect.width : 540;
  const h = 155;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const padL = 38, padR = 16, padT = 14, padB = 24;
  const cw = w - padL - padR;
  const ch = h - padT - padB;

  // Grid lines (25%, 50%, 75%, 100%)
  ctx.font = '10px Consolas, monospace';
  ctx.textAlign = 'right';
  [0, 50, 80, 100].forEach(pct => {
    const y = padT + ch - (pct / 100) * ch;
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.14)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.fillText(`${pct}%`, padL - 6, y + 3);
  });

  if (!metrics || metrics.length === 0) {
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.font = '12px "Segoe UI", sans-serif';
    ctx.fillText('Starte ein Training, um die Live-Lernkurve pro Epoche zu sehen', padL + cw / 2, padT + ch / 2 + 4);
    return;
  }

  const totalEp = Math.max(metrics.length, metrics[0].total_epochs || metrics.length, 2);
  const xFor = (ep) => padL + ((ep - 1) / Math.max(1, totalEp - 1)) * cw;
  const yForAcc = (acc) => padT + ch - (Math.min(100, Math.max(0, acc)) / 100) * ch;

  const maxLoss = Math.max(1.0, ...metrics.map(m => m.loss || 0));
  const yForLoss = (loss) => padT + ch - (Math.min(maxLoss, Math.max(0, loss)) / maxLoss) * ch;

  // Epoch X labels
  ctx.textAlign = 'center';
  ctx.fillStyle = '#94a3b8';
  for (let ep = 1; ep <= totalEp; ep++) {
    if (totalEp <= 10 || ep === 1 || ep === totalEp || ep % 2 === 1) {
      ctx.fillText(`E${ep}`, xFor(ep), h - 6);
    }
  }

  function drawSeries(points, color, dashed) {
    if (points.length === 0) return;
    ctx.save();
    if (dashed && ctx.setLineDash) ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    points.forEach((pt, idx) => {
      if (idx === 0) ctx.moveTo(pt[0], pt[1]);
      else ctx.lineTo(pt[0], pt[1]);
    });
    ctx.stroke();
    ctx.restore();

    points.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], 3.4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });
  }

  drawSeries(metrics.map(m => [xFor(m.epoch), yForLoss(m.loss || 0)]), '#fbbf24', true);
  drawSeries(metrics.map(m => [xFor(m.epoch), yForAcc(m.acc || 0)]), '#38bdf8', false);
  drawSeries(metrics.map(m => [xFor(m.epoch), yForAcc(m.val_acc || 0)]), '#34d399', false);
}

async function pollTrainingLogs() {
  if (trainPollTimer) clearTimeout(trainPollTimer);
  try {
    const resp = await fetch('/api/train/status');
    if (resp.ok) {
      const d = await resp.json();
      const box = document.getElementById('terminalOutput');
      box.textContent = (d.lines || []).join('\n');
      box.scrollTop = box.scrollHeight;

      if (d.metrics && d.metrics.length > 0) {
        latestTrainMetrics = d.metrics;
        renderTrainingChart(latestTrainMetrics);
      }
      updateTrainProgressUI(d.active, latestTrainMetrics);

      if (d.active) {
        trainPollTimer = setTimeout(pollTrainingLogs, 300);
      } else {
        await refreshStatus();
        if (d.script === 'train_digits' || d.script === 'train_smiley') {
          lastTrainedMode = d.script === 'train_digits' ? 'zahlen' : 'smiley';
          const btn = document.getElementById('btnOpenTrainedBrain');
          btn.textContent = lastTrainedMode === 'zahlen'
            ? '🧠 Neues Zahlen-Gehirn jetzt live in der Visualisierung testen →'
            : '🧠 Neues Smiley-Gehirn jetzt live in der Visualisierung testen →';
          document.getElementById('postTrainActionBox').classList.remove('hidden');
        }
      }
    }
  } catch (e) {}
}

async function openTrainedBrainInStudio() {
  const target = lastTrainedMode || labTargetMode || 'zahlen';
  brainData = null;
  smoothConv = [];
  smoothHiddenLayers = [];
  switchView(target);
  setTimeout(() => loadRandomSample(), 120);
}

function updateDataStorageUI(info) {
  if (!info) return;
  const exists = Boolean(info.exists);
  const folderExists = Boolean(info.folder_exists);
  const runtimeExists = Boolean(info.runtime_cache_exists);
  const sizeText = info.total_mb >= 1.0
    ? `${Number(info.total_mb).toFixed(2)} MB`
    : `${Number(info.total_kb || 0).toFixed(1)} KB`;
  const fileCount = (info.files || []).length;

  const dashPill = document.getElementById('dashDataPill');
  const labPill = document.getElementById('labDataPill');
  const labPath = document.getElementById('labDataPath');
  const btnDashReset = document.getElementById('btnDashResetData');
  const btnLabReset = document.getElementById('btnLabResetData');
  const btnDashOpen = document.getElementById('btnDashOpenData');
  const btnLabOpen = document.getElementById('btnLabOpenData');
  const btnDashCleanup = document.getElementById('btnDashCleanupAll');
  const btnLabCleanup = document.getElementById('btnLabCleanupAll');

  if (btnDashCleanup) btnDashCleanup.disabled = !folderExists;
  if (btnLabCleanup) btnLabCleanup.disabled = !folderExists;

  if (!exists) {
    const cleanTxt = runtimeExists
      ? '⚡ Schnellstart aktiv • 0 KB eigene Trainings (Werks-Gehirne)'
      : '✓ 0 KB • Keine eigenen Dateien angelegt (Werks-Gehirne aktiv)';
    if (dashPill) {
      dashPill.textContent = cleanTxt;
      dashPill.classList.remove('has-custom');
    }
    if (labPill) {
      labPill.textContent = cleanTxt;
      labPill.classList.remove('has-custom');
    }
    if (labPath) {
      labPath.textContent = folderExists
        ? `Ordner neben EXE: ${info.data_dir}  [0 eigene Modelle • Schnellstart-Cache aktiv]`
        : `Kein Ordner angelegt • Geplanter Pfad bei eigenem Training: ${info.data_dir}`;
    }
    if (btnDashReset) btnDashReset.disabled = true;
    if (btnLabReset) btnLabReset.disabled = true;
    const openTxt = folderExists ? `📂 Ordner „${info.data_dir_name}“ öffnen` : '📂 Anwendungs-Ordner zeigen';
    if (btnDashOpen) btnDashOpen.textContent = openTxt;
    if (btnLabOpen) btnLabOpen.textContent = openTxt;
  } else {
    const activeTxt = `💾 Eigene Trainings in „${info.data_dir_name}“ (${sizeText} • ${fileCount} Datei${fileCount === 1 ? '' : 'en'})`;
    if (dashPill) {
      dashPill.textContent = activeTxt;
      dashPill.classList.add('has-custom');
    }
    if (labPill) {
      labPill.textContent = activeTxt;
      labPill.classList.add('has-custom');
    }
    if (labPath) {
      const names = (info.files || []).map(f => f.name).join(', ');
      labPath.textContent = `Speicherort: ${info.models_dir || info.data_dir}${names ? '  [' + names + ']' : ''}`;
    }
    if (btnDashReset) btnDashReset.disabled = false;
    if (btnLabReset) btnLabReset.disabled = false;
    if (btnDashOpen) btnDashOpen.textContent = `📂 Ordner „${info.data_dir_name}“ öffnen`;
    if (btnLabOpen) btnLabOpen.textContent = `📂 Ordner „${info.data_dir_name}“ öffnen`;
  }
}

async function resetUserData() {
  const okConfirm = confirm(
    'Möchtest du alle selbst trainierten KI-Modelle löschen und sofort auf die eingebauten Werks-Gehirne zurücksetzen?'
  );
  if (!okConfirm) return;

  try {
    const resp = await fetch('/api/data/reset', { method: 'POST', body: '{}' });
    const d = await resp.json();
    if (!d.ok) {
      alert(d.message || 'Löschen fehlgeschlagen.');
      return;
    }
    latestTrainMetrics = [];
    brainData = null;
    smoothConv = [];
    smoothHiddenLayers = [];
    document.getElementById('postTrainActionBox').classList.add('hidden');
    document.getElementById('trainProgressLabel').textContent = 'Werkseinstellungen aktiv – eigene Trainingsdaten gelöscht ✓';
    document.getElementById('trainProgressPct').textContent = '0 %';
    document.getElementById('trainProgressFill').style.width = '0%';
    await refreshStatus();
    loadActiveModelIntoControls();
    if (currentView === 'zahlen' || currentView === 'smiley') {
      triggerAnalyzeNow();
    }
  } catch (e) {
    alert('Verbindungsfehler beim Zurücksetzen der Daten.');
  }
}

let shuttingDownForCleanup = false;
async function cleanupAllAndExit() {
  const okConfirm = confirm(
    'Möchtest du die Anwendung jetzt schließen und den gesamten Ordner „KI_Studio_Daten“ (inklusive Schnellstart-Engine und eigener Trainings) restlos löschen?\n\nDanach bleibt ausschließlich die einzelne TensorFlow_KI_Studio.exe übrig.'
  );
  if (!okConfirm) return;

  try {
    shuttingDownForCleanup = true;
    const overlay = document.getElementById('bootOverlay');
    if (overlay) {
      overlay.classList.remove('hidden', 'fade-out');
      document.getElementById('bootStepText').textContent = 'Schließe Anwendung & entferne gesamten Ordner „KI_Studio_Daten“...';
      document.getElementById('bootBarFill').style.width = '100%';
      document.getElementById('bootPctText').textContent = '100 %';
    }
    const resp = await fetch('/api/data/cleanup_and_exit', { method: 'POST', body: '{}' });
    const d = await resp.json();
    if (!d.ok) {
      shuttingDownForCleanup = false;
      if (overlay) overlay.classList.add('hidden');
      alert(d.message || 'Komplettlöschung fehlgeschlagen.');
      return;
    }
    setTimeout(() => {
      try { window.close(); } catch (e) {}
      const stepEl = document.getElementById('bootStepText');
      if (stepEl) stepEl.textContent = 'Ordner wird entfernt – du kannst dieses Fenster jetzt schließen ✓';
    }, 150);
  } catch (e) {
    shuttingDownForCleanup = false;
    alert('Verbindungsfehler beim Aufräumen.');
  }
}

async function openDataLocation() {
  await fetch('/api/data/open', { method: 'POST', body: '{}' });
}

// --- Smooth Live Boot Progress Poller ---
let bootFinished = false;
let bootServerPct = 18;
let bootDisplayPct = 18;

const bootAnimTimer = setInterval(() => {
  if (bootFinished || shuttingDownForCleanup) return;
  // Smoothly creep toward next milestone so the bar ALWAYS visibly moves while C++ DLLs load
  const ceiling = bootServerPct < 45 ? 64 : (bootServerPct < 72 ? 85 : 96);
  if (bootDisplayPct < bootServerPct) {
    bootDisplayPct += Math.max(1.2, (bootServerPct - bootDisplayPct) * 0.35);
  } else if (bootDisplayPct < ceiling) {
    bootDisplayPct += 0.35;
  }
  const shown = Math.min(99, Math.round(bootDisplayPct));
  const fill = document.getElementById('bootBarFill');
  const pctTxt = document.getElementById('bootPctText');
  if (fill) fill.style.width = `${shown}%`;
  if (pctTxt) pctTxt.textContent = `${shown} %`;
}, 55);

async function pollBootProgress() {
  if (bootFinished || shuttingDownForCleanup) return;
  try {
    const resp = await fetch('/api/boot');
    if (resp.ok) {
      const b = await resp.json();
      if (typeof b.percent === 'number' && b.percent > bootServerPct) {
        bootServerPct = b.percent;
      }
      const stepEl = document.getElementById('bootStepText');
      if (stepEl && b.step) stepEl.textContent = b.step;

      if (b.error) {
        clearInterval(bootAnimTimer);
        if (stepEl) {
          stepEl.style.color = '#fb7185';
          stepEl.textContent = b.step || `Fehler: ${b.error}`;
        }
        return;
      }

      if (b.ready) {
        bootFinished = true;
        clearInterval(bootAnimTimer);
        const fill = document.getElementById('bootBarFill');
        const pctTxt = document.getElementById('bootPctText');
        if (fill) fill.style.width = '100%';
        if (pctTxt) pctTxt.textContent = '100 %';
        if (stepEl) stepEl.textContent = 'KI-Gehirne bereit!';

        await refreshStatus();
        if (currentView === 'zahlen' || currentView === 'smiley') {
          triggerAnalyzeNow();
        }

        const overlay = document.getElementById('bootOverlay');
        if (overlay) {
          overlay.classList.add('fade-out');
          setTimeout(() => overlay.classList.add('hidden'), 260);
        }
        return;
      }
    }
  } catch (e) {}
  setTimeout(pollBootProgress, 120);
}

// --- Lifecycle Heartbeat & Clean Shutdown ---
setInterval(() => {
  if (shuttingDownForCleanup) return;
  fetch('/api/heartbeat', { method: 'POST', body: '{}' }).catch(() => {});
}, 4000);

window.addEventListener('pagehide', () => {
  if (shuttingDownForCleanup) return;
  try {
    navigator.sendBeacon('/api/close', '{}');
  } catch (e) {}
});

// Boot: Optimale Fenstergröße im App-Modus anfordern
try {
  if (window.outerWidth < 1180 || window.outerHeight < 760) {
    const targetW = Math.min(1460, window.screen.availWidth - 40);
    const targetH = Math.min(920, window.screen.availHeight - 40);
    window.resizeTo(targetW, targetH);
  }
} catch (e) {}
populateDataAmountOptions('zahlen');
onLabParamChange();
pollBootProgress();
refreshStatus();
requestAnimationFrame(renderBrainLoop);
