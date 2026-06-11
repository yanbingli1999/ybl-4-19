let fitChart = null;
let residualChart = null;
let currentResultId = null;
let currentDatasetId = null;
let isDirty = false;
let smoothedPoints = [];
let smoothConfig = {
  enabled: true,
  method: 'movingAverage',
  windowSize: 5,
  polyOrder: 2,
  dataSource: '手动输入',
  pointCount: 0
};

const STORAGE_KEY = 'curve_fit_smooth_config';

const modelTypeLabels = {
  linear: '线性模型',
  exponential: '指数模型',
  quadratic: '二次曲线'
};

const smoothMethodLabels = {
  movingAverage: '移动平均',
  median: '中位数滤波',
  savitzkyGolay: 'Savitzky-Golay'
};

function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  toast.className = `toast ${type} show`;
  toast.textContent = message;
  setTimeout(() => {
    toast.classList.remove('show');
  }, 3000);
}

function saveSmoothConfig() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(smoothConfig));
  } catch (e) {
    console.warn('保存平滑配置失败:', e);
  }
}

function loadSmoothConfig() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const config = JSON.parse(saved);
      smoothConfig = { ...smoothConfig, ...config };
    }
  } catch (e) {
    console.warn('加载平滑配置失败:', e);
  }
}

function applySmoothConfigToUI() {
  document.getElementById('enableSmooth').checked = smoothConfig.enabled;
  document.querySelector(`input[name="smoothMethod"][value="${smoothConfig.method}"]`).checked = true;
  document.getElementById('windowSize').value = smoothConfig.windowSize;
  document.getElementById('windowSizeValue').textContent = smoothConfig.windowSize;
  document.getElementById('polyOrder').value = smoothConfig.polyOrder;
  document.getElementById('polyOrderValue').textContent = smoothConfig.polyOrder;

  const polyOrderItem = document.getElementById('polyOrderItem');
  if (smoothConfig.method === 'savitzkyGolay') {
    polyOrderItem.style.display = 'flex';
  } else {
    polyOrderItem.style.display = 'none';
  }

  updateSmoothInfoDisplay();
}

function updateSmoothInfoDisplay() {
  document.getElementById('smoothPointCount').textContent = smoothConfig.pointCount > 0 ? smoothConfig.pointCount : '—';
  document.getElementById('smoothDataSource').textContent = smoothConfig.dataSource || '—';
}

function movingAverageSmooth(points, windowSize) {
  const n = points.length;
  const halfWindow = Math.floor(windowSize / 2);
  const result = [];

  const sortedPoints = [...points].sort((a, b) => a.x - b.x);

  for (let i = 0; i < n; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - halfWindow); j <= Math.min(n - 1, i + halfWindow); j++) {
      sum += sortedPoints[j].y;
      count++;
    }
    result.push({ x: sortedPoints[i].x, y: sum / count });
  }

  return result;
}

function medianFilterSmooth(points, windowSize) {
  const n = points.length;
  const halfWindow = Math.floor(windowSize / 2);
  const result = [];

  const sortedPoints = [...points].sort((a, b) => a.x - b.x);

  for (let i = 0; i < n; i++) {
    const windowValues = [];
    for (let j = Math.max(0, i - halfWindow); j <= Math.min(n - 1, i + halfWindow); j++) {
      windowValues.push(sortedPoints[j].y);
    }
    windowValues.sort((a, b) => a - b);
    const mid = Math.floor(windowValues.length / 2);
    const median = windowValues.length % 2 === 0
      ? (windowValues[mid - 1] + windowValues[mid]) / 2
      : windowValues[mid];
    result.push({ x: sortedPoints[i].x, y: median });
  }

  return result;
}

function savitzkyGolaySmooth(points, windowSize, polyOrder) {
  const n = points.length;
  if (n < windowSize) return [...points];

  const halfWindow = Math.floor(windowSize / 2);
  const result = [];

  const sortedPoints = [...points].sort((a, b) => a.x - b.x);
  const ys = sortedPoints.map(p => p.y);

  function vandermonde(x, order) {
    const m = x.length;
    const mat = [];
    for (let i = 0; i < m; i++) {
      const row = [];
      for (let j = 0; j <= order; j++) {
        row.push(Math.pow(x[i], j));
      }
      mat.push(row);
    }
    return mat;
  }

  function transpose(mat) {
    const rows = mat.length;
    const cols = mat[0].length;
    const result = [];
    for (let j = 0; j < cols; j++) {
      const row = [];
      for (let i = 0; i < rows; i++) {
        row.push(mat[i][j]);
      }
      result.push(row);
    }
    return result;
  }

  function multiplyMatrices(a, b) {
    const rowsA = a.length;
    const colsA = a[0].length;
    const colsB = b[0].length;
    const result = [];
    for (let i = 0; i < rowsA; i++) {
      const row = [];
      for (let j = 0; j < colsB; j++) {
        let sum = 0;
        for (let k = 0; k < colsA; k++) {
          sum += a[i][k] * b[k][j];
        }
        row.push(sum);
      }
      result.push(row);
    }
    return result;
  }

  function invertMatrix(mat) {
    const n = mat.length;
    const augmented = [];
    for (let i = 0; i < n; i++) {
      const row = [...mat[i]];
      for (let j = 0; j < n; j++) {
        row.push(i === j ? 1 : 0);
      }
      augmented.push(row);
    }

    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
          maxRow = k;
        }
      }
      [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

      const pivot = augmented[i][i];
      if (Math.abs(pivot) < 1e-10) return null;
      for (let j = 0; j < 2 * n; j++) {
        augmented[i][j] /= pivot;
      }

      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = augmented[k][i];
          for (let j = 0; j < 2 * n; j++) {
            augmented[k][j] -= factor * augmented[i][j];
          }
        }
      }
    }

    const inv = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = n; j < 2 * n; j++) {
        row.push(augmented[i][j]);
      }
      inv.push(row);
    }
    return inv;
  }

  const xWindow = [];
  for (let i = -halfWindow; i <= halfWindow; i++) {
    xWindow.push(i);
  }
  const V = vandermonde(xWindow, polyOrder);
  const VT = transpose(V);
  const VTV = multiplyMatrices(VT, V);
  const VTVInv = invertMatrix(VTV);
  if (!VTVInv) {
    return movingAverageSmooth(sortedPoints, windowSize);
  }
  const coefMatrix = multiplyMatrices(VTVInv, VT);

  for (let i = 0; i < n; i++) {
    let smoothedY;
    if (i < halfWindow || i >= n - halfWindow) {
      let sum = 0;
      let count = 0;
      const start = Math.max(0, i - halfWindow);
      const end = Math.min(n - 1, i + halfWindow);
      for (let j = start; j <= end; j++) {
        sum += ys[j];
        count++;
      }
      smoothedY = sum / count;
    } else {
      let y0 = 0;
      for (let j = 0; j < windowSize; j++) {
        y0 += coefMatrix[0][j] * ys[i - halfWindow + j];
      }
      smoothedY = y0;
    }
    result.push({ x: sortedPoints[i].x, y: smoothedY });
  }

  return result;
}

function applySmoothing() {
  const points = getTableData();
  if (points.length < 2) {
    showToast('请至少输入2个有效数据点', 'error');
    return;
  }

  const method = document.querySelector('input[name="smoothMethod"]:checked').value;
  const windowSize = parseInt(document.getElementById('windowSize').value);
  const polyOrder = parseInt(document.getElementById('polyOrder').value);

  smoothConfig.method = method;
  smoothConfig.windowSize = windowSize;
  smoothConfig.polyOrder = polyOrder;
  smoothConfig.pointCount = points.length;
  smoothConfig.dataSource = document.getElementById('datasetName').value || '手动输入';

  let result;
  switch (method) {
    case 'movingAverage':
      result = movingAverageSmooth(points, windowSize);
      break;
    case 'median':
      result = medianFilterSmooth(points, windowSize);
      break;
    case 'savitzkyGolay':
      result = savitzkyGolaySmooth(points, windowSize, polyOrder);
      break;
    default:
      result = points;
  }

  smoothedPoints = result;
  saveSmoothConfig();
  updateSmoothInfoDisplay();
  updateChartWithSmoothData();
  showToast(`已应用${smoothMethodLabels[method]}平滑`, 'success');
}

function updateChartWithSmoothData() {
  if (!fitChart) return;

  const originalPoints = getTableData();
  const normalPoints = [];
  const outlierPoints = [];

  if (fitChart.data.datasets.length >= 4) {
    fitChart.data.datasets[0].data = originalPoints;
    fitChart.data.datasets[1].data = smoothedPoints;
  }

  fitChart.update();
}

function updateDatasetButtons() {
  const updateBtn = document.getElementById('updateDatasetBtn');
  if (currentDatasetId) {
    updateBtn.style.display = 'block';
    if (isDirty) {
      updateBtn.textContent = '💾 更新当前数据集 *';
    } else {
      updateBtn.textContent = '💾 更新当前数据集';
    }
  } else {
    updateBtn.style.display = 'none';
  }
}

function markDirty() {
  isDirty = true;
  updateDatasetButtons();
}

function clearDirty() {
  isDirty = false;
  updateDatasetButtons();
}

function initCharts() {
  const fitCtx = document.getElementById('fitChart').getContext('2d');
  const residualCtx = document.getElementById('residualChart').getContext('2d');

  fitChart = new Chart(fitCtx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: '原始数据',
          data: [],
          backgroundColor: '#3b82f6',
          borderColor: '#3b82f6',
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: false,
          opacity: 0.7
        },
        {
          label: '平滑数据',
          data: [],
          backgroundColor: '#10b981',
          borderColor: '#10b981',
          pointRadius: 7,
          pointHoverRadius: 9,
          showLine: false
        },
        {
          label: '拟合曲线',
          data: [],
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderWidth: 3,
          pointRadius: 0,
          showLine: true,
          tension: 0.1,
          fill: false
        },
        {
          label: '异常点',
          data: [],
          backgroundColor: '#f59e0b',
          borderColor: '#d97706',
          pointRadius: 9,
          pointStyle: 'triangle',
          showLine: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(30, 41, 59, 0.95)',
          titleFont: { size: 13 },
          bodyFont: { size: 12 },
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            label: (context) => {
              const x = context.parsed.x?.toFixed(4) || 0;
              const y = context.parsed.y?.toFixed(4) || 0;
              return `(${x}, ${y})`;
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          grid: { color: 'rgba(148, 163, 184, 0.2)' },
          ticks: { font: { size: 12 }, color: '#64748b' },
          title: { display: true, text: 'X 轴', font: { size: 13, weight: '600' }, color: '#475569' }
        },
        y: {
          grid: { color: 'rgba(148, 163, 184, 0.2)' },
          ticks: { font: { size: 12 }, color: '#64748b' },
          title: { display: true, text: 'Y 轴', font: { size: 13, weight: '600' }, color: '#475569' }
        }
      }
    }
  });

  residualChart = new Chart(residualCtx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: '残差',
          data: [],
          backgroundColor: '#8b5cf6',
          borderColor: '#8b5cf6',
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: false
        },
        {
          label: '零参考线',
          data: [],
          borderColor: '#10b981',
          borderWidth: 2,
          borderDash: [8, 4],
          pointRadius: 0,
          showLine: true,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(30, 41, 59, 0.95)',
          titleFont: { size: 13 },
          bodyFont: { size: 12 },
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            label: (context) => {
              if (context.datasetIndex === 0) {
                const x = context.parsed.x?.toFixed(4) || 0;
                const y = context.parsed.y?.toFixed(6) || 0;
                return `x=${x}, 残差=${y}`;
              }
              return '';
            }
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          grid: { color: 'rgba(148, 163, 184, 0.2)' },
          ticks: { font: { size: 12 }, color: '#64748b' },
          title: { display: true, text: 'X 轴', font: { size: 13, weight: '600' }, color: '#475569' }
        },
        y: {
          grid: { color: 'rgba(148, 163, 184, 0.2)' },
          ticks: { font: { size: 12 }, color: '#64748b' },
          title: { display: true, text: '残差 (观测值 - 预测值)', font: { size: 13, weight: '600' }, color: '#475569' }
        }
      }
    }
  });
}

function addDataRow(x = '', y = '') {
  const tbody = document.getElementById('dataTableBody');
  const rowIndex = tbody.children.length + 1;
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${rowIndex}</td>
    <td><input type="number" step="any" class="x-input" value="${x}" placeholder="X"></td>
    <td><input type="number" step="any" class="y-input" value="${y}" placeholder="Y"></td>
    <td><button class="delete-row-btn" title="删除">✕</button></td>
  `;
  tr.querySelector('.delete-row-btn').addEventListener('click', () => {
    tr.remove();
    updateRowNumbers();
    markDirty();
  });
  tr.querySelectorAll('input').forEach(input => {
    input.addEventListener('input', markDirty);
  });
  tbody.appendChild(tr);
}

function updateRowNumbers() {
  const tbody = document.getElementById('dataTableBody');
  Array.from(tbody.children).forEach((tr, idx) => {
    tr.querySelector('td:first-child').textContent = idx + 1;
  });
}

function clearDataTable() {
  const tbody = document.getElementById('dataTableBody');
  tbody.innerHTML = '';
  for (let i = 0; i < 5; i++) {
    addDataRow();
  }
  currentDatasetId = null;
  currentResultId = null;
  smoothedPoints = [];
  smoothConfig.dataSource = '手动输入';
  smoothConfig.pointCount = 0;
  updateSmoothInfoDisplay();
  clearDirty();
  resetDisplay();
}

function resetDisplay() {
  document.getElementById('metricR2').textContent = '—';
  document.getElementById('metricMSE').textContent = '—';
  document.getElementById('metricRMSE').textContent = '—';
  document.getElementById('metricMAE').textContent = '—';
  document.getElementById('eqFormula').textContent = '等待拟合...';
  document.getElementById('outliersSection').style.display = 'none';

  if (fitChart) {
    fitChart.data.datasets.forEach(ds => ds.data = []);
    fitChart.update();
  }
  if (residualChart) {
    residualChart.data.datasets.forEach(ds => ds.data = []);
    residualChart.update();
  }
}

function getTableData() {
  const tbody = document.getElementById('dataTableBody');
  const points = [];
  Array.from(tbody.children).forEach(tr => {
    const xInput = tr.querySelector('.x-input');
    const yInput = tr.querySelector('.y-input');
    const x = parseFloat(xInput.value);
    const y = parseFloat(yInput.value);
    if (!isNaN(x) && !isNaN(y)) {
      points.push({ x, y });
    }
  });
  return points;
}

function setTableData(points) {
  const tbody = document.getElementById('dataTableBody');
  tbody.innerHTML = '';
  points.forEach(p => {
    addDataRow(p.x, p.y);
  });
}

function loadSampleData() {
  const samples = [
    { x: 1, y: 2.1 },
    { x: 2, y: 3.8 },
    { x: 3, y: 6.2 },
    { x: 4, y: 7.9 },
    { x: 5, y: 10.3 },
    { x: 6, y: 11.8 },
    { x: 7, y: 14.5 },
    { x: 8, y: 25.0 },
    { x: 9, y: 18.2 },
    { x: 10, y: 20.1 }
  ];
  setTableData(samples);
  document.getElementById('datasetName').value = '示例实验数据';
  currentDatasetId = null;
  currentResultId = null;
  smoothedPoints = [];
  smoothConfig.dataSource = '示例数据';
  smoothConfig.pointCount = samples.length;
  updateSmoothInfoDisplay();
  resetDisplay();
  clearDirty();
  showToast('已加载示例数据', 'success');
}

async function performFit() {
  const fitDataSource = document.querySelector('input[name="fitDataSource"]:checked').value;
  const originalPoints = getTableData();

  let points;
  if (fitDataSource === 'smoothed') {
    if (smoothedPoints.length === 0) {
      showToast('请先应用平滑处理', 'error');
      return;
    }
    points = smoothedPoints;
  } else {
    points = originalPoints;
  }

  if (points.length < 2) {
    showToast('请至少输入2个有效数据点', 'error');
    return;
  }

  const modelType = document.querySelector('input[name="modelType"]:checked').value;
  const datasetName = document.getElementById('datasetName').value || '未命名数据集';

  const fitBtn = document.getElementById('fitBtn');
  const originalText = fitBtn.textContent;
  fitBtn.textContent = '⏳ 计算中...';
  fitBtn.disabled = true;

  try {
    const res = await fetch('/api/fit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points, modelType, datasetName, datasetId: currentDatasetId })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || '拟合失败');

    displayFitResult(data, fitDataSource);
    currentResultId = data.id;
    showToast(`拟合完成！（使用${fitDataSource === 'smoothed' ? '平滑' : '原始'}数据）`, 'success');
    loadHistory();
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    fitBtn.textContent = originalText;
    fitBtn.disabled = false;
  }
}

function displayFitResult(result, dataSource = 'original') {
  document.getElementById('metricR2').textContent = result.metrics.rSquared.toFixed(6);
  document.getElementById('metricMSE').textContent = result.metrics.mse.toFixed(6);
  document.getElementById('metricRMSE').textContent = result.metrics.rmse.toFixed(6);
  document.getElementById('metricMAE').textContent = result.metrics.mae.toFixed(6);
  document.getElementById('eqFormula').textContent = result.modelEquation;

  const originalPoints = getTableData();
  const normalPoints = [];
  const outlierPoints = [];
  const outlierIndices = new Set(result.outliers.filter(o => o.isOutlier).map(o => o.index));

  result.points.forEach((p, i) => {
    if (outlierIndices.has(i)) {
      outlierPoints.push(p);
    } else {
      normalPoints.push(p);
    }
  });

  fitChart.data.datasets[0].data = originalPoints;
  fitChart.data.datasets[1].data = smoothedPoints;
  fitChart.data.datasets[2].data = result.curvePoints;
  fitChart.data.datasets[3].data = outlierPoints;
  fitChart.update();

  const residualData = result.points.map((p, i) => ({
    x: p.x,
    y: result.residuals[i]
  }));

  const xs = result.points.map(p => p.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const range = maxX - minX || 1;
  const zeroLine = [
    { x: minX - range * 0.1, y: 0 },
    { x: maxX + range * 0.1, y: 0 }
  ];

  residualChart.data.datasets[0].data = residualData;
  residualChart.data.datasets[1].data = zeroLine;
  residualChart.update();

  const outliersSection = document.getElementById('outliersSection');
  const outliersList = document.getElementById('outliersList');
  const actualOutliers = result.outliers.filter(o => o.isOutlier);

  if (actualOutliers.length > 0) {
    outliersSection.style.display = 'block';
    outliersList.innerHTML = actualOutliers.map(o => `
      <span class="outlier-badge">
        #${o.index + 1} (x=${result.points[o.index].x.toFixed(3)}, y=${result.points[o.index].y.toFixed(3)})
        Z=${o.zScore.toFixed(2)}
      </span>
    `).join('');
  } else {
    outliersSection.style.display = 'none';
  }
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const history = await res.json();
    const historyList = document.getElementById('historyList');

    if (history.length === 0) {
      historyList.innerHTML = '<div class="empty-state">暂无历史记录</div>';
      return;
    }

    historyList.innerHTML = history.map(h => `
      <div class="history-item" data-id="${h.id}">
        <div class="history-title">${h.datasetName}</div>
        <span class="history-model">${modelTypeLabels[h.modelType] || h.modelType}</span>
        <div class="history-meta">
          <span>${h.pointsCount} 个点 · R²=${h.metrics.rSquared.toFixed(4)}</span>
          <span>${new Date(h.createdAt).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div class="history-actions">
          <button class="btn-load" onclick="loadHistoryItem('${h.id}')">查看</button>
          <button class="btn-delete" onclick="deleteHistoryItem('${h.id}')">删除</button>
        </div>
      </div>
    `).join('');
  } catch (err) {
    console.error('加载历史失败:', err);
  }
}

async function loadHistoryItem(id) {
  try {
    const res = await fetch(`/api/history/${id}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    document.getElementById('datasetName').value = data.datasetName;
    document.querySelector(`input[name="modelType"][value="${data.modelType}"]`).checked = true;
    setTableData(data.points);
    smoothedPoints = [];
    smoothConfig.dataSource = '历史记录';
    smoothConfig.pointCount = data.points.length;
    updateSmoothInfoDisplay();
    displayFitResult(data);
    currentResultId = id;
    currentDatasetId = data.datasetId || null;
    clearDirty();
    showToast('已加载历史记录', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function deleteHistoryItem(id) {
  if (!confirm('确定删除这条历史记录吗？')) return;
  try {
    const res = await fetch(`/api/history/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('删除失败');
    if (currentResultId === id) {
      currentResultId = null;
    }
    showToast('已删除', 'success');
    loadHistory();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function loadDatasets() {
  try {
    const res = await fetch('/api/datasets');
    const datasets = await res.json();
    const datasetsList = document.getElementById('datasetsList');

    if (datasets.length === 0) {
      datasetsList.innerHTML = '<div class="empty-state">暂无保存的数据集</div>';
      return;
    }

    datasetsList.innerHTML = datasets.map(d => `
      <div class="dataset-item" data-id="${d.id}">
        <div class="history-title">${d.name}</div>
        <div class="history-meta">
          <span>${d.points.length} 个点</span>
          <span>${new Date(d.createdAt).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div class="history-actions">
          <button class="btn-load" onclick="loadDataset('${d.id}')">加载</button>
          <button class="btn-delete" onclick="deleteDataset('${d.id}')">删除</button>
        </div>
      </div>
    `).join('');
  } catch (err) {
    console.error('加载数据集失败:', err);
  }
}

async function saveCurrentDataset() {
  const points = getTableData();
  const name = document.getElementById('datasetName').value || '未命名数据集';

  if (points.length < 2) {
    showToast('请至少输入2个有效数据点', 'error');
    return;
  }

  try {
    const res = await fetch('/api/datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, points })
    });
    if (!res.ok) throw new Error('保存失败');
    const dataset = await res.json();
    currentDatasetId = dataset.id;
    clearDirty();
    showToast('已另存为新数据集', 'success');
    loadDatasets();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function updateCurrentDataset() {
  if (!currentDatasetId) {
    showToast('没有可更新的数据集，请先加载或另存为', 'error');
    return;
  }

  const points = getTableData();
  const name = document.getElementById('datasetName').value || '未命名数据集';

  if (points.length < 2) {
    showToast('请至少输入2个有效数据点', 'error');
    return;
  }

  try {
    const res = await fetch(`/api/datasets/${currentDatasetId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, points })
    });
    if (!res.ok) throw new Error('更新失败');
    clearDirty();
    showToast('数据集已更新', 'success');
    loadDatasets();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function loadDataset(id) {
  try {
    const res = await fetch('/api/datasets');
    const datasets = await res.json();
    const dataset = datasets.find(d => d.id === id);
    if (!dataset) throw new Error('数据集不存在');

    document.getElementById('datasetName').value = dataset.name;
    setTableData(dataset.points);
    currentDatasetId = id;
    currentResultId = null;
    smoothedPoints = [];
    smoothConfig.dataSource = dataset.name;
    smoothConfig.pointCount = dataset.points.length;
    updateSmoothInfoDisplay();
    resetDisplay();
    clearDirty();
    showToast('已加载数据集', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function deleteDataset(id) {
  if (!confirm('确定删除这个数据集吗？')) return;
  try {
    const res = await fetch(`/api/datasets/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('删除失败');
    if (currentDatasetId === id) {
      currentDatasetId = null;
      updateDatasetButtons();
    }
    showToast('已删除', 'success');
    loadDatasets();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
      document.getElementById('tab-history').style.display = tab === 'history' ? 'block' : 'none';
      document.getElementById('tab-datasets').style.display = tab === 'datasets' ? 'block' : 'none';
    });
  });
}

function initEventListeners() {
  document.getElementById('addRowBtn').addEventListener('click', () => {
    addDataRow();
    markDirty();
  });
  document.getElementById('clearDataBtn').addEventListener('click', () => {
    if (confirm('确定清空所有数据吗？')) clearDataTable();
  });
  document.getElementById('loadSampleBtn').addEventListener('click', loadSampleData);
  document.getElementById('fitBtn').addEventListener('click', performFit);
  document.getElementById('saveDatasetBtn').addEventListener('click', saveCurrentDataset);
  document.getElementById('updateDatasetBtn').addEventListener('click', updateCurrentDataset);
  document.getElementById('datasetName').addEventListener('input', markDirty);

  document.getElementById('applySmoothBtn').addEventListener('click', applySmoothing);

  document.querySelectorAll('input[name="smoothMethod"]').forEach(radio => {
    radio.addEventListener('change', () => {
      const method = radio.value;
      smoothConfig.method = method;
      const polyOrderItem = document.getElementById('polyOrderItem');
      if (method === 'savitzkyGolay') {
        polyOrderItem.style.display = 'flex';
      } else {
        polyOrderItem.style.display = 'none';
      }
      saveSmoothConfig();
    });
  });

  document.getElementById('windowSize').addEventListener('input', (e) => {
    const value = parseInt(e.target.value);
    document.getElementById('windowSizeValue').textContent = value;
    smoothConfig.windowSize = value;
    saveSmoothConfig();
  });

  document.getElementById('polyOrder').addEventListener('input', (e) => {
    const value = parseInt(e.target.value);
    document.getElementById('polyOrderValue').textContent = value;
    smoothConfig.polyOrder = value;
    saveSmoothConfig();
  });

  document.getElementById('enableSmooth').addEventListener('change', (e) => {
    smoothConfig.enabled = e.target.checked;
    saveSmoothConfig();
  });
}

function init() {
  loadSmoothConfig();
  initCharts();
  initTabs();
  initEventListeners();
  applySmoothConfigToUI();
  clearDataTable();
  loadHistory();
  loadDatasets();
  updateDatasetButtons();
}

document.addEventListener('DOMContentLoaded', init);
