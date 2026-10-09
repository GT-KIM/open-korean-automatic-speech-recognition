const OVERALL_TAB = "overall";
const ON_DEVICE_TAB = "on_device";
const ALL_SLICES = "__all_slices__";
const AIHUB_DATASET = "AIHubLowQualityTelephone";
const STANDARD_PROTOCOL = "v1/kspon/cer>1.0";
const OVERALL_SLICES = [
  { dataset: "KsponSpeech", subset: "clean", samples: 3000 },
  { dataset: "KsponSpeech", subset: "other", samples: 3000 },
  { dataset: AIHUB_DATASET, subset: "all", samples: 39916 },
];

// Metrics where a higher value is better (e.g. RTFx = audio/processing speedup).
// Everything else (error rates, latency, outlier rate) is lower-is-better.
const HIGHER_IS_BETTER = new Set(["rtfx"]);

const state = {
  rows: [],
  onDeviceRows: [],
  search: "",
  activeTab: OVERALL_TAB,
  view: "table",
  subsetByDataset: {},
  model: "all",
  sortMetric: "cer",
  expandedKey: null,
  execution: { hardware: "all", batch: "all", precision: "all" },
  comparison: [],
  comparisonOpen: false,
  tradeoff: { slice: "", condition: "all", metric: "cer", selected: null, points: [], configured: false },
};

const metricLabels = {
  cer: "Main CER (outliers excluded)",
  all_samples_cer: "All-sample CER",
  wer: "WER",
  mer: "MER",
  jer: "JER",
  ser: "SER",
  rtfx: "RTFx",
  latency: "Latency",
  outlier_rate: "Outlier rate",
};

const datasetColumns = [
  "Rank",
  "Model",
  "Main CER ↓",
  "Outlier rate ↓",
  "All-sample CER ↓",
  "RTFx ↑",
  "평가 범위",
];

const overallColumns = [
  "Rank",
  "Model",
  "Main CER ↓",
  "Outlier rate ↓",
  "All-sample CER ↓",
  "RTFx ↑",
  "평가 범위",
];

const onDeviceColumns = [
  "Rank",
  "Model",
  "Main CER ↓",
  "Outlier rate ↓",
  "All-sample CER ↓",
  "QNN RTFx ↑",
  "Device",
  "평가 범위",
];

const els = {
  body: document.getElementById("leaderboardBody"),
  head: document.getElementById("leaderboardHead"),
  table: document.getElementById("leaderboardTable"),
  rowCount: document.getElementById("rowCount"),
  overallCoverage: document.getElementById("overallCoverage"),
  references: document.getElementById("referenceResults"),
  referenceSummary: document.getElementById("referenceSummary"),
  referenceHead: document.getElementById("referenceHead"),
  referenceBody: document.getElementById("referenceBody"),
  status: document.getElementById("dataStatus"),
  tabs: document.getElementById("datasetTabs"),
  subsetTabs: document.getElementById("subsetTabs"),
  search: document.getElementById("searchInput"),
  model: document.getElementById("modelFilter"),
  sortMetric: document.getElementById("sortMetric"),
  resultsTitle: document.getElementById("resultsTitle"),
  tableSection: document.getElementById("tableSection"),
  tableView: document.getElementById("tableView"),
  chartView: document.getElementById("chartView"),
  sortControl: document.getElementById("sortControl"),
  resetFilters: document.getElementById("resetFilters"),
  executionFilters: document.getElementById("executionFilters"),
  executionCount: document.getElementById("executionFilterCount"),
  executionScope: document.getElementById("executionScope"),
  hardwareFilter: document.getElementById("hardwareFilter"),
  batchFilter: document.getElementById("batchFilter"),
  batchControl: document.getElementById("batchControl"),
  precisionFilter: document.getElementById("precisionFilter"),
  comparisonToggle: document.getElementById("comparisonToggle"),
  comparisonClear: document.getElementById("comparisonClear"),
  comparisonSection: document.getElementById("comparisonSection"),
  comparisonContent: document.getElementById("comparisonContent"),
  shareButton: document.getElementById("shareButton"),
  shareFeedback: document.getElementById("shareFeedback"),
  shareStatus: document.getElementById("shareStatus"),
  shareUrl: document.getElementById("shareUrl"),
  tradeoffSection: document.getElementById("tradeoffSection"),
  tradeoffSlice: document.getElementById("tradeoffSlice"),
  tradeoffCondition: document.getElementById("tradeoffCondition"),
  tradeoffMetric: document.getElementById("tradeoffMetric"),
  tradeoffSpeed: document.getElementById("tradeoffSpeed"),
  tradeoffStatus: document.getElementById("tradeoffStatus"),
  tradeoffPlot: document.getElementById("tradeoffPlot"),
  tradeoffDetail: document.getElementById("tradeoffDetail"),
  tradeoffModels: document.getElementById("tradeoffModels"),
  tradeoffNote: document.getElementById("tradeoffNote"),
};

document.addEventListener("DOMContentLoaded", () => {
  wireControls();
  loadLeaderboard();
});

function wireControls() {
  els.search.addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    render();
  });
  els.model.addEventListener("change", (event) => {
    state.model = event.target.value;
    state.expandedKey = null;
    render();
  });
  els.sortMetric.addEventListener("change", (event) => {
    state.sortMetric = event.target.value;
    state.expandedKey = null;
    render();
  });
  els.tabs.addEventListener("click", handleTabClick);
  els.subsetTabs.addEventListener("click", handleSubsetClick);
  els.body.addEventListener("click", handleTableClick);
  els.referenceBody.addEventListener("click", handleTableClick);
  els.head.addEventListener("click", handleSortClick);
  els.referenceHead.addEventListener("click", handleSortClick);
  els.tableView.addEventListener("click", () => setView("table"));
  els.chartView.addEventListener("click", () => setView("chart"));
  els.resetFilters.addEventListener("click", resetFilters);
  for (const key of ["hardware", "batch", "precision"]) {
    els[`${key}Filter`].addEventListener("change", (event) => {
      state.execution[key] = event.target.value;
      state.expandedKey = null;
      render();
    });
  }
  els.body.addEventListener("change", handleComparisonChange);
  els.comparisonToggle.addEventListener("click", () => {
    state.comparisonOpen = state.comparison.length >= 2 && !state.comparisonOpen;
    render();
  });
  els.comparisonClear.addEventListener("click", () => {
    state.comparison = [];
    state.comparisonOpen = false;
    render();
  });
  els.shareButton.addEventListener("click", copyShareLink);
  if (typeof window !== "undefined") window.addEventListener("popstate", () => {
    restoreUrlState(window.location.search);
    populateModelFilter();
    renderTabs();
    render();
  });
  document.querySelectorAll('a[href="#evaluation-method"]').forEach((link) => {
    link.addEventListener("click", () => { document.getElementById("evaluation-method").open = true; });
  });
  for (const [element, key] of [[els.tradeoffSlice, "slice"], [els.tradeoffCondition, "condition"], [els.tradeoffMetric, "metric"], [els.tradeoffSpeed, "speed"]]) {
    element.addEventListener("change", (event) => {
      state.tradeoff[key] = event.target.value;
      state.tradeoff.configured = true;
      if (key === "slice" || key === "speed") state.tradeoff.condition = "all";
      renderTradeoff();
      syncUrlState();
    });
  }
  els.tradeoffPlot.addEventListener("click", handleTradeoffPoint);
  els.tradeoffModels.addEventListener("click", handleTradeoffPoint);
  els.tradeoffPlot.addEventListener("keydown", (event) => {
    if ((event.key === "Enter" || event.key === " ") && event.target.closest("[data-tradeoff-point]")) {
      event.preventDefault();
      handleTradeoffPoint(event);
    }
  });
}

function setView(view) {
  if (!["table", "chart"].includes(view)) return;
  state.view = view;
  if (view === "chart") state.tradeoff.configured = true;
  renderView();
  syncUrlState();
}

function renderView() {
  const chart = state.view === "chart";
  els.tableSection.hidden = chart;
  els.tradeoffSection.hidden = !chart;
  els.sortControl.hidden = chart;
  els.tableView.setAttribute("aria-pressed", String(!chart));
  els.chartView.setAttribute("aria-pressed", String(chart));
  els.resetFilters.disabled = !state.search && state.model === "all" && !hasExecutionFilter();
  els.executionScope.hidden = chart || state.activeTab !== OVERALL_TAB || !hasExecutionFilter();
  els.comparisonSection.hidden = chart || !state.comparisonOpen || state.comparison.length < 2;
}

function resetFilters() {
  state.search = "";
  state.model = "all";
  state.execution = { hardware: "all", batch: "all", precision: "all" };
  state.expandedKey = null;
  els.search.value = "";
  els.model.value = "all";
  render();
}

function handleSortClick(event) {
  const button = event.target.closest("[data-sort-metric]");
  if (!button) return;
  const metric = button.getAttribute("data-sort-metric");
  if (!Object.hasOwn(metricLabels, metric)) return;
  state.sortMetric = metric;
  state.expandedKey = null;
  els.sortMetric.value = metric;
  render();
  // Header rendering replaces the focused button; keep keyboard users in place.
  event.currentTarget.querySelector(`[data-sort-metric="${metric}"]`)?.focus();
}

async function loadLeaderboard() {
  try {
    const [serverResponse, onDeviceResponse] = await Promise.all([
      fetch("leaderboard_data.json", { cache: "no-store" }),
      fetch("ondevice_leaderboard_data.json", { cache: "no-store" }),
    ]);
    if (!serverResponse.ok || !onDeviceResponse.ok) {
      throw new Error(`HTTP ${serverResponse.status}/${onDeviceResponse.status}`);
    }
    const [serverData, onDeviceData] = await Promise.all([
      serverResponse.json(),
      onDeviceResponse.json(),
    ]);
    state.rows = Array.isArray(serverData) ? serverData.map(normalizeRow) : [];
    state.rows = state.rows.filter((row) => row.is_full_evaluation !== false);
    state.onDeviceRows = Array.isArray(onDeviceData) ? onDeviceData.map(normalizeRow) : [];
    state.onDeviceRows = state.onDeviceRows.filter((row) => row.is_full_evaluation !== false);
    if (typeof window !== "undefined") restoreUrlState(window.location.search);
    populateModelFilter();
    renderTabs();
    render();
    els.status.textContent = "";
    els.status.hidden = true;
  } catch (error) {
    els.status.textContent = "데이터 로드 실패";
    els.status.hidden = false;
    els.rowCount.textContent = "";
    els.body.innerHTML =
      `<tr><td colspan="${datasetColumns.length}" class="empty-state error-state">평가 결과를 불러오지 못했습니다.</td></tr>`;
    console.error(error);
  }
}

function populateModelFilter() {
  const sourceRows = state.activeTab === ON_DEVICE_TAB ? state.onDeviceRows : state.rows;
  const models = modelOptions(sourceRows);
  if (state.model !== "all" && !models.some((model) => model.value === state.model)) {
    state.model = "all";
  }
  fillSelect(els.model, "all", "전체 모델", models);
  els.model.value = state.model;
}

function fillSelect(select, allValue, allLabel, options) {
  select.innerHTML = [
    `<option value="${escapeAttr(allValue)}">${escapeHtml(allLabel)}</option>`,
    ...options.map(({value, label}) => `<option value="${escapeAttr(value)}">${escapeHtml(label)}</option>`),
  ].join("");
}

function canonicalModelId(row) {
  const repo = String(row.model_repo || "").trim()
    .replace(/^https?:\/\/(?:www\.)?huggingface\.co\//i, "")
    .replace(/\/+$/, "");
  return repo ? repo.toLowerCase() : `name:${row.model}`;
}

// Prefer artifacts and the latest dated run; never select by score.
function compareRepresentativeRuns(a, b) {
  const isArtifact = (row) => {
    const path = String(row._artifact || "").replaceAll("\\", "/");
    return path.endsWith("/leaderboard_row.json") || path.startsWith("results/");
  };
  const timestamp = (row) => String(row.run_id || "").match(/\d{8}T\d{6}\d*Z/)?.[0]
    .replace(/Z$/, "").padEnd(21, "0") || "";
  return Number(isArtifact(b)) - Number(isArtifact(a))
    || timestamp(b).localeCompare(timestamp(a))
    || String(b.run_id || "").localeCompare(String(a.run_id || ""))
    || String(a.model).localeCompare(String(b.model));
}

function modelOptions(rows) {
  const models = new Map();
  for (const row of [...rows].sort(compareRepresentativeRuns)) {
    const id = canonicalModelId(row);
    if (row.model && !models.has(id)) {
      models.set(id, {value: id, label: displayModelName(row.model)});
    }
  }
  return [...models.values()].sort((a, b) => a.label.localeCompare(b.label) || a.value.localeCompare(b.value));
}

function renderTabs() {
  const datasets = uniqueSorted(state.rows.map((row) => row.dataset).filter(Boolean));
  const tabs = [
    {
      key: OVERALL_TAB,
      label: "Overall",
      count: buildOverallLeaderboard().ranked.length,
    },
    ...datasets.map((dataset) => ({
      key: dataset,
      label: displayDatasetName(dataset),
      count: state.rows.filter((row) => row.dataset === dataset).length,
    })),
    {
      key: ON_DEVICE_TAB,
      label: "On-device",
      count: state.onDeviceRows.length,
    },
  ];

  els.tabs.innerHTML = tabs
    .map(
      (tab) => `
        <button
          class="tab-button ${state.activeTab === tab.key ? "active" : ""}"
          type="button"
          data-tab="${escapeAttr(tab.key)}"
          aria-pressed="${state.activeTab === tab.key}"
        >
          <span>${escapeHtml(tab.label)}</span>
          <b>${tab.count}</b>
        </button>`,
    )
    .join("");
}

function handleTabClick(event) {
  const button = event.target.closest("[data-tab]");
  if (!button) {
    return;
  }
  state.activeTab = button.getAttribute("data-tab");
  state.comparison = [];
  state.comparisonOpen = false;
  state.expandedKey = null;
  els.references.open = false;
  populateModelFilter();
  renderTabs();
  renderSubsetTabs();
  render();
}

function handleSubsetClick(event) {
  const button = event.target.closest("[data-subset]");
  if (!button || state.activeTab === OVERALL_TAB || state.activeTab === ON_DEVICE_TAB) {
    return;
  }
  state.subsetByDataset[state.activeTab] = button.getAttribute("data-subset");
  state.comparison = [];
  state.comparisonOpen = false;
  state.expandedKey = null;
  renderSubsetTabs();
  render();
}

function renderSubsetTabs() {
  if (state.activeTab === OVERALL_TAB || state.activeTab === ON_DEVICE_TAB) {
    els.subsetTabs.innerHTML = "";
    els.subsetTabs.hidden = true;
    return;
  }

  const datasetRows = state.rows.filter((row) => row.dataset === state.activeTab);
  const subsets = uniqueSorted(datasetRows.map((row) => row.subset || "default")).sort(compareSubsets);
  if (subsets.length <= 1) {
    els.subsetTabs.innerHTML = "";
    els.subsetTabs.hidden = true;
    return;
  }

  const activeSubset = activeDatasetSubset();
  const buttons = [
    { key: ALL_SLICES, label: "All subsets", count: datasetRows.length },
    ...subsets.map((subset) => ({
      key: subset,
      label: displaySubsetName(subset),
      count: datasetRows.filter((row) => (row.subset || "default") === subset).length,
    })),
  ];

  els.subsetTabs.hidden = false;
  els.subsetTabs.innerHTML = buttons
    .map(
      (button) => `
        <button
          class="subset-button ${activeSubset === button.key ? "active" : ""}"
          type="button"
          data-subset="${escapeAttr(button.key)}"
          aria-pressed="${activeSubset === button.key}"
        >
          <span>${escapeHtml(button.label)}</span>
          <b>${button.count}</b>
        </button>`,
    )
    .join("");
}

function render() {
  renderSubsetTabs();
  renderExecutionFilters();
  normalizeComparison();
  els.overallCoverage.hidden = state.activeTab !== OVERALL_TAB;
  els.references.hidden = true;
  if (state.activeTab === OVERALL_TAB) {
    renderOverall();
  } else if (state.activeTab === ON_DEVICE_TAB) {
    renderOnDevice();
  } else {
    renderDataset();
  }
  renderTradeoff();
  renderComparison();
  renderView();
  syncUrlState();
}

const UNKNOWN_EXECUTION = "__unknown__";

function executionValues(row, key) {
  if (state.activeTab === ON_DEVICE_TAB) {
    return [String((key === "hardware" ? row.device : key === "precision" ? row.precision : null) || UNKNOWN_EXECUTION)];
  }
  const metadata = row.reproducibility || {};
  const records = Array.isArray(metadata.source_runs)
    ? metadata.source_runs.map((source) => source.reproducibility || {}) : [metadata];
  const values = records.map((record) => {
    if (key === "hardware") return record.environment?.gpu || row.gpu;
    if (key === "precision") return serverPrecision(record, row);
    const batch = record.execution?.batch_size;
    return Number.isInteger(batch) && batch > 0 ? String(batch) : null;
  });
  return [...new Set((values.length ? values : [null]).map((value) => String(value || UNKNOWN_EXECUTION)))];
}

function matchesExecution(row) {
  return Object.entries(state.execution).every(([key, selected]) => selected === "all"
    || (key === "batch" && state.activeTab === ON_DEVICE_TAB)
    || executionValues(row, key).every((value) => value === selected));
}

function hasExecutionFilter() {
  return Object.values(state.execution).some((value) => value !== "all");
}

function renderExecutionFilters() {
  const device = state.activeTab === ON_DEVICE_TAB;
  let rows;
  if (device) rows = state.onDeviceRows;
  else if (state.activeTab === OVERALL_TAB) {
    const overall = buildOverallLeaderboard();
    rows = [...overall.ranked, ...overall.incomplete].flatMap((row) => row.rows.length ? row.rows : [row.representative_run]);
  } else rows = state.rows.filter((row) => row.dataset === state.activeTab
    && (activeDatasetSubset() === ALL_SLICES || (row.subset || "default") === activeDatasetSubset()));
  for (const key of ["hardware", "batch", "precision"]) {
    const values = [...new Set(rows.flatMap((row) => executionValues(row, key)))].sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
    if (!values.includes(state.execution[key]) || (device && key === "batch")) state.execution[key] = "all";
    fillSelect(els[`${key}Filter`], "all", "전체", values.map((value) => ({value, label: value === UNKNOWN_EXECUTION ? "미확인" : value})));
    els[`${key}Filter`].value = state.execution[key];
  }
  els.batchControl.hidden = device;
  const count = Object.values(state.execution).filter((value) => value !== "all").length;
  els.executionCount.textContent = count ? String(count) : "";
  if (count) els.executionFilters.open = true;
}

function comparisonKey(row) {
  return state.activeTab === OVERALL_TAB ? row.key : `${state.activeTab === ON_DEVICE_TAB ? "device" : "run"}:${row.run_id}`;
}

function comparisonCandidates() {
  if (state.activeTab === OVERALL_TAB) return filterOverallRows(buildOverallLeaderboard().ranked);
  if (state.activeTab === ON_DEVICE_TAB) return filterOnDeviceRows(state.onDeviceRows);
  return filterDatasetRows(state.rows, state.activeTab).filter(hasStandardProtocol);
}

function normalizeComparison() {
  const keys = new Set(comparisonCandidates().map(comparisonKey));
  state.comparison = [...new Set(state.comparison)].filter((key) => keys.has(key)).slice(0, 4);
  if (state.comparison.length < 2) state.comparisonOpen = false;
}

function renderComparisonCheckbox(row) {
  const key = comparisonKey(row);
  const checked = state.comparison.includes(key);
  const label = [displayModelName(row.model), row.key ? "Overall" : compactDatasetLabel(row), row.precision].filter(Boolean).join(" · ");
  return `<input class="compare-checkbox" type="checkbox" data-compare-key="${escapeAttr(key)}"
    aria-label="${escapeAttr(label)} 비교 선택"${checked ? " checked" : state.comparison.length >= 4 ? " disabled" : ""} />`;
}

function handleComparisonChange(event) {
  const input = event.target.closest("input[data-compare-key]");
  if (!input) return;
  const key = input.getAttribute("data-compare-key");
  if (!comparisonCandidates().some((row) => comparisonKey(row) === key)) return;
  if (input.checked && !state.comparison.includes(key) && state.comparison.length < 4) state.comparison.push(key);
  if (!input.checked) state.comparison = state.comparison.filter((selected) => selected !== key);
  render();
  [...els.body.querySelectorAll("input[data-compare-key]")].find((checkbox) => checkbox.getAttribute("data-compare-key") === key)?.focus();
}

function renderComparison() {
  const count = state.comparison.length;
  els.comparisonToggle.textContent = `${state.comparisonOpen ? "비교 닫기" : "선택 비교"} (${count}/4)`;
  els.comparisonToggle.disabled = count < 2;
  els.comparisonToggle.setAttribute("aria-expanded", String(state.comparisonOpen));
  els.comparisonClear.hidden = !count;
  if (count < 2 || !state.comparisonOpen) {
    els.comparisonContent.innerHTML = "";
    return;
  }
  const candidates = new Map(comparisonCandidates().map((row) => [comparisonKey(row), row]));
  const selected = state.comparison.map((key) => candidates.get(key));
  const overall = state.activeTab === OVERALL_TAB;
  const device = state.activeTab === ON_DEVICE_TAB;
  const runs = selected.map((row) => overall ? row.rows : [row]);
  const slices = new Map(runs.flat().map((row) => [tradeoffSliceKey(row), compactDatasetLabel(row)]));
  const metricRows = [["cer", "Main CER ↓"], ["all_samples_cer", "All-sample CER ↓"],
    ["outlier_rate", "Outlier rate ↓"], ["rtfx", device ? "QNN RTFx ↑" : "RTFx ↑"]];
  const rowHtml = (label, values, numeric = false) => `<tr><th scope="row">${escapeHtml(label)}</th>${values.map((value) =>
    `<td${numeric ? ' class="numeric"' : ""}>${escapeHtml(value)}</td>`).join("")}</tr>`;
  const groupHtml = (label, note = "") => `<tr class="comparison-group"><th colspan="${count + 1}">${escapeHtml(label)}${note ? `<span>${escapeHtml(note)}</span>` : ""}</th></tr>`;
  let content = "";
  if (overall) {
    content += groupHtml("Overall · 3개 구간 동일 가중 평균");
    content += metricRows.map(([metric, label]) => rowHtml(label, selected.map((row) => formatMetric(metric, row.metrics[metric])), true)).join("");
  }
  for (const [key, label] of slices) {
    const sliceRuns = runs.map((rows) => rows.find((row) => tradeoffSliceKey(row) === key));
    const conditions = sliceRuns.filter(Boolean).map(tradeoffCondition);
    const notes = [];
    if (new Set(conditions.map((condition) => condition.key)).size > 1) notes.push("실행 조건 다름");
    if (conditions.some((condition) => !condition.complete)) notes.push("실행 조건 미확인·혼합");
    content += groupHtml(label, notes.join(" · "));
    content += metricRows.map(([metric, title]) => rowHtml(title, sliceRuns.map((row) => row
      ? formatMetric(metric, metric === "outlier_rate" ? outlierRate(row) : metricValue(row, metric)) : "—"), true)).join("");
    content += rowHtml("샘플 수", sliceRuns.map((row) => row ? formatInteger(row.evaluated_samples) : "—"), true);
    content += rowHtml("실행 조건", sliceRuns.map((row) => row ? tradeoffCondition(row).label : "—"));
    content += rowHtml("정밀도", sliceRuns.map((row) => row ? executionValues(row, "precision").map((value) => value === UNKNOWN_EXECUTION ? "미확인" : value).join(" / ") : "—"));
  }
  els.comparisonContent.innerHTML = `<table class="comparison-table" style="--comparison-columns: ${count}"><thead><tr><th scope="col">지표 / 조건</th>${selected.map((row) =>
    `<th scope="col"><span class="model-name">${escapeHtml(displayModelName(row.model))}</span><span class="model-repo">${escapeHtml(overall ? row.model_repo : [compactDatasetLabel(row), row.precision].filter(Boolean).join(" · "))}</span></th>`).join("")}</tr></thead><tbody>${content}</tbody></table>${device ? '<p class="detail-note">속도 측정 범위: QNN 그래프 실행 (전처리·전송·토큰화 제외)</p>' : ""}`;
}

const URL_STATE_KEYS = ["tab", "view", "subset", "q", "model", "sort", "hardware", "batch", "precision",
  "slice", "cohort", "metric", "speed", "point", "compare", "compareOpen", "detail"];

function restoreUrlState(search) {
  const params = new URLSearchParams(search);
  const read = (key, limit = 512) => (params.get(key) || "").slice(0, limit);
  const tabs = [OVERALL_TAB, ON_DEVICE_TAB, ...state.rows.map((row) => row.dataset)];
  state.activeTab = tabs.includes(read("tab")) ? read("tab") : OVERALL_TAB;
  state.view = read("view") === "chart" ? "chart" : "table";
  state.search = read("q", 200).trim().toLowerCase();
  state.model = read("model") || "all";
  state.sortMetric = Object.hasOwn(metricLabels, read("sort")) ? read("sort") : "cer";
  state.subsetByDataset = {};
  const subsets = state.rows.filter((row) => row.dataset === state.activeTab).map((row) => row.subset || "default");
  if ([ALL_SLICES, ...subsets].includes(read("subset"))) state.subsetByDataset[state.activeTab] = read("subset");
  state.execution = Object.fromEntries(["hardware", "batch", "precision"].map((key) => [key, read(key) || "all"]));
  state.comparison = [...new Set(params.getAll("compare").filter((key) => key.length <= 512))].slice(0, 4);
  state.comparisonOpen = read("compareOpen") === "1";
  const expandable = state.activeTab === OVERALL_TAB ? buildOverallLeaderboard().ranked.map((row) => row.key)
    : (state.activeTab === ON_DEVICE_TAB ? state.onDeviceRows : state.rows.filter((row) => row.dataset === state.activeTab)).map((row) => row.run_id);
  state.expandedKey = expandable.includes(read("detail")) ? read("detail") : null;
  state.tradeoff = {slice: read("slice"), condition: read("cohort", 8192) || "all",
    metric: read("metric") === "all_samples_cer" ? "all_samples_cer" : "cer", selected: read("point") || null, points: [],
    speed: ["b1", "b4"].includes(read("speed")) ? read("speed") : "rtfx",
    configured: ["slice", "cohort", "metric", "speed", "point"].some((key) => params.has(key))};
  els.search.value = state.search;
  els.sortMetric.value = state.sortMetric;
}

function workspaceUrl(href) {
  const url = new URL(href);
  for (const key of URL_STATE_KEYS) url.searchParams.delete(key);
  const put = (key, value, fallback = "") => {
    if (value != null && value !== fallback) url.searchParams.set(key, value);
  };
  put("tab", state.activeTab, OVERALL_TAB);
  put("view", state.view, "table");
  if (![OVERALL_TAB, ON_DEVICE_TAB].includes(state.activeTab)) put("subset", activeDatasetSubset(), defaultSubsetForDataset(state.activeTab));
  put("q", state.search);
  put("model", state.model, "all");
  put("sort", state.sortMetric, "cer");
  for (const [key, value] of Object.entries(state.execution)) put(key, value, "all");
  for (const key of state.comparison) url.searchParams.append("compare", key);
  if (state.comparisonOpen) put("compareOpen", "1");
  put("detail", state.expandedKey);
  const chart = state.tradeoff;
  // Keep a default table link short, while retaining a configured chart across view switches.
  if (state.view === "chart" || chart.configured || chart.condition !== "all" || chart.metric !== "cer"
    || chart.selected !== (chart.points[0]?.row.run_id || null)) {
    put("slice", chart.slice);
    put("cohort", chart.condition, "all");
    put("metric", chart.metric, "cer");
    put("speed", chart.speed, "rtfx");
    put("point", chart.selected);
  }
  return url.href;
}

function syncUrlState() {
  if (typeof window === "undefined") return;
  const url = workspaceUrl(window.location.href);
  if (url !== window.location.href) {
    window.history.replaceState(null, "", url);
    els.shareFeedback.hidden = true;
  }
}

async function copyShareLink() {
  syncUrlState();
  const url = window.location.href;
  try {
    await navigator.clipboard.writeText(url);
    if (window.location.href !== url) return;
    els.shareStatus.textContent = "링크 복사됨";
    els.shareUrl.hidden = true;
  } catch {
    if (window.location.href !== url) return;
    els.shareStatus.textContent = "링크를 복사하세요.";
    els.shareUrl.value = url;
    els.shareUrl.hidden = false;
  }
  els.shareFeedback.hidden = false;
  if (!els.shareUrl.hidden) { els.shareUrl.focus(); els.shareUrl.select(); }
}

function tradeoffSliceKey(row) {
  return JSON.stringify([row.dataset, row.subset || "default"]);
}

// Compare recorded configurations, never infer missing hardware or batch sizes.
function serverPrecision(record, row) {
  // A failed observation must not fall back to the requested load dtype.
  if (record.inference && "effective_dtype" in record.inference) return record.inference.effective_dtype;
  return record.model_config?.torch_dtype || record.model_config?.dtype || row.precision;
}

function tradeoffCondition(row) {
  const policy = [row.normalization_preset || null, row.outlier_policy?.metric, row.outlier_policy?.threshold];
  let dimensions, label, complete;
  if (state.activeTab === ON_DEVICE_TAB) {
    dimensions = [row.device, row.soc, row.accelerator, row.backend, row.runtime, row.os_abi,
      row.precision, row.performance_scope];
    complete = dimensions.every(Boolean) && Boolean(row.outlier_policy?.metric);
    label = [row.device || "기기 미확인", row.runtime || "런타임 미확인", row.precision || "정밀도 미확인"].join(" · ");
  } else {
    const metadata = row.reproducibility || {};
    const records = metadata.source_runs ? metadata.source_runs.map((run) => run.reproducibility || {}) : [metadata];
    const configurations = records.map((record) => {
      const env = record.environment || {};
      const execution = record.execution || {};
      return [env.gpu || row.gpu || null, env.torch || row.torch || null,
        env.cuda || row.cuda || null, execution.batch_size ?? null,
        env.platform || null, execution.num_workers ?? null, execution.warmup_samples ?? null,
        serverPrecision(record, row) || null,
        record.inference?.backend || row.backend || null,
        record.inference?.performance_scope || row.performance_scope || null,
        record.inference?.backend_batch_size ?? null,
        execution.warmup_mode || null,
        env.packages?.transformers || null, env.packages?.["qwen-asr"] || null];
    });
    const signatures = [...new Set(configurations.map((configuration) => JSON.stringify(configuration)))].sort();
    dimensions = [signatures, row.backend || null, row.performance_scope || null];
    complete = configurations.length > 0 && signatures.length === 1
      && configurations.every(([gpu, torch, cuda, batch, platform, workers, warmup, precision, backend, scope, innerBatch]) =>
        [gpu, torch, cuda, platform, precision, backend, scope].every((value) => typeof value === "string" && value.trim()
          && !["unknown", "n/a", "미확인"].includes(value.trim().toLowerCase()))
        && Number.isInteger(batch) && batch > 0 && Number.isInteger(workers) && workers >= 0
        && Number.isInteger(warmup) && warmup >= 0
        && Number.isInteger(innerBatch) && innerBatch > 0);
    const first = configurations[0] || [];
    label = signatures.length > 1 ? `${row.gpu || "기기 미확인"} · 구간별 조건 혼합`
      : `${first[0] || "GPU 미확인 / API"} · batch ${first[3] ?? "미확인"} · PyTorch ${first[1] || "미확인"}`;
  }
  return {
    key: JSON.stringify([dimensions, policy, row.evaluated_samples]),
    label, complete,
  };
}

function tradeoffSourceRows() {
  let rows;
  if (state.activeTab === ON_DEVICE_TAB) rows = state.onDeviceRows;
  else {
    rows = state.rows.filter(hasStandardProtocol);
    if (state.activeTab === OVERALL_TAB) {
      rows = rows.filter((row) => OVERALL_SLICES.some((slice) => row.dataset === slice.dataset && row.subset === slice.subset));
    } else {
      rows = rows.filter((row) => row.dataset === state.activeTab
        && (activeDatasetSubset() === ALL_SLICES || (row.subset || "default") === activeDatasetSubset()));
    }
  }
  return rows.filter((row) => row.is_full_evaluation === true && matchesExecution(row));
}

function tradeoffRepresentatives(rows, speed = "rtfx") {
  const selected = new Map();
  for (const row of [...rows].sort(compareRepresentativeRuns)) {
    const key = JSON.stringify([canonicalModelId(row), tradeoffSliceKey(row), speedCondition(row, speed).key]);
    if (!selected.has(key)) selected.set(key, row);
  }
  return [...selected.values()];
}

function speedCondition(row, speed = "rtfx") {
  if (speed === "rtfx") return tradeoffCondition(row);
  const measured = row.curated_speed;
  const track = measured?.tracks?.[speed];
  const dimensions = [measured?.environment_id, measured?.image_id, measured?.source_sha256,
    measured?.executed_protocol_sha256, measured?.group, measured?.samples_per_repeat, measured?.repetitions, track?.batch_size];
  return {key: JSON.stringify(["curated", dimensions, row.evaluation_protocol, row.evaluated_samples]),
    complete: measured?.status === "verified" && dimensions.every(Boolean),
    label: `${measured?.gpu || "GPU 미확인"} · BF16 · batch ${track?.batch_size ?? "미확인"} · curated 256 × 3`};
}

function tradeoffSpeedValue(row) {
  const speed = state.tradeoff.speed || "rtfx";
  return speed === "rtfx" ? metricValue(row, "rtfx") : row.curated_speed?.tracks?.[speed]?.throughput_rtfx?.median;
}

function tradeoffSpeedLabel() {
  if (state.activeTab === ON_DEVICE_TAB) return "QNN RTFx";
  return "RTFx";
}

// Equality in both coordinates is a tie; improvement in one is required to dominate.
function paretoFrontier(points) {
  return points.filter((point) => !points.some((other) => other.speed >= point.speed && other.error <= point.error
    && (other.speed > point.speed || other.error < point.error)))
    .sort((a, b) => a.speed - b.speed || a.error - b.error);
}

function renderTradeoff() {
  if (!els.tradeoffSection) return;
  const chart = state.tradeoff;
  const source = tradeoffSourceRows();
  const hasCurated = state.activeTab !== ON_DEVICE_TAB && source.some((row) => row.curated_speed?.status === "verified");
  if (!chart.configured && hasCurated) chart.speed = "b4";
  if (!hasCurated || !["rtfx", "b1", "b4"].includes(chart.speed)) chart.speed = "rtfx";
  els.tradeoffSpeed.innerHTML = `${hasCurated ? '<option value="b4">RTFx · batch 4 · curated</option><option value="b1">RTFx · batch 1 · curated</option>' : ""}<option value="rtfx">${state.activeTab === ON_DEVICE_TAB ? "QNN RTFx" : "RTFx · 전체 평가 · outlier 제외"}</option>`;
  els.tradeoffSpeed.value = chart.speed;
  els.tradeoffSpeed.disabled = !hasCurated;
  const slices = new Map(source.map((row) => [tradeoffSliceKey(row), compactDatasetLabel(row)]));
  const orderedSlices = [...slices].sort(([a], [b]) => {
    const [datasetA, subsetA] = JSON.parse(a), [datasetB, subsetB] = JSON.parse(b);
    return (datasetA === "KsponSpeech" ? 0 : 1) - (datasetB === "KsponSpeech" ? 0 : 1)
      || datasetA.localeCompare(datasetB) || compareSubsets(subsetA, subsetB);
  });
  if (!slices.has(chart.slice)) chart.slice = orderedSlices[0]?.[0] || "";
  els.tradeoffSlice.innerHTML = orderedSlices.map(([key, label]) => `<option value="${escapeAttr(key)}">${escapeHtml(label)}</option>`).join("");
  els.tradeoffSlice.value = chart.slice;
  els.tradeoffSlice.disabled = orderedSlices.length <= 1;
  const sliceRows = tradeoffRepresentatives(source.filter((row) => tradeoffSliceKey(row) === chart.slice
    && (chart.speed === "rtfx" || row.curated_speed?.status === "verified")), chart.speed);
  const groups = new Map();
  for (const row of sliceRows) {
    const condition = speedCondition(row, chart.speed);
    const group = groups.get(condition.key) || {...condition, count: 0};
    group.count++;
    groups.set(group.key, group);
  }
  const conditions = [...groups.values()].sort((a, b) => b.count - a.count || a.key.localeCompare(b.key));
  if (!groups.has(chart.condition)) chart.condition = "all";
  fillSelect(els.tradeoffCondition, "all", "모든 실행 조건", conditions.map((group, index) => ({
    value: group.key, label: `${index + 1}. ${group.label} · ${group.count}개 결과`,
  })));
  els.tradeoffCondition.value = chart.condition;
  els.tradeoffMetric.value = chart.metric;
  const candidates = sliceRows.filter((row) => (chart.condition === "all" || speedCondition(row, chart.speed).key === chart.condition)
    && (state.model === "all" || canonicalModelId(row) === state.model)
    && (!state.search || searchText(row).includes(state.search)));
  chart.points = candidates.map((row) => ({row, speed: tradeoffSpeedValue(row), error: metricValue(row, chart.metric)}))
    .filter((point) => Number.isFinite(point.speed) && point.speed > 0 && Number.isFinite(point.error) && point.error >= 0)
    .sort((a, b) => displayModelName(a.row.model).localeCompare(displayModelName(b.row.model))
      || String(a.row.run_id).localeCompare(String(b.row.run_id)));
  const canCompare = chart.points.length >= 2 && groups.get(chart.condition)?.complete === true;
  const frontier = canCompare ? paretoFrontier(chart.points) : [];
  chart.points.forEach((point) => { point.frontier = frontier.includes(point); });
  if (!chart.points.some((point) => point.row.run_id === chart.selected)) chart.selected = chart.points[0]?.row.run_id || null;
  const omitted = candidates.length - chart.points.length;
  els.tradeoffStatus.textContent = omitted ? `CER 또는 유효한 RTFx가 없는 ${omitted}개 결과 제외` : "";
  els.tradeoffStatus.hidden = !omitted;
  els.tradeoffNote.textContent = canCompare ? "파레토 경계"
    : chart.condition !== "all" && chart.points.length && !groups.get(chart.condition)?.complete
      ? "실행 조건 미확인·혼합: 파레토 비교 제외" : "";
  els.tradeoffNote.className = canCompare ? "tradeoff-note tradeoff-legend" : "tradeoff-note";
  els.tradeoffNote.hidden = !els.tradeoffNote.textContent;
  els.tradeoffPlot.innerHTML = chart.points.length ? renderTradeoffSvg(chart.points, frontier, chart.metric)
    : '<p class="empty-state">조건에 맞는 CER·RTFx 결과가 없습니다.</p>';
  els.tradeoffModels.innerHTML = chart.points.map((point, index) => `
    <button type="button" class="tradeoff-model${point.frontier ? " on-frontier" : ""}${point.row.run_id === chart.selected ? " selected" : ""}"
      data-tradeoff-point="${index}" aria-pressed="${point.row.run_id === chart.selected}"${point.frontier ? ' aria-description="파레토 경계"' : ""}>
      <span class="tradeoff-number">${index + 1}</span><span>${escapeHtml(displayModelName(point.row.model))}
      <small>${formatPercent(point.error)} · ${formatMetric("rtfx", point.speed)}</small></span>
    </button>`).join("");
  renderTradeoffDetail();
}

function renderTradeoffSvg(points, frontier, metric) {
  const width = 760, height = 380, left = 72, right = 28, top = 42, bottom = 64;
  const speeds = points.map((point) => Math.log10(point.speed));
  let minX = Math.min(...speeds), maxX = Math.max(...speeds);
  const pad = Math.max((maxX - minX) * 0.12, 0.12);
  minX -= pad; maxX += pad;
  const maxError = Math.max(0.05, ...points.map((point) => point.error)) * 1.15;
  const rawStep = maxError / 5;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const step = [1, 2, 5, 10].find((factor) => factor * magnitude >= rawStep) * magnitude;
  const maxY = Math.ceil(maxError / step) * step;
  const x = (speed) => left + (Math.log10(speed) - minX) / (maxX - minX) * (width - left - right);
  const y = (error) => height - bottom - error / maxY * (height - top - bottom);
  const xTicks = [];
  for (let power = Math.floor(minX); power <= Math.ceil(maxX); power++) {
    for (const multiplier of [1, 2, 5]) {
      const value = multiplier * 10 ** power;
      if (Math.log10(value) >= minX && Math.log10(value) <= maxX) xTicks.push(value);
    }
  }
  if (xTicks.length < 2) xTicks.splice(0, xTicks.length, 10 ** minX, 10 ** ((minX + maxX) / 2), 10 ** maxX);
  const tickLabel = (value) => Number(value.toPrecision(3)).toString();
  const gridX = xTicks.map((value) => `<line x1="${x(value)}" y1="${top}" x2="${x(value)}" y2="${height - bottom}"/>
    <text x="${x(value)}" y="${height - bottom + 24}" text-anchor="middle">${tickLabel(value)}×</text>`).join("");
  const gridY = Array.from({length: Math.round(maxY / step) + 1}, (_, index) => index * step).map((value) => `
    <line x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}"/>
    <text x="${left - 12}" y="${y(value) + 4}" text-anchor="end">${tickLabel(value * 100)}%</text>`).join("");
  const label = metric === "cer" ? "Main CER" : "All-sample CER";
  const speedLabel = tradeoffSpeedLabel();
  const line = frontier.length >= 2 ? `<polyline class="tradeoff-frontier" points="${frontier.map((point) => `${x(point.speed)},${y(point.error)}`).join(" ")}"/>` : "";
  const markers = points.map((point, index) => {
    const selected = point.row.run_id === state.tradeoff.selected;
    const description = `${displayModelName(point.row.model)} · ${label} ${formatPercent(point.error)} · ${speedLabel} ${formatMetric("rtfx", point.speed)}${point.frontier ? " · 파레토 경계" : ""}`;
    return `<g class="tradeoff-point${point.frontier ? " on-frontier" : ""}${selected ? " selected" : ""}" transform="translate(${x(point.speed)},${y(point.error)})"
      data-tradeoff-point="${index}" role="button" tabindex="0" aria-pressed="${selected}" aria-label="${escapeAttr(description)}">
      <title>${escapeHtml(description)}</title><circle class="point-halo" r="16"/><circle class="point-dot" r="10"/>
      <text text-anchor="middle" dy="4">${index + 1}</text></g>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="group" aria-labelledby="tradeoffSvgTitle tradeoffSvgDesc">
    <title id="tradeoffSvgTitle">${label}–${speedLabel} 산점도</title>
    <desc id="tradeoffSvgDesc">오른쪽은 빠른 속도, 아래쪽은 낮은 오류율입니다. 각 점은 키보드로 선택할 수 있으며 아래 모델 목록에서도 같은 결과를 확인할 수 있습니다.</desc>
    <g class="tradeoff-grid">${gridX}${gridY}</g>
    <text class="tradeoff-axis" x="${left}" y="22">${label} (%) ↓</text>
    <text class="tradeoff-axis" x="${width - right}" y="${height - 12}" text-anchor="end">${speedLabel} (×)</text>
    ${line}${markers}</svg>`;
}

function handleTradeoffPoint(event) {
  const target = event.target.closest("[data-tradeoff-point]");
  if (!target) return;
  const point = state.tradeoff.points[Number(target.getAttribute("data-tradeoff-point"))];
  if (!point) return;
  state.tradeoff.selected = point.row.run_id;
  for (const parent of [els.tradeoffPlot, els.tradeoffModels]) {
    for (const element of parent.querySelectorAll("[data-tradeoff-point]")) {
      const selected = element.getAttribute("data-tradeoff-point") === target.getAttribute("data-tradeoff-point");
      element.classList.toggle("selected", selected);
      element.setAttribute("aria-pressed", String(selected));
    }
  }
  renderTradeoffDetail();
  syncUrlState();
}

function renderTradeoffDetail() {
  const point = state.tradeoff.points.find((candidate) => candidate.row.run_id === state.tradeoff.selected);
  els.tradeoffDetail.hidden = !point;
  if (!point) {
    els.tradeoffDetail.innerHTML = "";
    return;
  }
  const row = point.row;
  const device = state.activeTab === ON_DEVICE_TAB;
  const measured = !device && state.tradeoff.speed !== "rtfx" ? row.curated_speed : null;
  const track = measured?.tracks?.[state.tradeoff.speed];
  const throughputRange = track ? `${formatMetric("rtfx", track.throughput_rtfx.min)}–${formatMetric("rtfx", track.throughput_rtfx.max)}` : "";
  els.tradeoffDetail.innerHTML = `<h3>${escapeHtml(displayModelName(row.model))}</h3>
    <dl class="meta-list">
      ${definition(state.tradeoff.metric === "cer" ? "Main CER" : "All-sample CER", formatPercent(point.error))}
      ${definition(tradeoffSpeedLabel(), formatMetric("rtfx", point.speed))}
      ${track ? definition("3회 최솟값–최댓값", throughputRange) : ""}
      ${track ? definition("단건 p95 · B1", `${measured.tracks.b1.request_latency_p95_ms.median.toFixed(1)} ms`) : ""}
      ${track ? definition("속도 측정 상한 종료", `${track.token_limit} / ${track.measured_generations} (${formatPercent(track.token_limit_rate)})`) : ""}
      ${definition("Outlier rate", formatPercent(outlierRate(row)))}
      ${definition(device ? "기기" : "GPU", row.device || row.gpu || "미확인 / API")}
      ${definition("런타임", row.runtime || (row.torch ? `PyTorch ${row.torch}${row.cuda ? ` · CUDA ${row.cuda}` : ""}` : "미기록"))}
      ${definition(track ? "정확도 / 속도 batch" : "Batch size", track ? `4 / ${track.batch_size}` : batchSizeLabel(row) || "미기록")}
      ${definition("정밀도", executionValues(row, "precision").map(value => value === UNKNOWN_EXECUTION ? "미기록" : value).join(" / "))}
    </dl>
    ${track ? '<p class="detail-note">정확도: 전체 데이터 · 속도: 고정 256개, 3회 중앙값</p>' : ""}
    ${device ? '<p class="detail-note">QNN 그래프 실행 시간 기준. 전처리·전송·토큰화 제외.</p>' : ""}`;
}

function renderOverall() {
  const leaderboard = buildOverallLeaderboard();
  const rows = sortOverallRows(filterOverallRows(leaderboard.ranked));
  const incomplete = filterOverallRows(leaderboard.incomplete);
  els.table.className = "overall-table";
  els.resultsTitle.textContent = "Overall Model Leaderboard";
  els.rowCount.textContent =
    `${rows.length}개 모델 · 3개 구간 동일 가중 평균`;
  els.overallCoverage.textContent = incomplete.length
    ? `Overall 미완료: ${incomplete.map((row) => `${displayModelName(row.model)} (${row.missing_slices.join(", ")} 미충족)`).join(" · ")}`
    : "";
  els.overallCoverage.hidden = !incomplete.length;
  els.head.innerHTML = renderHeader(overallColumns);
  els.body.innerHTML = rows.length
    ? rows.map((row, index) => renderOverallRow(row, canRankSpeed(rows) ? index + 1 : null)).join("")
    : `<tr><td colspan="${overallColumns.length}" class="empty-state">필수 3개 평가 구간을 모두 충족하는 모델이 없습니다.</td></tr>`;
}

function renderDataset() {
  const filtered = sortRows(filterDatasetRows(state.rows, state.activeTab));
  const rows = filtered.filter(hasStandardProtocol);
  const references = filtered.filter((row) => !hasStandardProtocol(row));
  els.table.className = "dataset-table";
  const subset = activeDatasetSubset();
  const showRanks = subset !== ALL_SLICES && canRankSpeed(rows);
  const subsetLabel = subset === ALL_SLICES ? "" : ` · ${displaySubsetName(subset)}`;
  els.resultsTitle.textContent = `${displayDatasetName(state.activeTab)} Results${subsetLabel}`;
  els.rowCount.textContent =
    `${rows.length}개 ${showRanks ? "순위 결과" : "결과"}`;
  els.head.innerHTML = renderHeader(datasetColumns);
  els.body.innerHTML = rows.length
    ? rows.map((row, index) => renderDatasetRow(row, showRanks ? index + 1 : null)).join("")
    : `<tr><td colspan="${datasetColumns.length}" class="empty-state">조건에 맞는 ${showRanks ? "순위 결과" : "결과"}가 없습니다.</td></tr>`;
  els.references.hidden = !references.length;
  if (!references.length) els.references.open = false;
  els.referenceSummary.textContent = `참고 결과 ${references.length}개`;
  els.referenceHead.innerHTML = renderHeader(datasetColumns);
  els.referenceBody.innerHTML = references.map((row) => renderDatasetRow(row, "—")).join("");
}

function renderOnDevice() {
  const rows = sortRows(filterOnDeviceRows(state.onDeviceRows));
  els.table.className = "ondevice-table";
  els.resultsTitle.textContent = "On-device Leaderboard";
  els.rowCount.textContent =
    `${rows.length}개 결과`;
  els.head.innerHTML = renderHeader(onDeviceColumns);
  els.body.innerHTML = rows.length
    ? rows.map((row, index) => renderOnDeviceRow(row, canRankSpeed(rows) ? index + 1 : null)).join("")
    : `<tr><td colspan="${onDeviceColumns.length}" class="empty-state">조건에 맞는 온디바이스 결과가 없습니다.</td></tr>`;
}

function canRankSpeed(rows) {
  if (!["rtfx", "latency"].includes(state.sortMetric)) return true;
  if (!rows.length) return false;
  const slices = new Map();
  for (const row of rows.flatMap((entry) => entry.rows || [entry])) {
    const condition = tradeoffCondition(row);
    if (!condition.complete) return false;
    const slice = tradeoffSliceKey(row);
    if (slices.has(slice) && slices.get(slice) !== condition.key) return false;
    slices.set(slice, condition.key);
  }
  return true;
}

function renderHeader(columns) {
  const sortable = {"Main CER ↓": "cer", "Outlier rate ↓": "outlier_rate",
    "All-sample CER ↓": "all_samples_cer", "RTFx ↑": "rtfx", "QNN RTFx ↑": "rtfx"};
  return `<tr>${columns
    .map((column) => {
      const numeric = isNumericColumn(column) ? ' class="numeric"' : "";
      const metric = sortable[column];
      const active = metric === state.sortMetric;
      const order = HIGHER_IS_BETTER.has(metric) ? "descending" : "ascending";
      const sort = active ? ` aria-sort="${order}"` : "";
      const label = metric ? `<button type="button" class="sort-button" data-sort-metric="${metric}">${escapeHtml(column)}</button>` : escapeHtml(column);
      return `<th scope="col"${numeric}${sort}>${label}</th>`;
    })
    .join("")}</tr>`;
}

function isNumericColumn(column) {
  return (
    column === "Rank" ||
    column === "Samples" ||
    column.includes("CER") ||
    column.includes("WER") ||
    column.includes("MER") ||
    column.includes("JER") ||
    column.includes("SER") ||
    column.includes("RTFx") ||
    column.includes("Latency") ||
    column.includes("P95") ||
    column.includes("Outlier")
  );
}

function filterOnDeviceRows(rows) {
  return rows.filter((row) => {
    if (!matchesExecution(row)) return false;
    if (state.model !== "all" && canonicalModelId(row) !== state.model) {
      return false;
    }
    if (!state.search) {
      return true;
    }
    return searchText(row).includes(state.search);
  });
}

function filterDatasetRows(rows, dataset) {
  const subset = activeDatasetSubset();
  return rows.filter((row) => {
    if (!matchesExecution(row)) return false;
    if (row.dataset !== dataset) {
      return false;
    }
    if (subset !== ALL_SLICES && (row.subset || "default") !== subset) {
      return false;
    }
    if (state.model !== "all" && canonicalModelId(row) !== state.model) {
      return false;
    }
    if (!state.search) {
      return true;
    }
    return searchText(row).includes(state.search);
  });
}

function filterOverallRows(rows) {
  return rows.filter((row) => {
    const runs = row.rows.length ? row.rows : [row.representative_run];
    if (!runs.every(matchesExecution)) return false;
    if (state.model !== "all" && row.model_id !== state.model) {
      return false;
    }
    if (!state.search) {
      return true;
    }
    return overallSearchText(row).includes(state.search);
  });
}

function sortRows(rows) {
  return [...rows].sort((a, b) => {
    const aValue = sortValue(a, state.sortMetric);
    const bValue = sortValue(b, state.sortMetric);
    if (aValue !== bValue) {
      return aValue - bValue;
    }
    return compactDatasetLabel(a).localeCompare(compactDatasetLabel(b)) || String(a.model).localeCompare(String(b.model));
  });
}

function sortOverallRows(rows) {
  return [...rows].sort((a, b) => {
    const aValue = overallSortValue(a, state.sortMetric);
    const bValue = overallSortValue(b, state.sortMetric);
    if (aValue !== bValue) {
      return aValue - bValue;
    }
    return String(a.model).localeCompare(String(b.model)) || a.model_id.localeCompare(b.model_id);
  });
}

function buildOverallLeaderboard() {
  const byModel = new Map();
  for (const row of state.rows) {
    if (!row.model || !row.dataset) {
      continue;
    }
    const id = canonicalModelId(row);
    const modelRows = byModel.get(id) || [];
    modelRows.push(row);
    byModel.set(id, modelRows);
  }

  const ranked = [];
  const incomplete = [];
  for (const [id, modelRows] of byModel) {
    const candidates = [...modelRows].sort(compareRepresentativeRuns);
    const rows = [];
    const missing = [];
    for (const slice of OVERALL_SLICES) {
      const matching = candidates.filter((row) => row.dataset === slice.dataset && row.subset === slice.subset
        && row.is_full_evaluation === true && row.evaluated_samples === slice.samples
        && row.total_samples === slice.samples && row.dataset_total_samples === slice.samples
        && Number.isFinite(metricValue(row, "cer")));
      const run = matching.find(hasStandardProtocol);
      if (run) rows.push(run);
      else missing.push(compactDatasetLabel(slice) + (matching.length ? " · 평가 규약 다름/미확인" : ""));
    }
    const representative = [...rows].sort(compareRepresentativeRuns)[0] || candidates[0];
    const entry = {
      key: `overall:${id}`,
      model_id: id,
      model: representative.model,
      aliases: uniqueSorted(modelRows.map((row) => row.model)),
      model_repo: representative.model_repo,
      rows,
      dataset_count: rows.length,
      datasets: rows.map(compactDatasetLabel),
      dataset_groups: groupDatasetCoverage(rows),
      sources: uniqueSorted(rows.map((row) => row.source || "run artifact")),
      representative_run: representative,
      missing_slices: missing,
    };
    if (missing.length) {
      incomplete.push(entry);
      continue;
    }
    entry.metrics = {
      cer: averageMetric(rows, "cer"),
      all_samples_cer: averageMetric(rows, "all_samples_cer"),
      wer: averageMetric(rows, "wer"),
      mer: averageMetric(rows, "mer"),
      jer: averageMetric(rows, "jer"),
      ser: averageMetric(rows, "ser"),
      rtfx: averageMetric(rows, "rtfx"),
      latency: averageMetric(rows, "latency"),
      outlier_rate: averageValues(rows.map(outlierRate)),
    };
    ranked.push(entry);
  }
  return {ranked, incomplete};
}

function hasStandardProtocol(row) {
  return row.evaluation_protocol === STANDARD_PROTOCOL && row.normalization_preset === "kspon"
    && row.outlier_policy?.metric === "cer" && row.outlier_policy?.threshold === 1;
}

function renderOverallRow(row, rank) {
  const expanded = state.expandedKey === row.key;
  const detail = expanded ? renderOverallDetailRow(row) : "";
  return `
    <tr>
      <td class="numeric">${Number.isInteger(rank) ? `<span class="rank-number${rank <= 3 ? " rank-leading" : ""}">${rank}</span>` : "—"}</td>
      <td>
        <span class="model-cell">
          ${renderComparisonCheckbox(row)}
          <button class="row-toggle" type="button" data-expand-key="${escapeAttr(row.key)}" aria-expanded="${expanded}" aria-label="${escapeAttr(row.model)} 상세 보기">${expanded ? "-" : "+"}</button>
          ${renderModelIdentity(row.model, row.model_repo, row.metrics[state.sortMetric])}
        </span>
      </td>
      ${renderOverallMetricCell(row, "cer")}
      <td class="numeric outlier-cell">${formatPercent(row.metrics.outlier_rate)}</td>
      ${renderOverallMetricCell(row, "all_samples_cer")}
      <td class="numeric">${formatMetric("rtfx", row.metrics.rtfx)}</td>
      <td class="coverage-cell">
        ${renderDatasetCoverageSummary(row.dataset_groups)}
      </td>
    </tr>
    ${detail}`;
}

function renderOverallMetricCell(row, metric) {
  const value = row.metrics[metric];
  return `
    <td class="numeric metric-cell${metric === "cer" ? " main-score" : ""}">
      <span class="metric-value">${formatMetric(metric, value)}</span>
    </td>`;
}

function renderDatasetRow(row, rank) {
  const expanded = state.expandedKey === row.run_id;
  const detail = expanded ? renderDatasetDetailRow(row) : "";
  return `
    <tr>
      <td class="numeric">${hasStandardProtocol(row) && Number.isInteger(rank) ? `<span class="rank-number${rank <= 3 ? " rank-leading" : ""}">${rank}</span>` : "—"}</td>
      <td>
        <span class="model-cell">
          ${hasStandardProtocol(row) ? renderComparisonCheckbox(row) : ""}
          <button class="row-toggle" type="button" data-expand-key="${escapeAttr(row.run_id)}" aria-expanded="${expanded}" aria-label="${escapeAttr(row.model)} 상세 보기">${expanded ? "-" : "+"}</button>
          ${renderModelIdentity(row.model, row.model_repo, metricValue(row, state.sortMetric))}
        </span>
        ${hasStandardProtocol(row) ? "" : `<span class="subset-name">${escapeHtml(protocolLabel(row))}</span>`}
      </td>
      ${renderMetricCell(row, "cer")}
      <td class="numeric outlier-cell">${formatPercent(outlierRate(row))}</td>
      ${renderMetricCell(row, "all_samples_cer")}
      <td class="numeric">${formatMetric("rtfx", metricValue(row, "rtfx"))}</td>
      <td>
        <span class="dataset-name">${escapeHtml(compactDatasetLabel(row))}</span>
        <span class="subset-name">${formatInteger(row.evaluated_samples || row.total_samples || 0)}개 샘플</span>
      </td>
    </tr>
    ${detail}`;
}

function renderOnDeviceRow(row, rank) {
  const expanded = state.expandedKey === row.run_id;
  const detail = expanded ? renderOnDeviceDetailRow(row) : "";
  return `
    <tr>
      <td class="numeric">${Number.isInteger(rank) ? `<span class="rank-number${rank <= 3 ? " rank-leading" : ""}">${rank}</span>` : "—"}</td>
      <td>
        <span class="model-cell">
          ${renderComparisonCheckbox(row)}
          <button class="row-toggle" type="button" data-expand-key="${escapeAttr(row.run_id)}" aria-expanded="${expanded}" aria-label="${escapeAttr(row.model)} 상세 보기">${expanded ? "-" : "+"}</button>
          ${renderModelIdentity(row.model, row.model_repo, metricValue(row, state.sortMetric))}
        </span>
      </td>
      ${renderMetricCell(row, "cer")}
      <td class="numeric outlier-cell">${formatPercent(outlierRate(row))}</td>
      ${renderMetricCell(row, "all_samples_cer")}
      <td class="numeric">${formatMetric("rtfx", metricValue(row, "rtfx"))}</td>
      <td class="device-cell">
        <strong>${escapeHtml(row.device || "-")}</strong>
        <small>${escapeHtml(row.precision || "-")} · ${escapeHtml(row.backend || "-")}</small>
      </td>
      <td>
        <span class="dataset-name">${escapeHtml(compactDatasetLabel(row))}</span>
        <span class="subset-name">${formatInteger(row.evaluated_samples || row.total_samples || 0)}개 샘플</span>
      </td>
    </tr>
    ${detail}`;
}

function renderMetricCell(row, metric) {
  const value = metricValue(row, metric);
  return `
    <td class="numeric metric-cell${metric === "cer" ? " main-score" : ""}">
      <span class="metric-value">${formatMetric(metric, value)}</span>
    </td>`;
}

function renderModelIdentity(model, repo, sortValue) {
  const url = modelRepoUrl(repo);
  const modelName = escapeHtml(displayModelName(model));
  const repoName = escapeHtml(repo || "");
  const sortNote = ["wer", "mer", "jer", "ser", "latency"].includes(state.sortMetric)
    ? `<span class="sort-value">${metricLabels[state.sortMetric]} ${formatMetric(state.sortMetric, sortValue)}</span>` : "";
  if (!url) {
    return `
      <span class="model-stack">
        <span class="model-name">${modelName}</span>
        <span class="model-repo">${repoName}</span>
        ${sortNote}
      </span>`;
  }
  return `
    <span class="model-stack">
      <a class="model-name model-name-link" href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">${modelName}</a>
      <span class="model-repo">${repoName}</span>
      ${sortNote}
    </span>`;
}

function displayModelName(model) {
  return String(model || "-")
    .replace(/^openai\/whisper-/i, "Whisper ")
    .replace(/^Qwen\/Qwen3-ASR-/i, "Qwen3-ASR ")
    .replace(/^whisper[_-]/, "Whisper ")
    .replace(/^qwen3_asr_/, "Qwen3-ASR ")
    .replace(/^google_speech_recognition$/, "Google Speech Recognition")
    .replace(/(\d)_(\d)/g, "$1.$2")
    .replaceAll("_", " ")
    .replace(/\blarge v3\b/g, "large-v3")
    .replace(/(\d)b\b/g, "$1B");
}

function modelRepoUrl(repo) {
  const value = String(repo || "").trim();
  if (!value) {
    return "";
  }
  if (/^https?:\/\//i.test(value)) {
    return value;
  }
  if (!value.includes("/")) {
    return "";
  }
  return `https://huggingface.co/${value.split("/").map(encodeURIComponent).join("/")}`;
}

function renderOverallDetailRow(row) {
  const coverage = row.rows
    .map(
      (run) => `
        <div class="coverage-item">
          <span>
            <strong>${escapeHtml(compactDatasetLabel(run))}</strong>
          </span>
          <span class="coverage-metrics">
            Main CER ${formatPercent(metricValue(run, "cer"))}
            <i>Outlier ${formatPercent(outlierRate(run))}</i>
            <i>All-sample CER ${formatMetric("all_samples_cer", metricValue(run, "all_samples_cer"))}</i>
          </span>
        </div>`,
    )
    .join("");
  return `
    <tr class="detail-row">
      <td colspan="${overallColumns.length}">
        <div class="detail-panel">
          <div class="detail-group">
            <h3>평가 구간별 성능</h3>
            <div class="coverage-list">${coverage}</div>
            <p class="detail-note">Kspon 정규화 · CER 100% 초과 제외</p>
            <a class="artifact-link" href="leaderboard_data.json">원본 JSON · 재현 정보</a>
          </div>
          <div class="detail-group">
            <h3>보조 지표 · 평가 구간별 평균</h3>
            <dl class="metric-list">
              ${definition("WER", formatPercent(row.metrics.wer))}
              ${definition("MER", formatPercent(row.metrics.mer))}
              ${definition("JER", formatPercent(row.metrics.jer))}
              ${definition("SER", formatPercent(row.metrics.ser))}
              ${definition("평균 지연", formatLatency(row.metrics.latency))}
            </dl>
          </div>
        </div>
      </td>
    </tr>`;
}

function renderDatasetDetailRow(row) {
  const micro = (row.metrics && row.metrics.micro) || {};
  const latency = (row.metrics && row.metrics.latency_percentiles) || {};
  const batchSize = batchSizeLabel(row);
  return `
    <tr class="detail-row">
      <td colspan="${datasetColumns.length}">
        <div class="detail-panel">
          <div class="detail-group">
            <h3>보조 지표 · outlier 제외</h3>
            <dl class="metric-list">
              ${definition("WER · macro", formatPercent(metricValue(row, "wer")))}
              ${definition("MER · macro", formatPercent(metricValue(row, "mer")))}
              ${definition("JER · macro", formatPercent(metricValue(row, "jer")))}
              ${definition("SER · macro", formatPercent(metricValue(row, "ser")))}
              ${definition("CER · corpus", formatPercent(micro.cer))}
              ${definition("평균 지연", formatLatency(metricValue(row, "latency")))}
              ${Number.isFinite(latency.p50) ? definition("지연 p50", formatLatency(latency.p50)) : ""}
              ${Number.isFinite(latency.p95) ? definition("지연 p95", formatLatency(latency.p95)) : ""}
            </dl>
          </div>
          <div class="detail-group">
            <h3>평가 조건</h3>
            <dl class="meta-list">
              ${row.gpu ? definition("GPU", row.gpu) : ""}
              ${batchSize ? definition("Batch size", batchSize) : ""}
              ${definition("정규화", normalizationLabel(row.normalization_preset))}
              ${definition("Outlier 기준", outlierPolicy(row))}
            </dl>
            <div class="artifact-links"><a class="artifact-link" href="leaderboard_data.json">원본 JSON · 재현 정보</a></div>
          </div>
        </div>
      </td>
    </tr>`;
}

function renderOnDeviceDetailRow(row) {
  const micro = (row.metrics && row.metrics.micro) || {};
  const latency = (row.metrics && row.metrics.latency_percentiles) || {};
  return `
    <tr class="detail-row">
      <td colspan="${onDeviceColumns.length}">
        <div class="detail-panel">
          <div class="detail-group">
            <h3>보조 지표 · outlier 제외</h3>
            <dl class="metric-list">
              ${definition("WER · macro", formatPercent(metricValue(row, "wer")))}
              ${definition("SER · macro", formatPercent(metricValue(row, "ser")))}
              ${definition("평균 지연", formatLatency(metricValue(row, "latency")))}
              ${definition("CER · corpus", formatPercent(micro.cer))}
              ${Number.isFinite(latency.p50) ? definition("지연 p50", formatLatency(latency.p50)) : ""}
              ${Number.isFinite(latency.p95) ? definition("지연 p95", formatLatency(latency.p95)) : ""}
            </dl>
            ${row.performance_scope ? `<p class="detail-note">${escapeHtml(row.performance_scope.startsWith("QNN graph execution only;") ? "속도 측정 범위: QNN 그래프 실행 (전처리·전송·토큰화 제외)" : row.performance_scope)}</p>` : ""}
            ${renderArtifactLinks(row)}
          </div>
          <div class="detail-group">
            <h3>기기·평가 조건</h3>
            <dl class="meta-list">
              ${definition("기기", row.device || "-")}
              ${definition("SoC", row.soc || "-")}
              ${definition("가속기", row.accelerator || "-")}
              ${definition("런타임", row.runtime || "-")}
              ${definition("정밀도", row.precision || "-")}
              ${definition("Outlier 기준", outlierPolicy(row))}
            </dl>
          </div>
        </div>
      </td>
    </tr>`;
}

function renderArtifactLinks(row) {
  const links = [
    [row.report_url, "벤치마크 보고서"],
    [row.result_url, "원본 JSON"],
  ]
    .filter(([url]) => /^https:\/\//i.test(String(url || "")))
    .map(
      ([url, label]) =>
        `<a class="artifact-link" href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`,
    )
    .join("");
  return links ? `<div class="artifact-links">${links}</div>` : "";
}

function handleTableClick(event) {
  const toggle = event.target.closest("[data-expand-key]");
  if (toggle) {
    const key = toggle.getAttribute("data-expand-key");
    state.expandedKey = state.expandedKey === key ? null : key;
    render();
  }
}

function definition(term, value) {
  return `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`;
}

function batchSizeLabel(row) {
  const data = row.reproducibility || {};
  const records = data.source_runs ? data.source_runs.map((source) => source.reproducibility || {}) : [data];
  const sizes = records.map((record) => record.execution?.batch_size);
  const known = sizes.filter((size) => Number.isInteger(size) && size > 0);
  if (!known.length) return "";
  const values = [...new Set(known)].sort((a, b) => a - b);
  return values.join(" / ") + (known.length < sizes.length ? " (일부 미확인)" : values.length > 1 ? " (구간별)" : "");
}

function normalizeRow(row) {
  const metrics = row.metrics || {};
  const macro = metrics.macro || {};
  if (!Number.isFinite(macro.rtfx) && Number.isFinite(macro.rtf) && macro.rtf > 0) {
    return {
      ...row,
      metrics: {
        ...metrics,
        macro: {
          ...macro,
          rtfx: 1 / macro.rtf,
        },
      },
    };
  }
  return row;
}

function metricValue(row, metric) {
  if (metric === "all_samples_cer") {
    const value = row.metrics && row.metrics.all_samples_micro && row.metrics.all_samples_micro.cer;
    return Number.isFinite(value) ? value : NaN;
  }
  const macro = (row.metrics && row.metrics.macro) || {};
  const value = macro[metric];
  if (metric === "rtfx" && !Number.isFinite(value)) {
    const rtf = macro.rtf;
    return Number.isFinite(rtf) && rtf > 0 ? 1 / rtf : NaN;
  }
  return Number.isFinite(value) ? value : NaN;
}

function sortValue(row, metric) {
  if (metric === "outlier_rate") {
    return outlierRate(row);
  }
  return orderableScore(metric, metricValue(row, metric));
}

function overallSortValue(row, metric) {
  return orderableScore(metric, row.metrics[metric]);
}

// Raw (human-readable) value of the active ranking metric, without sort negation.
function overallScore(row, metric) {
  const value = row.metrics[metric];
  return Number.isFinite(value) ? value : NaN;
}

// Map a metric value to an always-ascending score: smaller = better.
// Higher-is-better metrics are negated; missing values always sink to the bottom.
function orderableScore(metric, value) {
  if (!Number.isFinite(value)) {
    return Number.POSITIVE_INFINITY;
  }
  return HIGHER_IS_BETTER.has(metric) ? -value : value;
}

function averageMetric(rows, metric) {
  const values = rows.map((row) => metricValue(row, metric));
  return values.every(Number.isFinite) ? averageValues(values) : NaN;
}

function averageValues(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) {
    return NaN;
  }
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function outlierRate(row) {
  const denominator = row.evaluated_samples || row.total_samples || 0;
  return denominator > 0 ? (row.outlier_count || 0) / denominator : Number.POSITIVE_INFINITY;
}

function outlierPolicy(row) {
  const policy = row.outlier_policy || {};
  if (!policy.metric) {
    return "미확인";
  }
  const threshold = formatMetric(policy.metric, policy.threshold).replace(/\.00%$/, "%");
  return `${policy.metric.toUpperCase()} ${threshold} 초과 제외`;
}

function normalizationLabel(preset) {
  return {kspon: "Kspon 정규화", punctuation_agnostic: "구두점 무시 정규화", strict: "유니코드·공백 정규화", raw: "정규화 없음"}[preset] || "미확인";
}

function protocolLabel(row) {
  return row.evaluation_protocol
    ? `${normalizationLabel(row.normalization_preset)} · ${outlierPolicy(row)}`
    : "평가 규약 미확인";
}

function renderDatasetCoverageSummary(groups) {
  return Object.entries(groups)
    .map(
      ([dataset, subsets]) => `
        <span class="coverage-summary">
          <span>${escapeHtml(displayDatasetName(dataset))}</span>
          ${subsets.map((subset) => `<i>${escapeHtml(displaySubsetName(subset))}</i>`).join("")}
        </span>`,
    )
    .join("");
}

function groupDatasetCoverage(rows) {
  const groups = {};
  for (const row of rows) {
    const dataset = row.dataset || "-";
    groups[dataset] = groups[dataset] || [];
    groups[dataset].push(row.subset || "default");
  }
  return Object.fromEntries(
    Object.entries(groups).map(([dataset, subsets]) => [
      dataset,
      uniqueSorted(subsets).sort(compareSubsets),
    ]),
  );
}

function activeDatasetSubset() {
  if (state.activeTab === OVERALL_TAB || state.activeTab === ON_DEVICE_TAB) {
    return ALL_SLICES;
  }
  return state.subsetByDataset[state.activeTab] || defaultSubsetForDataset(state.activeTab);
}

function defaultSubsetForDataset(dataset) {
  const subsets = new Set(
    state.rows
      .filter((row) => row.dataset === dataset)
      .map((row) => row.subset || "default"),
  );
  if (dataset === AIHUB_DATASET && subsets.has("all")) {
    return "all";
  }
  return ALL_SLICES;
}

function displayDatasetName(dataset) {
  if (dataset === AIHUB_DATASET) {
    return "AIHub";
  }
  return dataset || "-";
}

function displaySubsetName(subset) {
  if (!subset || subset === "default") {
    return "default";
  }
  if (String(subset).toLowerCase() === "all") {
    return "All";
  }
  return subset;
}

function compactDatasetLabel(row) {
  const dataset = displayDatasetName(row.dataset);
  return row.subset ? `${dataset} ${displaySubsetName(row.subset)}` : dataset;
}

function compareSubsets(a, b) {
  const order = ["clean", "other", "D01", "D02", "D03", "D04", "all", "default"];
  const aIndex = order.indexOf(a);
  const bIndex = order.indexOf(b);
  if (aIndex !== -1 || bIndex !== -1) {
    return (aIndex === -1 ? Number.MAX_SAFE_INTEGER : aIndex) - (bIndex === -1 ? Number.MAX_SAFE_INTEGER : bIndex);
  }
  return String(a).localeCompare(String(b));
}

function searchText(row) {
  return [
    row.model,
    row.model_repo,
    row.dataset,
    row.subset,
    compactDatasetLabel(row),
    row.gpu,
    row.device,
    row.soc,
    row.accelerator,
    row.backend,
    row.runtime,
    row.precision,
    row.run_id,
    row.command,
    row.source,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function overallSearchText(row) {
  return [
    row.model,
    row.model_id,
    row.aliases.join(" "),
    row.model_repo,
    row.datasets.join(" "),
    row.sources.join(" "),
    row.representative_run.run_id,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b));
}

function formatMetric(metric, value) {
  if (!Number.isFinite(value)) {
    return metric === "all_samples_cer" ? "N/A" : "-";
  }
  if (metric === "rtfx") {
    return `${value.toFixed(2)}×`;
  }
  if (metric === "latency") {
    return formatLatency(value);
  }
  return formatPercent(value);
}

function formatLatency(value) {
  return Number.isFinite(value)
    ? `${(value * 1000).toLocaleString("en-US", { maximumFractionDigits: 1 })} ms` : "-";
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "-";
}

function formatInteger(value) {
  return Number.isFinite(value) ? Math.round(value).toLocaleString("en-US") : "-";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}
