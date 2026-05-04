const OVERALL_TAB = "overall";
const ALL_SLICES = "__all_slices__";
const AIHUB_DATASET = "AIHubLowQualityTelephone";

const state = {
  rows: [],
  search: "",
  activeTab: OVERALL_TAB,
  subsetByDataset: {},
  model: "all",
  sortMetric: "cer",
  expandedKey: null,
};

const metricLabels = {
  cer: "CER",
  wer: "WER",
  mer: "MER",
  jer: "JER",
  ser: "SER",
  rtf: "RTFx",
  latency: "Latency",
  outlier_rate: "Outlier",
};

const datasetColumns = [
  "Rank",
  "Model",
  "Dataset",
  "CER",
  "WER",
  "MER",
  "JER",
  "SER",
  "RTFx",
  "Latency",
  "Samples",
  "GPU",
];

const overallColumns = [
  "Rank",
  "Model",
  "Avg CER",
  "Avg WER",
  "Avg MER",
  "Avg JER",
  "Avg SER",
  "Avg RTFx",
  "Avg Latency",
  "Datasets",
];

const els = {
  body: document.getElementById("leaderboardBody"),
  head: document.getElementById("leaderboardHead"),
  table: document.getElementById("leaderboardTable"),
  rowCount: document.getElementById("rowCount"),
  status: document.getElementById("dataStatus"),
  tabs: document.getElementById("datasetTabs"),
  subsetTabs: document.getElementById("subsetTabs"),
  search: document.getElementById("searchInput"),
  model: document.getElementById("modelFilter"),
  sortMetric: document.getElementById("sortMetric"),
  resultsTitle: document.getElementById("resultsTitle"),
  summaryRuns: document.getElementById("summaryRuns"),
  summaryModels: document.getElementById("summaryModels"),
  summaryDatasets: document.getElementById("summaryDatasets"),
  summaryBestCer: document.getElementById("summaryBestCer"),
  summaryBestCerLabel: document.getElementById("summaryBestCerLabel"),
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
}

async function loadLeaderboard() {
  try {
    const response = await fetch("leaderboard_data.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    state.rows = Array.isArray(data) ? data : [];
    state.rows = state.rows.filter((row) => row.is_full_evaluation !== false);
    populateModelFilter();
    renderTabs();
    render();
    els.status.textContent = `${state.rows.length} full run(s)`;
  } catch (error) {
    els.status.textContent = "data load failed";
    els.rowCount.innerHTML =
      '<span class="error-state">leaderboard_data.json을 불러오지 못했습니다.</span>';
    els.body.innerHTML =
      `<tr><td colspan="${datasetColumns.length}" class="empty-state error-state">정적 빌드에 데이터 파일이 포함되어 있는지 확인하세요.</td></tr>`;
    console.error(error);
  }
}

function populateModelFilter() {
  const models = uniqueSorted(state.rows.map((row) => row.model).filter(Boolean));
  fillSelect(els.model, "all", "전체 모델", models);
}

function fillSelect(select, allValue, allLabel, values) {
  select.innerHTML = [
    `<option value="${escapeAttr(allValue)}">${escapeHtml(allLabel)}</option>`,
    ...values.map((value) => `<option value="${escapeAttr(value)}">${escapeHtml(value)}</option>`),
  ].join("");
}

function renderTabs() {
  const datasets = uniqueSorted(state.rows.map((row) => row.dataset).filter(Boolean));
  const tabs = [
    {
      key: OVERALL_TAB,
      label: "Overall",
      note: "model average",
      count: buildOverallRows().length,
    },
    ...datasets.map((dataset) => ({
      key: dataset,
      label: displayDatasetName(dataset),
      note: "dataset",
      count: state.rows.filter((row) => row.dataset === dataset).length,
    })),
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
          <small>${escapeHtml(tab.note)}</small>
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
  state.expandedKey = null;
  renderTabs();
  renderSubsetTabs();
  render();
}

function handleSubsetClick(event) {
  const button = event.target.closest("[data-subset]");
  if (!button || state.activeTab === OVERALL_TAB) {
    return;
  }
  state.subsetByDataset[state.activeTab] = button.getAttribute("data-subset");
  state.expandedKey = null;
  renderSubsetTabs();
  render();
}

function renderSubsetTabs() {
  if (state.activeTab === OVERALL_TAB) {
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
  renderSummary();
  renderSubsetTabs();
  if (state.activeTab === OVERALL_TAB) {
    renderOverall();
  } else {
    renderDataset();
  }
}

function renderSummary() {
  const rows = state.rows;
  const models = new Set(rows.map((row) => row.model).filter(Boolean));
  const datasetSlices = new Set(rows.map(datasetLabel));
  const best = rows
    .map((row) => ({ row, cer: metricValue(row, "cer") }))
    .filter((item) => Number.isFinite(item.cer))
    .sort((a, b) => a.cer - b.cer)[0];

  els.summaryRuns.textContent = rows.length.toString();
  els.summaryModels.textContent = models.size.toString();
  els.summaryDatasets.textContent = datasetSlices.size.toString();
  els.summaryBestCer.textContent = best ? formatNumber(best.cer) : "-";
  els.summaryBestCerLabel.textContent = best
    ? `${best.row.model} · ${compactDatasetLabel(best.row)}`
    : "waiting for data";
}

function renderOverall() {
  const rows = sortOverallRows(filterOverallRows(buildOverallRows()));
  els.table.className = "overall-table";
  els.resultsTitle.textContent = "Overall Model Leaderboard";
  els.rowCount.textContent =
    `${rows.length} model(s), ranked by average ${metricLabels[state.sortMetric]}. ` +
    "Each model/dataset slice contributes its best full run once.";
  els.head.innerHTML = renderHeader(overallColumns);
  els.body.innerHTML = rows.length
    ? rows.map((row, index) => renderOverallRow(row, index + 1)).join("")
    : `<tr><td colspan="${overallColumns.length}" class="empty-state">조건에 맞는 모델이 없습니다.</td></tr>`;
}

function renderDataset() {
  const rows = sortRows(filterDatasetRows(state.rows, state.activeTab));
  els.table.className = "dataset-table";
  const subset = activeDatasetSubset();
  const subsetLabel = subset === ALL_SLICES ? "" : ` · ${displaySubsetName(subset)}`;
  els.resultsTitle.textContent = `${displayDatasetName(state.activeTab)} Results${subsetLabel}`;
  els.rowCount.textContent =
    `${rows.length} run(s), sorted by ${metricLabels[state.sortMetric]}. ` +
    "Subset selection applies within the active dataset.";
  els.head.innerHTML = renderHeader(datasetColumns);
  els.body.innerHTML = rows.length
    ? rows.map((row, index) => renderDatasetRow(row, index + 1)).join("")
    : `<tr><td colspan="${datasetColumns.length}" class="empty-state">조건에 맞는 결과가 없습니다.</td></tr>`;
}

function renderHeader(columns) {
  return `<tr>${columns
    .map((column) => {
      const numeric = isNumericColumn(column) ? ' class="numeric"' : "";
      return `<th scope="col"${numeric}>${escapeHtml(column)}</th>`;
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
    column.includes("Latency")
  );
}

function filterDatasetRows(rows, dataset) {
  const subset = activeDatasetSubset();
  return rows.filter((row) => {
    if (row.dataset !== dataset) {
      return false;
    }
    if (subset !== ALL_SLICES && (row.subset || "default") !== subset) {
      return false;
    }
    if (state.model !== "all" && row.model !== state.model) {
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
    if (state.model !== "all" && row.model !== state.model) {
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
    return String(a.model).localeCompare(String(b.model));
  });
}

function buildOverallRows() {
  const byModelAndDataset = new Map();
  for (const row of state.rows) {
    if (!row.model || !row.dataset) {
      continue;
    }
    const key = `${row.model}::${row.dataset}::${row.subset || "default"}`;
    const current = byModelAndDataset.get(key);
    if (!current || sortValue(row, state.sortMetric) < sortValue(current, state.sortMetric)) {
      byModelAndDataset.set(key, row);
    }
  }

  const byModel = new Map();
  for (const row of byModelAndDataset.values()) {
    const modelRows = byModel.get(row.model) || [];
    modelRows.push(row);
    byModel.set(row.model, modelRows);
  }

  return [...byModel.entries()].map(([model, rows]) => {
    const bestRun = sortRows(rows)[0];
    return {
      key: `overall:${model}`,
      model,
      model_repo: bestRun.model_repo,
      rows: sortRows(rows),
      dataset_count: rows.length,
      datasets: rows.map(compactDatasetLabel),
      dataset_groups: groupDatasetCoverage(rows),
      sources: uniqueSorted(rows.map((row) => row.source || "run artifact")),
      best_run: bestRun,
      metrics: {
        cer: averageMetric(rows, "cer"),
        wer: averageMetric(rows, "wer"),
        mer: averageMetric(rows, "mer"),
        jer: averageMetric(rows, "jer"),
        ser: averageMetric(rows, "ser"),
        rtf: averageMetric(rows, "rtf"),
        latency: averageMetric(rows, "latency"),
        outlier_rate: averageValues(rows.map(outlierRate)),
      },
    };
  });
}

function renderOverallRow(row, rank) {
  const expanded = state.expandedKey === row.key;
  const detail = expanded ? renderOverallDetailRow(row) : "";
  return `
    <tr>
      <td class="numeric">${rank}</td>
      <td>
        <span class="model-cell">
          <button class="row-toggle" type="button" data-expand-key="${escapeAttr(row.key)}" aria-expanded="${expanded}" aria-label="${escapeAttr(row.model)} 상세 보기">${expanded ? "-" : "+"}</button>
          ${renderModelIdentity(row.model, row.model_repo)}
        </span>
      </td>
      ${renderOverallMetricCell(row, "cer")}
      ${renderOverallMetricCell(row, "wer")}
      ${renderOverallMetricCell(row, "mer")}
      ${renderOverallMetricCell(row, "jer")}
      ${renderOverallMetricCell(row, "ser")}
      <td class="numeric latency">${formatNumber(row.metrics.rtf)}</td>
      <td class="numeric latency">${formatSeconds(row.metrics.latency)}</td>
      <td class="coverage-cell">
        <span class="dataset-name">${formatInteger(row.dataset_count)} slices</span>
        ${renderDatasetCoverageSummary(row.dataset_groups)}
      </td>
    </tr>
    ${detail}`;
}

function renderOverallMetricCell(row, metric) {
  const value = row.metrics[metric];
  const width = Number.isFinite(value) ? Math.max(3, Math.min(100, value * 100)) : 0;
  return `
    <td class="numeric metric-cell">
      <span class="metric-value">${formatNumber(value)}</span>
      <span class="metric-bar" aria-hidden="true"><span style="width: ${width}%"></span></span>
    </td>`;
}

function renderDatasetRow(row, rank) {
  const expanded = state.expandedKey === row.run_id;
  const detail = expanded ? renderDatasetDetailRow(row) : "";
  return `
    <tr>
      <td class="numeric">${rank}</td>
      <td>
        <span class="model-cell">
          <button class="row-toggle" type="button" data-expand-key="${escapeAttr(row.run_id)}" aria-expanded="${expanded}" aria-label="${escapeAttr(row.model)} 상세 보기">${expanded ? "-" : "+"}</button>
          ${renderModelIdentity(row.model, row.model_repo)}
        </span>
      </td>
      <td>
        <span class="dataset-name">${escapeHtml(displayDatasetName(row.dataset))}</span>
        <span class="subset-name">${escapeHtml(displaySubsetName(row.subset || "default"))}</span>
      </td>
      ${renderMetricCell(row, "cer")}
      ${renderMetricCell(row, "wer")}
      ${renderMetricCell(row, "mer")}
      ${renderMetricCell(row, "jer")}
      ${renderMetricCell(row, "ser")}
      <td class="numeric latency">${formatNumber(metricValue(row, "rtf"))}</td>
      <td class="numeric latency">${formatSeconds(metricValue(row, "latency"))}</td>
      <td class="numeric">${samplePill(row)}</td>
      <td class="gpu-cell">${escapeHtml(row.gpu || "-")}</td>
    </tr>
    ${detail}`;
}

function renderMetricCell(row, metric) {
  const value = metricValue(row, metric);
  const width = Number.isFinite(value) ? Math.max(3, Math.min(100, value * 100)) : 0;
  return `
    <td class="numeric metric-cell">
      <span class="metric-value">${formatNumber(value)}</span>
      <span class="metric-bar" aria-hidden="true"><span style="width: ${width}%"></span></span>
    </td>`;
}

function renderModelIdentity(model, repo) {
  const url = modelRepoUrl(repo);
  const modelName = escapeHtml(model || "-");
  const repoName = escapeHtml(repo || "");
  if (!url) {
    return `
      <span class="model-stack">
        <span class="model-name">${modelName}</span>
        <span class="model-repo">${repoName}</span>
      </span>`;
  }
  return `
    <span class="model-stack">
      <a class="model-name model-name-link" href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">${modelName}</a>
      <span class="model-repo">${repoName}</span>
      <a class="model-card-link" href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">model card</a>
    </span>`;
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
            <small>${escapeHtml(run.run_id || "-")}</small>
          </span>
          <span class="coverage-metrics">
            CER ${formatNumber(metricValue(run, "cer"))}
            <i>WER ${formatNumber(metricValue(run, "wer"))}</i>
          </span>
        </div>`,
    )
    .join("");
  return `
    <tr class="detail-row">
      <td colspan="${overallColumns.length}">
        <div class="detail-panel">
          <div class="detail-group">
            <h3>Dataset coverage used for average</h3>
            <div class="coverage-list">${coverage}</div>
          </div>
          <div class="detail-group">
            <h3>Overall aggregation</h3>
            <dl class="meta-list">
              ${definition("ranking metric", `average ${metricLabels[state.sortMetric]}`)}
              ${definition("dataset slices", formatInteger(row.dataset_count))}
              ${definition("best contributing run", row.best_run.run_id || "-")}
              ${definition("dedupe policy", "best model/dataset slice by current sort metric")}
              ${definition("primary score", formatNumber(overallSortValue(row, state.sortMetric)))}
            </dl>
            <div class="detail-inline-list">
              <span>Sources</span>
              ${renderSourcePills(row.sources)}
            </div>
          </div>
        </div>
      </td>
    </tr>`;
}

function renderDatasetDetailRow(row) {
  const micro = (row.metrics && row.metrics.micro) || {};
  const latency = (row.metrics && row.metrics.latency_percentiles) || {};
  const modelMetrics = row.model_metrics || {};
  return `
    <tr class="detail-row">
      <td colspan="${datasetColumns.length}">
        <div class="detail-panel">
          <div class="detail-group">
            <h3>Micro metrics and latency percentiles</h3>
            <dl class="metric-list">
              ${definition("micro WER", formatNumber(micro.wer))}
              ${definition("micro CER", formatNumber(micro.cer))}
              ${definition("micro MER", formatNumber(micro.mer))}
              ${definition("micro JER", formatNumber(micro.jer))}
              ${definition("p50 latency", formatSeconds(latency.p50))}
              ${definition("p90 latency", formatSeconds(latency.p90))}
              ${definition("p95 latency", formatSeconds(latency.p95))}
              ${definition("p99 latency", formatSeconds(latency.p99))}
            </dl>
            ${renderCommand(row.command)}
          </div>
          <div class="detail-group">
            <h3>Run metadata</h3>
            <dl class="meta-list">
              ${definition("run id", row.run_id || "-")}
              ${definition("samples", `${row.evaluated_samples || row.total_samples || 0} / ${row.dataset_total_samples || row.total_samples || 0}`)}
              ${definition("outliers", `${row.outlier_count || 0} (${formatPercent(outlierRate(row))})`)}
              ${definition("torch", row.torch || "-")}
              ${definition("cuda", row.cuda || "-")}
              ${definition("source", row.source || "run artifact")}
              ${definition("params", modelMetrics.params || formatInteger(modelMetrics.total_parameters))}
              ${definition("flops", modelMetrics.flops || "-")}
              ${definition("outlier policy", outlierPolicy(row))}
              ${definition("artifact", row._artifact || "-")}
            </dl>
          </div>
        </div>
      </td>
    </tr>`;
}

function renderCommand(command) {
  if (!command) {
    return "";
  }
  return `
    <div class="command-box">
      <div class="command-head">
        <h3>Reproduction command</h3>
        <button class="copy-button" type="button" data-copy="${escapeAttr(command)}">copy</button>
      </div>
      <code class="command">${escapeHtml(command)}</code>
    </div>`;
}

function handleTableClick(event) {
  const toggle = event.target.closest("[data-expand-key]");
  if (toggle) {
    const key = toggle.getAttribute("data-expand-key");
    state.expandedKey = state.expandedKey === key ? null : key;
    render();
    return;
  }

  const copyButton = event.target.closest("[data-copy]");
  if (copyButton) {
    const command = copyButton.getAttribute("data-copy");
    navigator.clipboard
      .writeText(command)
      .then(() => flashButton(copyButton, "copied"))
      .catch(() => flashButton(copyButton, "copy failed"));
  }
}

function flashButton(button, text) {
  const original = button.textContent;
  button.textContent = text;
  window.setTimeout(() => {
    button.textContent = original;
  }, 1400);
}

function definition(term, value) {
  return `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`;
}

function metricValue(row, metric) {
  const macro = (row.metrics && row.metrics.macro) || {};
  const value = macro[metric];
  return Number.isFinite(value) ? value : NaN;
}

function sortValue(row, metric) {
  if (metric === "outlier_rate") {
    return outlierRate(row);
  }
  const value = metricValue(row, metric);
  return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
}

function overallSortValue(row, metric) {
  const value = row.metrics[metric];
  return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
}

function averageMetric(rows, metric) {
  return averageValues(rows.map((row) => metricValue(row, metric)));
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
    return "-";
  }
  return `${policy.metric} > ${policy.threshold}`;
}

function samplePill(row) {
  const evaluated = row.evaluated_samples || row.total_samples || 0;
  const total = row.dataset_total_samples || row.total_samples || 0;
  return `<span class="sample-pill">${formatInteger(evaluated)} / ${formatInteger(total)}</span>`;
}

function renderSourcePills(sources) {
  return sources.map((source) => `<span class="source-pill">${escapeHtml(source)}</span>`).join("");
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
  if (state.activeTab === OVERALL_TAB) {
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

function datasetLabel(row) {
  return compactDatasetLabel(row);
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
    row.model_repo,
    row.datasets.join(" "),
    row.sources.join(" "),
    row.best_run.run_id,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b));
}

function formatNumber(value) {
  return Number.isFinite(value) ? value.toFixed(4) : "-";
}

function formatSeconds(value) {
  return Number.isFinite(value) ? `${value.toFixed(4)}s` : "-";
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
