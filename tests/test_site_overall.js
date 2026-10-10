const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { test } = require("node:test");

const context = vm.createContext({
  document: { getElementById: () => null, addEventListener: () => {} },
});
vm.runInContext(fs.readFileSync("site/app.js", "utf8"), context);

function row(model, dataset, subset, cer) {
  const samples = dataset === "KsponSpeech" ? 3000 : 39916;
  return {
    model, model_repo: `org/${model}`, dataset, subset, run_id: `20261001-${subset}`,
    is_full_evaluation: true, total_samples: samples, evaluated_samples: samples,
    normalization_preset: "kspon", outlier_policy: {metric: "cer", threshold: 1},
    evaluation_protocol: "v1/kspon/cer>1.0",
    dataset_total_samples: samples, metrics: { macro: { cer } },
  };
}

function overall(rows) {
  context.inputRows = rows;
  return JSON.parse(vm.runInContext(
    "state.rows = inputRows; JSON.stringify(buildOverallLeaderboard().ranked)", context,
  ));
}

function suite(model, cer = 0.1) {
  return [
    row(model, "KsponSpeech", "clean", cer),
    row(model, "KsponSpeech", "other", cer),
    row(model, "AIHubLowQualityTelephone", "all", cer),
  ];
}

test("published Nemotron enters Overall once with all six source runs", () => {
  const rows = JSON.parse(fs.readFileSync("doc/leaderboard_data.json", "utf8"));
  const results = overall(rows);
  const nemotron = results.find(row => row.model_id === "nvidia/nemotron-3.5-asr-streaming-0.6b");
  assert.ok(nemotron);
  assert.ok(Math.abs(nemotron.metrics.cer - .31354459611898633) < 1e-12);
  assert.equal(nemotron.rows.length, 3);
  assert.equal(rows.filter(row => row.model_repo === nemotron.model_id).length, 7);
  context.nemotron = nemotron;
  const html = vm.runInContext("renderOverallDetailRow(nemotron)", context);
  assert.match(html, /6,923 \/ 45,916/);
  assert.match(html, /스트리밍 지연 미측정/);
});

test("AIHub all and its domains contribute only once, without changing dataset rows", () => {
  const rows = [
    row("model-a", "KsponSpeech", "clean", 0.1),
    row("model-a", "KsponSpeech", "other", 0.2),
    ...["D01", "D02", "D03", "D04"].map((subset) =>
      row("model-a", "AIHubLowQualityTelephone", subset, 0.8)),
    row("model-a", "AIHubLowQualityTelephone", "all", 0.3),
  ];
  const result = overall(rows)[0];
  assert.equal(result.dataset_count, 3);
  assert.ok(Math.abs(result.metrics.cer - 0.2) < 1e-12);
  assert.equal(vm.runInContext("state.rows.length", context), 7);
  const reversed = overall([...rows].reverse())[0];
  assert.deepEqual(reversed.metrics, result.metrics);
  assert.deepEqual(reversed.rows, result.rows);
  assert.equal(reversed.dataset_count, result.dataset_count);
});

test("incomplete models are unranked and keep their dataset results", () => {
  const results = overall([
    ...suite("model-a", 0.3),
    row("model-b", "KsponSpeech", "clean", 0.01),
    row("model-b", "KsponSpeech", "other", 0.01),
    ...["D01", "D02", "D03", "D04"].map((subset) => row("model-b", "AIHubLowQualityTelephone", subset, 0.01)),
  ]);
  assert.deepEqual(results.map((result) => result.model), ["model-a"]);
  assert.deepEqual(JSON.parse(vm.runInContext('JSON.stringify(buildOverallLeaderboard().incomplete[0].missing_slices)', context)), ["AIHub All"]);
  assert.equal(vm.runInContext('filterDatasetRows(state.rows, "KsponSpeech").length', context), 4);
  assert.equal(vm.runInContext('state.rows.length', context), 9);
});

test("aliases share one identity and a fixed representative independent of sort and input order", () => {
  const rows = suite("whisper_large_v3");
  for (const input of rows) {
    input.model_repo = "openai/whisper-large-v3";
    input._artifact = `results/${input.run_id}/leaderboard_row.json`;
    Object.assign(input.metrics.macro, {wer: 0.3, rtfx: 2});
  }
  rows[1].model = "whisper-large-v3";
  rows[1].model_repo = "https://huggingface.co/OpenAI/whisper-large-v3/";
  const legacy = {...rows[0], model: "whisper-large-v3", run_id: "readme-legacy", _artifact: "doc/submitted_results.json", metrics: {macro: {cer: 0.001, wer: 0.001, rtfx: 100}}};
  const oldRun = {...rows[0], run_id: "20250101-clean", metrics: legacy.metrics};
  const original = JSON.stringify([...rows, legacy, oldRun]);
  const result = overall(JSON.parse(original));
  assert.equal(result.length, 1);
  assert.equal(result[0].model_id, "openai/whisper-large-v3");
  assert.ok(Math.abs(result[0].metrics.cer - 0.1) < 1e-12);
  assert.deepEqual(result[0].aliases, ["whisper_large_v3", "whisper-large-v3"].sort((a, b) => a.localeCompare(b)));
  for (const metric of ["wer", "rtfx", "all_samples_cer", "cer"]) {
    vm.runInContext(`state.sortMetric = "${metric}"`, context);
    assert.deepEqual(overall(JSON.parse(original).reverse()), result);
  }
  assert.equal(vm.runInContext('JSON.stringify(state.rows)', context), JSON.stringify(JSON.parse(original).reverse()));
  assert.deepEqual(JSON.parse(vm.runInContext('JSON.stringify(modelOptions(state.rows))', context)).map((item) => item.value), ["openai/whisper-large-v3"]);
  vm.runInContext('state.model = "openai/whisper-large-v3"', context);
  assert.equal(vm.runInContext('filterDatasetRows(state.rows, "KsponSpeech").length', context), 4);
  vm.runInContext('state.model = "all"', context);
});

test("same display names from different repositories cannot complete each other's coverage", () => {
  const rows = suite("same-name");
  rows[1].model_repo = "another/same-name";
  assert.equal(overall(rows).length, 0);
  assert.equal(vm.runInContext('buildOverallLeaderboard().incomplete.length', context), 2);
});

test("aggregate run dates take precedence over alias spelling when selecting the latest run", () => {
  const rows = suite("model-a");
  rows[2].run_id = "aggregate-aaa-20261002T000000Z";
  const old = {...rows[2], run_id: "aggregate-zzz-20261001T000000999999Z", metrics: {macro: {cer: 0.001}}};
  const result = overall([...rows, old])[0];
  assert.equal(result.rows[2].run_id, rows[2].run_id);
  assert.ok(Math.abs(result.metrics.cer - 0.1) < 1e-12);
});

test("each required slice must have a full run, expected sample count and finite main CER", () => {
  for (const override of [
    {is_full_evaluation: false}, {is_full_evaluation: undefined},
    {evaluated_samples: 2999}, {total_samples: 2999}, {dataset_total_samples: 3001},
    {metrics: {macro: {cer: NaN}}},
  ]) {
    const rows = suite("model-a");
    Object.assign(rows[0], override);
    assert.equal(overall(rows).length, 0);
    assert.deepEqual(JSON.parse(vm.runInContext('JSON.stringify(buildOverallLeaderboard().incomplete[0].missing_slices)', context)), ["KsponSpeech clean"]);
  }
});

test("published data ranks nine complete models and retains distinct protocol records", () => {
  const rows = JSON.parse(fs.readFileSync("doc/leaderboard_data.json", "utf8"));
  const original = JSON.stringify(rows);
  const results = overall(rows);
  assert.equal(results.length, 9);
  assert.equal(new Set(results.map((result) => result.model_id)).size, 9);
  const whisper = results.find((result) => result.model_id === "openai/whisper-large-v3");
  const report = JSON.parse(fs.readFileSync("doc/benchmarks/server_accuracy_results_20261008.json", "utf8"));
  const verifiedSlices = report.rows.filter(row => row.model_repo === whisper.model_id
    && ((row.dataset === "KsponSpeech" && ["clean", "other"].includes(row.subset))
      || (row.dataset === "AIHubLowQualityTelephone" && row.subset === "all")));
  assert.equal(verifiedSlices.length, 3);
  assert.deepEqual(new Set(whisper.rows.map(row => row.run_id)), new Set(verifiedSlices.map(row => row.run_id)));
  const expectedCER = verifiedSlices.reduce((sum, row) => sum + row.metrics.macro.cer, 0) / 3;
  assert.ok(Math.abs(whisper.metrics.cer - expectedCER) < 1e-12);
  assert.equal(whisper.rows.some((run) => run.run_id.startsWith("readme-legacy")), false);
  assert.equal(vm.runInContext('buildOverallLeaderboard().incomplete[0].model', context), "google_speech_recognition");
  assert.equal(vm.runInContext('modelOptions(state.rows).length', context), 10);
  assert.equal(JSON.stringify(rows), original);
  assert.equal(rows.length, 71);
  assert.equal(rows.filter((row) => row.evaluation_protocol === "v1/punctuation_agnostic/cer>1.0").length, 2);
});

test("repository model names and older aliases retain readable labels", () => {
  for (const [model, label] of [
    ["openai/whisper-large-v3-turbo", "Whisper large-v3-turbo"],
    ["whisper_large_v3_turbo", "Whisper large-v3 turbo"],
    ["Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B"],
    ["qwen3_asr_0_6b", "Qwen3-ASR 0.6B"],
    ["custom/model", "custom/model"],
  ]) {
    context.inputModel = model;
    assert.equal(vm.runInContext('displayModelName(inputModel)', context), label);
  }
});

test("different or unknown protocols cannot enter Overall but remain in dataset results", () => {
  for (const changes of [
    {normalization_preset: "raw"}, {outlier_policy: {metric: "cer", threshold: 9999}},
    {evaluation_protocol: null}, {evaluation_protocol: "v2/kspon/cer>1.0"},
  ]) {
    const rows = suite("model-a"); Object.assign(rows[0], changes);
    assert.equal(overall(rows).length, 0);
    assert.match(vm.runInContext('buildOverallLeaderboard().incomplete[0].missing_slices[0]', context), /평가 규약/);
    assert.equal(vm.runInContext('filterDatasetRows(state.rows, "KsponSpeech").length', context), 2);
  }
});

test("nonstandard dataset results explain their protocol without a rank", () => {
  context.inputRow = {...row("model-a", "KsponSpeech", "clean", 0.01),
    normalization_preset: "raw", evaluation_protocol: "v1/raw/cer>1.0"};
  const html = vm.runInContext('renderDatasetRow(inputRow, 1)', context);
  assert.match(html, /정규화 없음 · CER 100% 초과 제외/);
  assert.doesNotMatch(html, /v1\/raw/);
  assert.match(html, /class="numeric">—<\/td>/);
});

test("dataset ranking and reference rows stay separate through filtering and expansion", () => {
  const elements = {};
  const page = vm.createContext({document: {
    getElementById: (id) => elements[id] ||= {innerHTML: "", textContent: "", hidden: false, open: false, setAttribute() {}},
    addEventListener: () => {},
  }});
  vm.runInContext(fs.readFileSync("site/app.js", "utf8"), page);
  page.rows = [
    {...row("model-a", "KsponSpeech", "clean", 0.2), run_id: "standard-a"},
    {...row("model-b", "KsponSpeech", "clean", 0.3), run_id: "standard-b"},
    {...row("model-a", "KsponSpeech", "clean", 0.01), run_id: "reference-a", evaluation_protocol: null},
  ];
  const original = JSON.stringify(page.rows);
  vm.runInContext('state.rows = rows; state.activeTab = "KsponSpeech"; render()', page);
  assert.equal(elements.referenceResults.open, false);
  assert.equal(elements.referenceResults.hidden, false);
  assert.equal(elements.referenceSummary.textContent, "참고 결과 1개");
  assert.match(elements.leaderboardBody.innerHTML, /standard-a/);
  assert.doesNotMatch(elements.leaderboardBody.innerHTML, /reference-a/);
  assert.match(elements.referenceBody.innerHTML, /reference-a/);
  assert.doesNotMatch(elements.referenceBody.innerHTML, /standard-a/);
  elements.referenceResults.open = true;
  vm.runInContext('state.expandedKey = "reference-a"; render()', page);
  assert.equal(elements.referenceResults.open, true);
  assert.match(elements.referenceBody.innerHTML, /평가 조건/);
  vm.runInContext('state.model = "org/model-b"; render()', page);
  assert.equal(elements.referenceResults.hidden, true);
  assert.equal(elements.referenceBody.innerHTML, "");
  assert.equal(elements.referenceResults.open, false);
  vm.runInContext('state.model = "all"; state.search = "reference-a"; render()', page);
  assert.match(elements.leaderboardBody.innerHTML, /조건에 맞는 결과가 없습니다/);
  assert.equal(elements.referenceResults.hidden, false);
  vm.runInContext('state.activeTab = "overall"; render()', page);
  assert.equal(elements.referenceResults.hidden, true);
  assert.equal(JSON.stringify(page.rows), original);
});

test("public details retain comparison conditions while raw JSON retains provenance", () => {
  const rows = JSON.parse(fs.readFileSync("doc/leaderboard_data.json", "utf8"));
  const input = rows.find((run) => run.model_repo === "openai/whisper-base" && run.subset === "clean" && run.normalization_preset === "kspon");
  const original = JSON.stringify(input);
  context.inputRow = input;
  const html = vm.runInContext('renderDatasetDetailRow(inputRow)', context);
  for (const label of ["WER", "MER", "JER", "SER", "CER · corpus", "GPU", "Batch size", "Kspon 정규화", "CER 100% 초과 제외", "원본 JSON"]) {
    assert.ok(html.includes(label), label);
  }
  assert.doesNotMatch(html, /not recorded|SHA-256|model revision|code commit|workers|canonical model id|Reproduction command|v1\/kspon/);
  assert.ok(!html.includes(input.run_id));
  assert.ok(!html.includes(input._artifact));
  assert.match(html, /href="leaderboard_data.json"/);
  assert.equal(input.reproducibility.full_accuracy.protocol_sha256, input.accuracy_validation.protocol_sha256);
  assert.match(input.reproducibility.full_accuracy.input_manifest_sha256, /^[a-f0-9]{64}$/);
  assert.equal(JSON.stringify(input), original);
  const aggregate = rows.find((run) => run.metadata_status === "aggregate");
  context.inputRow = aggregate;
  assert.notEqual(vm.runInContext('batchSizeLabel(inputRow)', context), "");
  assert.doesNotMatch(vm.runInContext('renderDatasetDetailRow(inputRow)', context), /source_runs|집계 원본 실행|SHA-256/);
});

test("overall shows all-sample CER separately without changing main CER ranking", () => {
  const a = row("model-a", "KsponSpeech", "clean", 0.1);
  const b = row("model-a", "KsponSpeech", "other", 0.3);
  const c = row("model-a", "AIHubLowQualityTelephone", "all", 0.2);
  a.metrics.all_samples_micro = { cer: 0.4 };
  b.metrics.all_samples_micro = { cer: 0.8 };
  c.metrics.all_samples_micro = { cer: 0.6 };
  a.outlier_count = 300;
  b.outlier_count = 900;
  c.outlier_count = 0;
  const result = overall([a, b, c])[0];
  assert.ok(Math.abs(result.metrics.cer - 0.2) < 1e-12);
  assert.ok(Math.abs(result.metrics.all_samples_cer - 0.6) < 1e-12);
  assert.ok(Math.abs(result.metrics.outlier_rate - 0.4 / 3) < 1e-12);
  context.inputRows = [a, b, c, ...suite("model-b", 0.3)];
  assert.equal(vm.runInContext('state.rows = inputRows; state.sortMetric = "cer"; sortOverallRows(buildOverallLeaderboard().ranked)[0].model', context), "model-a");
});

test("missing all-sample statistics are not replaced with filtered scores or partial averages", () => {
  const a = row("model-a", "KsponSpeech", "clean", 0.1);
  const b = row("model-a", "KsponSpeech", "other", 0.3);
  a.metrics.all_samples_micro = { cer: 0.4 };
  assert.equal(overall([a, b, row("model-a", "AIHubLowQualityTelephone", "all", 0.2)])[0].metrics.all_samples_cer, null);
  context.inputRow = b;
  assert.ok(vm.runInContext('Number.isNaN(metricValue(inputRow, "all_samples_cer"))', context));
  assert.match(vm.runInContext('renderDatasetRow(inputRow, 1)', context), /N\/A/);
});

test("secondary metrics also require all three slices rather than a partial average", () => {
  const rows = suite("model-a");
  rows[0].metrics.macro.rtfx = 10;
  rows[1].metrics.macro.rtfx = 20;
  const result = overall(rows)[0];
  assert.equal(result.metrics.rtfx, null);
  assert.ok(Math.abs(result.metrics.cer - 0.1) < 1e-12);
});

test("all-sample CER is independently sortable and the three quality columns are adjacent", () => {
  const a = row("model-a", "KsponSpeech", "clean", 0.1);
  const b = row("model-b", "KsponSpeech", "clean", 0.2);
  a.metrics.all_samples_micro = { cer: 0.5 };
  b.metrics.all_samples_micro = { cer: 0.3 };
  context.inputRows = [a, b];
  assert.equal(vm.runInContext('state.sortMetric = "all_samples_cer"; sortRows(inputRows)[0].model', context), "model-b");
  vm.runInContext('state.sortMetric = "cer"', context);
  assert.deepEqual(JSON.parse(vm.runInContext('JSON.stringify(datasetColumns.slice(2, 5))', context)),
    ["Main CER ↓", "Outlier rate ↓", "All-sample CER ↓"]);
  context.inputRow = a;
  const html = vm.runInContext('renderDatasetRow(inputRow, 1)', context);
  assert.equal((html.match(/<td\b/g) || []).length, vm.runInContext('datasetColumns.length', context));
});

test("display units are consistent without rounding the ranking inputs", () => {
  assert.equal(vm.runInContext('formatMetric("cer", 0.150721)', context), "15.07%");
  assert.equal(vm.runInContext('formatMetric("cer", 1.2567)', context), "125.67%");
  assert.equal(vm.runInContext('formatMetric("rtfx", 19.307)', context), "19.31×");
  assert.equal(vm.runInContext('formatMetric("latency", 0.2365)', context), "236.5 ms");
  assert.equal(vm.runInContext('formatMetric("all_samples_cer", NaN)', context), "N/A");
  assert.equal(vm.runInContext('formatMetric("cer", NaN)', context), "-");
  assert.equal(vm.runInContext('formatMetric("cer", 0)', context), "0.00%");
  context.inputRows = [row("a", "KsponSpeech", "clean", 0.10004), row("b", "KsponSpeech", "clean", 0.10003)];
  assert.equal(vm.runInContext('state.sortMetric = "cer"; sortRows(inputRows)[0].model', context), "b");
});

test("compact rows retain secondary metrics in details and surface the selected sort value", () => {
  const input = row("whisper_large_v3", "KsponSpeech", "clean", 0.1);
  Object.assign(input, {run_id: "run", gpu: "test GPU", device: "Phone", precision: "float"});
  Object.assign(input.metrics.macro, {wer: 0.25, mer: 0.2, jer: 0.15, ser: 0.6, latency: 0.25, rtfx: 4});
  context.inputRow = input;
  for (const [renderer, columns, size] of [["renderDatasetRow", "datasetColumns", 7], ["renderOnDeviceRow", "onDeviceColumns", 8]]) {
    const html = vm.runInContext(`${renderer}(inputRow, 1)`, context);
    assert.equal((html.match(/<td\b/g) || []).length, size);
    assert.equal(vm.runInContext(`${columns}.length`, context), size);
    assert.match(html, /10.00%/);
    assert.match(html, /4.00×/);
    assert.doesNotMatch(html, /metric-bar/);
  }
  const detail = vm.runInContext('renderDatasetDetailRow(inputRow)', context);
  for (const value of ["WER", "MER", "JER", "SER", "25.00%", "250 ms", "test GPU"]) assert.ok(detail.includes(value), value);
  const overallRow = overall([input, ...suite(input.model).slice(1).map((item) => ({...item, metrics: input.metrics}))])[0];
  context.inputOverall = overallRow;
  assert.equal((vm.runInContext('renderOverallRow(inputOverall, 1)', context).match(/<td\b/g) || []).length, 7);
  assert.match(vm.runInContext('renderOverallDetailRow(inputOverall)', context), /25.00%/);
  assert.match(vm.runInContext('state.sortMetric = "wer"; renderDatasetRow(inputRow, 1)', context), /WER 25.00%/);
  vm.runInContext('state.sortMetric = "cer"', context);
});

function interactivePage() {
  const elements = {};
  const page = vm.createContext({ document: {
    getElementById: (id) => elements[id] ||= {
      innerHTML: "", textContent: "", hidden: false, open: false, value: "",
      attributes: {}, listeners: {},
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener(type, listener) { this.listeners[type] = listener; },
      querySelector() { return { focus() {} }; },
    },
    querySelectorAll: () => [], addEventListener() {},
  }});
  vm.runInContext(fs.readFileSync("site/app.js", "utf8"), page);
  page.rows = JSON.parse(fs.readFileSync("doc/leaderboard_data.json", "utf8"));
  page.devices = JSON.parse(fs.readFileSync("doc/ondevice_leaderboard_data.json", "utf8"));
  const run = (code) => vm.runInContext(code, page);
  run("state.rows = rows.map(normalizeRow); state.onDeviceRows = devices.map(normalizeRow); wireControls(); render()");
  return {elements, run};
}

test("all-subset views never rank different populations, even after filtering or sorting", () => {
  const {elements, run} = interactivePage();
  for (const dataset of ["KsponSpeech", "AIHubLowQualityTelephone"]) {
    run(`state.activeTab = "${dataset}"; state.subsetByDataset[state.activeTab] = ALL_SLICES; state.search = ""; render()`);
    assert.doesNotMatch(elements.leaderboardBody.innerHTML, /rank-number|rank-leading/);
    assert.doesNotMatch(elements.rowCount.textContent, /순위/);
    const expected = run("filterDatasetRows(state.rows, state.activeTab).filter(hasStandardProtocol).length");
    assert.equal((elements.leaderboardBody.innerHTML.match(/data-compare-key=/g) || []).length, expected);
    run('state.sortMetric = "rtfx"; state.search = "clean"; render()');
    assert.doesNotMatch(elements.leaderboardBody.innerHTML, /rank-number|rank-leading/);
  }
});

test("individual dataset slices restore ranks and keep reference results unranked", () => {
  const {elements, run} = interactivePage();
  run('state.activeTab = "KsponSpeech"; state.subsetByDataset.KsponSpeech = "clean"; render()');
  assert.equal((elements.leaderboardBody.innerHTML.match(/class="rank-number/g) || []).length, 10);
  assert.equal(elements.rowCount.textContent, "10개 순위 결과");
  assert.match(elements.leaderboardBody.innerHTML, /rank-leading">1<\/span>/);
  assert.doesNotMatch(elements.referenceBody.innerHTML, /rank-number|rank-leading/);
  run('state.activeTab = AIHUB_DATASET; render()');
  assert.equal((elements.leaderboardBody.innerHTML.match(/class="rank-number/g) || []).length, 9);
  assert.equal(elements.rowCount.textContent, "9개 순위 결과");
});

test("table is the default view and chart switching preserves filters and selection", () => {
  const {elements, run} = interactivePage();
  assert.equal(elements.tableSection.hidden, false);
  assert.equal(elements.tradeoffSection.hidden, true);
  elements.searchInput.listeners.input({target: {value: "Qwen"}});
  const filteredTable = elements.leaderboardBody.innerHTML;
  assert.doesNotMatch(filteredTable, /Whisper/);
  elements.chartView.listeners.click();
  assert.equal(elements.tableSection.hidden, true);
  assert.equal(elements.tradeoffSection.hidden, false);
  assert.equal(elements.sortControl.hidden, true);
  assert.equal(elements.chartView.attributes["aria-pressed"], "true");
  assert.equal(run('state.tradeoff.points.length'), 2);
  assert.ok(run('state.tradeoff.points.every(p => p.row.model_repo.toLowerCase().startsWith("qwen/"))'));
  const selected = run("state.tradeoff.selected");
  elements.tableView.listeners.click();
  assert.equal(elements.leaderboardBody.innerHTML, filteredTable);
  assert.equal(run("state.search"), "qwen");
  assert.equal(run("state.tradeoff.selected"), selected);
  assert.equal(elements.sortControl.hidden, false);
  assert.equal(elements.tableView.attributes["aria-pressed"], "true");
});

test("column sorting and the sort menu agree on rank direction across device and server views", () => {
  const {elements, run} = interactivePage();
  const sortBy = (metric) => elements.leaderboardHead.listeners.click({
    target: {closest: () => ({getAttribute: () => metric})}, currentTarget: elements.leaderboardHead,
  });
  sortBy("rtfx");
  assert.equal(elements.sortMetric.value, "rtfx");
  assert.match(elements.leaderboardHead.innerHTML, /aria-sort="descending"><button[^>]+data-sort-metric="rtfx"/);
  const firstBySpeed = run("sortOverallRows(buildOverallLeaderboard().ranked)[0]");
  assert.ok(elements.leaderboardBody.innerHTML.indexOf(firstBySpeed.model_repo) < elements.leaderboardBody.innerHTML.indexOf("</tr>"));
  sortBy("all_samples_cer");
  assert.match(elements.leaderboardHead.innerHTML, /aria-sort="ascending"><button[^>]+data-sort-metric="all_samples_cer"/);
  run('state.activeTab = "on_device"; render()');
  sortBy("rtfx");
  assert.match(elements.leaderboardHead.innerHTML, /data-sort-metric="rtfx">QNN Macro RTFx ↑/);
  assert.equal((elements.leaderboardHead.innerHTML.match(/aria-sort=/g) || []).length, 1);
  elements.sortMetric.listeners.change({target: {value: "wer"}});
  assert.doesNotMatch(elements.leaderboardHead.innerHTML, /aria-sort=/);
  assert.match(elements.leaderboardBody.innerHTML, /sort-value">WER /);
});

test("filter reset recovers empty results without changing dataset or sort", () => {
  const {elements, run} = interactivePage();
  run('state.activeTab = "KsponSpeech"; state.subsetByDataset.KsponSpeech = "clean"; state.sortMetric = "rtfx"; render()');
  elements.modelFilter.listeners.change({target: {value: "openai/whisper-tiny"}});
  elements.searchInput.listeners.input({target: {value: "no-such-model"}});
  assert.equal(elements.resetFilters.disabled, false);
  assert.match(elements.leaderboardBody.innerHTML, /조건에 맞는 (순위 )?결과가 없습니다/);
  elements.resetFilters.listeners.click();
  assert.equal(elements.searchInput.value, "");
  assert.equal(elements.modelFilter.value, "all");
  assert.equal(elements.resetFilters.disabled, true);
  assert.equal(run("state.activeTab"), "KsponSpeech");
  assert.equal(run("activeDatasetSubset()"), "clean");
  assert.equal(run("state.sortMetric"), "rtfx");
  assert.doesNotMatch(elements.leaderboardBody.innerHTML, /조건에 맞는 (순위 )?결과가 없습니다/);
});
