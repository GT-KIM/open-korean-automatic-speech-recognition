const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { test } = require("node:test");

function page() {
  const elements = {};
  const context = vm.createContext({ document: {
    getElementById: (id) => elements[id] ||= { innerHTML: "", textContent: "", hidden: false },
    addEventListener: () => {},
  }});
  vm.runInContext(fs.readFileSync("site/app.js", "utf8"), context);
  return { context, elements,
    run: (code) => vm.runInContext(code, context),
    json: (code) => JSON.parse(vm.runInContext(`JSON.stringify(${code})`, context)),
  };
}

function row(model, error, speed, extra = {}) {
  return {
    model, model_repo: `org/${model}`, run_id: `${model}-20261001T000000Z`,
    dataset: "KsponSpeech", subset: "clean", is_full_evaluation: true, evaluated_samples: 3000,
    evaluation_protocol: "v1/kspon/cer>1.0", normalization_preset: "kspon",
    outlier_policy: {metric: "cer", threshold: 1}, outlier_count: 0,
    gpu: "GPU A", torch: "2.11", cuda: "12.8",
    reproducibility: { execution: {batch_size: 8, num_workers: 0, warmup_samples: 10},
      environment: {platform: "Linux"},
      inference: {effective_dtype: "float16", backend: "transformers", performance_scope: "asr_transcribe_call", batching: "native", backend_batch_size: 8} },
    metrics: {macro: {cer: error, rtfx: speed}, all_samples_micro: {cer: error + 0.1}},
    ...extra,
  };
}

test("published Nemotron remains a distinct runtime and shows native RNNT diagnostics", () => {
  const p = page();
  p.context.rows = JSON.parse(fs.readFileSync("doc/leaderboard_data.json", "utf8"))
    .filter(row => row.dataset === "KsponSpeech" && row.subset === "clean" && row.curated_speed);
  p.run("state.rows = rows; renderTradeoff()");
  const nemotron = p.context.rows.find(row => row.model_repo.startsWith("nvidia/nemotron"));
  p.context.nemotron = nemotron;
  assert.match(p.elements.tradeoffCondition.innerHTML, /Transformers 5.13.0/);
  assert.match(p.elements.tradeoffCondition.innerHTML, /Transformers 4.57.6/);
  assert.equal(p.run("state.tradeoff.points.filter(p => p.frontier).length"), 0);
  p.run("state.tradeoff.selected = nemotron.run_id; renderTradeoffDetail()");
  assert.match(p.elements.tradeoffDetail.innerHTML, /프레임 강제 진행/);
  assert.match(p.elements.tradeoffDetail.innerHTML, /B1·B4 출력 차이/);
  assert.doesNotMatch(p.elements.tradeoffDetail.innerHTML, /상한 종료|undefined|NaN/);
  p.run("state.tradeoff.condition = speedCondition(nemotron, 'b4').key; renderTradeoff()");
  assert.equal(p.run("state.tradeoff.points.length"), 1);
  p.run("state.tradeoff.configured = true; state.tradeoff.speed = 'b1'; state.tradeoff.condition = 'all'; renderTradeoff(); state.tradeoff.selected = nemotron.run_id; renderTradeoffDetail()");
  assert.match(p.elements.tradeoffDetail.innerHTML, /4 \/ 1/);
  assert.match(p.elements.tradeoffDetail.innerHTML, /B4 정확도/);
});

test("curated tracks use separate measured speed, matching cohorts, cap counts and batch-1 latency", () => {
  const p = page();
  const measured = {status: "verified", environment_id: "server", image_id: "image", source_sha256: "source",
    executed_protocol_sha256: "protocol", group: "clean", samples_per_repeat: 256, repetitions: 3, gpu: "GPU A",
    tracks: {b1: {batch_size: 1, throughput_rtfx: {median: 20, min: 19, max: 21},
      request_latency_p95_ms: {median: 120}, token_limit: 3, measured_generations: 768, token_limit_rate: 3 / 768},
    b4: {batch_size: 4, throughput_rtfx: {median: 45, min: 44, max: 46},
      token_limit: 6, measured_generations: 768, token_limit_rate: 6 / 768}}};
  p.context.rows = [row("a", .1, 123, {curated_speed: measured}), row("b", .2, 999, {curated_speed: structuredClone(measured)}),
    row("old", .01, 9999)];
  p.run("state.rows = rows; renderTradeoff()");
  assert.equal(p.run("state.tradeoff.speed"), "b4");
  assert.deepEqual(p.json("state.tradeoff.points.map(p => p.speed)"), [45, 45]);
  assert.match(p.elements.tradeoffPlot.innerHTML, /RTFx \(×\)/);
  assert.doesNotMatch(p.elements.tradeoffPlot.innerHTML, /Macro RTFx/);
  assert.equal(p.elements.tradeoffCondition.value, p.run("speedCondition(rows[0], 'b4').key"));
  assert.deepEqual(p.json("state.tradeoff.points.filter(p => p.frontier).map(p => p.row.model)"), ["a"]);
  assert.match(p.elements.tradeoffDetail.innerHTML, /120.0 ms/);
  assert.match(p.elements.tradeoffDetail.innerHTML, /6 \/ 768/);
  p.run("state.tradeoff.configured = true; state.tradeoff.condition = speedCondition(rows[0], 'b4').key; renderTradeoff()");
  assert.deepEqual(p.json("state.tradeoff.points.filter(p => p.frontier).map(p => p.row.model)"), ["a"]);
  p.run("state.tradeoff.speed = 'b1'; state.tradeoff.condition = 'all'; renderTradeoff()");
  assert.deepEqual(p.json("state.tradeoff.points.map(p => p.speed)"), [20, 20]);
  assert.match(p.elements.tradeoffDetail.innerHTML, /4 \/ 1/);
  p.context.URL = URL;
  p.context.URLSearchParams = URLSearchParams;
  const url = p.run("workspaceUrl('https://example.test/')");
  assert.equal(new URL(url).searchParams.get('speed'), 'b1');
  p.context.query = new URL(url).search;
  p.run("restoreUrlState(query)");
  assert.equal(p.run("state.tradeoff.speed"), 'b1');
  p.run("state.tradeoff.speed = 'rtfx'; renderTradeoff()");
  assert.deepEqual(p.json("state.tradeoff.points.map(p => p.speed)"), [123, 999, 9999]);
  assert.match(p.elements.tradeoffPlot.innerHTML, /Macro RTFx \(×\)/);
  assert.match(p.elements.tradeoffDetail.innerHTML, /<dt>Macro RTFx<\/dt>/);
  p.run("state.activeTab = 'on_device'; state.onDeviceRows = rows; renderTradeoff()");
  assert.equal(p.run("state.tradeoff.speed"), "rtfx");
});

test("single verified condition is selected automatically without pinning later filters", () => {
  const p = page();
  p.context.rows = [row("a", .1, 10), row("b", .2, 20)];
  p.run("state.rows = rows; renderTradeoff()");
  const key = p.run("tradeoffCondition(rows[0]).key");
  assert.equal(p.elements.tradeoffCondition.value, key);
  assert.equal(p.elements.tradeoffCondition.disabled, true);
  assert.match(p.elements.tradeoffPlot.innerHTML, /class="tradeoff-point on-frontier/);
  assert.match(p.elements.tradeoffModels.innerHTML, /class="tradeoff-model on-frontier/);
  assert.doesNotMatch(p.elements.tradeoffPlot.innerHTML, /<polyline/);
  assert.equal(p.elements.tradeoffNote.hidden, true);

  p.context.extra = row("c", .15, 50, {gpu: "GPU B"});
  p.run("state.rows = [...rows, extra]; renderTradeoff()");
  assert.equal(p.elements.tradeoffCondition.value, "all");
  assert.equal(p.elements.tradeoffCondition.disabled, false);
  assert.equal(p.run("state.tradeoff.points.length"), 3);
  assert.doesNotMatch(p.elements.tradeoffPlot.innerHTML, /on-frontier/);

  p.context.key = key;
  p.run("state.tradeoff.condition = key; renderTradeoff()");
  assert.equal(p.elements.tradeoffCondition.value, key);
  assert.equal(p.run("state.tradeoff.points.length"), 2);
  assert.match(p.elements.tradeoffPlot.innerHTML, /class="tradeoff-point on-frontier/);
  p.run("state.rows = [extra]; renderTradeoff()");
  assert.equal(p.elements.tradeoffCondition.value, p.run("tradeoffCondition(extra).key"));
  assert.equal(p.run("state.tradeoff.points.length"), 1);
  assert.doesNotMatch(p.elements.tradeoffPlot.innerHTML, /on-frontier/);
});

test("single incomplete condition is not selected or compared automatically", () => {
  const p = page();
  p.context.rows = [row("a", .1, 10, {reproducibility: {}}), row("b", .2, 20, {reproducibility: {}})];
  p.run("state.rows = rows; renderTradeoff()");
  assert.equal(p.elements.tradeoffCondition.value, "all");
  assert.equal(p.elements.tradeoffCondition.disabled, false);
  assert.equal(p.run("state.tradeoff.points.some(p => p.frontier)"), false);
  assert.doesNotMatch(p.elements.tradeoffPlot.innerHTML, /on-frontier/);
  p.run("state.rows = []; renderTradeoff()");
  assert.equal(p.elements.tradeoffCondition.value, "all");
  assert.equal(p.run("state.tradeoff.points.length"), 0);
});

test("Pareto minimizes CER and maximizes speed, preserves ties and raw precision", () => {
  const p = page();
  p.context.points = [
    {id: "accurate", error: 0.1, speed: 1}, {id: "fast", error: 0.3, speed: 4},
    {id: "middle", error: 0.2, speed: 2}, {id: "tie", error: 0.2, speed: 2},
    {id: "slower", error: 0.2, speed: 1}, {id: "worse", error: 0.4, speed: 4},
    {id: "rounding", error: 0.200001, speed: 2},
  ];
  assert.deepEqual(p.json("paretoFrontier(points).map(p => p.id)"), ["accurate", "middle", "tie", "fast"]);
  assert.equal(p.run("points.length"), 7);
});

test("aggregate chart details show recorded precision and retain missing-source uncertainty", () => {
  const p = page();
  p.context.aggregate = JSON.parse(fs.readFileSync("doc/leaderboard_data.json", "utf8"))
    .find(row => row.model_repo === "openai/whisper-base" && row.subset === "all");
  p.run("state.rows = [aggregate]; renderTradeoff()");
  assert.match(p.elements.tradeoffDetail.innerHTML, /<dt>정밀도<\/dt><dd>bfloat16<\/dd>/);
  p.run("aggregate.reproducibility.source_runs[3].reproducibility.inference.effective_dtype = null; renderTradeoff()");
  assert.match(p.elements.tradeoffDetail.innerHTML, /<dt>정밀도<\/dt><dd>bfloat16 \/ 미기록<\/dd>/);
});

test("chart separates slices and conditions and selects latest runs without score selection", () => {
  const p = page();
  const latest = row("a", 0.2, 10);
  p.context.rows = [latest, row("b", 0.1, 5),
    {...latest, run_id: "a-20250101T000000Z", metrics: {macro: {cer: 0.001, rtfx: 500}}},
    row("a", 0.3, 15, {gpu: "GPU B", run_id: "different-device"}),
    row("other-slice", 0.1, 30, {subset: "other"}),
    row("legacy", 0.01, 30, {evaluation_protocol: null}),
    row("partial", 0.01, 30, {is_full_evaluation: false}),
  ];
  const original = JSON.stringify(p.context.rows);
  p.run("state.rows = rows; renderTradeoff()");
  assert.deepEqual(p.json("state.tradeoff.points.map(p => p.row.run_id)"), [latest.run_id, "different-device", "b-20261001T000000Z"]);
  assert.doesNotMatch(p.elements.tradeoffPlot.innerHTML, /on-frontier/);
  p.run("state.tradeoff.condition = tradeoffCondition(rows[0]).key; renderTradeoff()");
  assert.equal(p.run("state.tradeoff.points.length"), 2);
  assert.match(p.elements.tradeoffPlot.innerHTML, /class="tradeoff-point on-frontier/);
  p.run('state.tradeoff.slice = tradeoffSliceKey(rows[4]); state.tradeoff.condition = "all"; renderTradeoff()');
  assert.equal(p.run("state.tradeoff.points.length"), 1);
  assert.equal(p.run("state.tradeoff.points[0].row.subset"), "other");
  assert.equal(JSON.stringify(p.context.rows), original);
});

test("batch, runtime, precision, platform and mixed aggregates cannot share a frontier", () => {
  const p = page();
  p.context.base = row("a", 0.2, 10);
  const originalKey = p.run("tradeoffCondition(base).key");
  for (const extra of [
    {reproducibility: {execution: {batch_size: 4}}}, {torch: "different"},
    {reproducibility: {...p.context.base.reproducibility, inference: {...p.context.base.reproducibility.inference, effective_dtype: "int8"}}},
    {reproducibility: {execution: {batch_size: 8}, environment: {platform: "Windows"}}},
    {performance_scope: "graph only"},
  ]) {
    p.context.changed = row("a", 0.2, 10, extra);
    assert.notEqual(p.run("tradeoffCondition(changed).key"), originalKey);
  }
  p.context.mixed = row("a", 0.2, 10, {reproducibility: {source_runs: [
    {reproducibility: {execution: {batch_size: 4}}}, {reproducibility: {execution: {batch_size: 8}}},
  ]}});
  assert.equal(p.run("tradeoffCondition(mixed).complete"), false);
  p.context.missing = row("b", 0.1, 5, {reproducibility: {}});
  p.run("state.rows = [missing, {...missing, model: 'c', model_repo: 'org/c', run_id: 'c'}]; state.tradeoff.condition = tradeoffCondition(missing).key; renderTradeoff()");
  assert.match(p.elements.tradeoffNote.textContent, /미확인/);
  assert.equal(p.run("state.tradeoff.points.some(p => p.frontier)"), false);
});

test("omit invalid coordinates while retaining zero CER, slow RTFx and CER above 100%", () => {
  const p = page();
  p.context.rows = [row("zero-error", 0, 0.25), row("large-error", 1.5, 2),
    row("no-speed", 0.2, null), row("zero-speed", 0.2, 0), row("negative-speed", 0.2, -1),
    row("no-error", null, 10), row("nan", NaN, 10), row("inf", 0.2, Infinity)];
  p.run("state.rows = rows; renderTradeoff()");
  assert.equal(p.run("state.tradeoff.points.length"), 2);
  assert.match(p.elements.tradeoffStatus.textContent, /6개 결과 제외/);
  assert.doesNotMatch(p.elements.tradeoffPlot.innerHTML, /NaN|Infinity/);
  assert.match(p.elements.tradeoffModels.innerHTML, /150.00%/);
  p.run("rows[0].metrics.all_samples_micro = {}; state.tradeoff.metric = 'all_samples_cer'; renderTradeoff()");
  assert.ok(!p.json("state.tradeoff.points.map(p => p.row.model)").includes("zero-error"));
});

test("filters and empty states clear selection; singleton and identical coordinates stay finite", () => {
  const p = page();
  p.context.rows = [row("a", 0.1, 10), row("b", 0.1, 10)];
  p.run("state.rows = rows; renderTradeoff()");
  assert.doesNotMatch(p.elements.tradeoffPlot.innerHTML, /NaN|Infinity/);
  p.run("state.model = 'org/a'; renderTradeoff()");
  assert.deepEqual(p.json("state.tradeoff.points.map(p => p.row.model)"), ["a"]);
  assert.doesNotMatch(p.elements.tradeoffPlot.innerHTML, /NaN|Infinity/);
  p.run("state.model = 'all'; state.search = 'not-found'; renderTradeoff()");
  assert.match(p.elements.tradeoffPlot.innerHTML, /조건에 맞는/);
  assert.equal(p.elements.tradeoffModels.innerHTML, "");
  assert.equal(p.run("state.tradeoff.selected"), null);
  p.run("state.search = ''; state.activeTab = 'KsponSpeech'; state.subsetByDataset.KsponSpeech = 'other'; renderTradeoff()");
  assert.equal(p.run("state.tradeoff.points.length"), 0);
});

test("CER switch recomputes frontier without changing the measured speed", () => {
  const p = page();
  p.context.rows = [row("a", 0.1, 10), row("b", 0.2, 5)];
  p.context.rows[0].metrics.all_samples_micro.cer = 0.4;
  p.run("state.rows = rows; state.tradeoff.condition = tradeoffCondition(rows[0]).key; renderTradeoff()");
  assert.deepEqual(p.json("state.tradeoff.points.filter(p => p.frontier).map(p => p.row.model)"), ["a"]);
  p.run("state.tradeoff.metric = 'all_samples_cer'; renderTradeoff()");
  assert.deepEqual(p.json("state.tradeoff.points.filter(p => p.frontier).map(p => p.row.model)"), ["a", "b"]);
  assert.match(p.elements.tradeoffPlot.innerHTML, /All-sample CER/);
  assert.deepEqual(p.json("state.tradeoff.points.map(p => p.speed)"), [10, 5]);
});

test("published device results separate precision and retain QNN measurement scope", () => {
  const p = page();
  p.context.rows = JSON.parse(fs.readFileSync("doc/ondevice_leaderboard_data.json", "utf8"));
  p.run("state.activeTab = 'on_device'; state.onDeviceRows = rows; renderTradeoff()");
  assert.equal(p.run("state.tradeoff.points.length"), 5);
  assert.match(p.elements.tradeoffPlot.innerHTML, /QNN Macro RTFx/);
  p.run("state.tradeoff.condition = tradeoffCondition(rows.find(r => r.precision === 'float')).key; renderTradeoff()");
  assert.equal(p.run("state.tradeoff.points.length"), 4);
  assert.match(p.elements.tradeoffDetail.innerHTML, /전처리·전송·토큰화 제외/);
});

test("published server results stay per-slice; metadata is escaped in accessible controls", () => {
  const p = page();
  p.context.rows = JSON.parse(fs.readFileSync("doc/leaderboard_data.json", "utf8"));
  p.run("state.rows = rows; state.tradeoff.speed = 'rtfx'; state.tradeoff.configured = true; renderTradeoff()");
  assert.equal(p.run("state.tradeoff.points.length"), 10);
  assert.ok(p.run("state.tradeoff.points.every(p => p.row.subset === 'clean')"));
  for (const condition of p.json("[...new Set(state.tradeoff.points.map(p => tradeoffCondition(p.row).key))]")) {
    p.context.condition = condition;
    p.run("state.tradeoff.condition = condition; renderTradeoff()");
    if (!p.run("tradeoffCondition(state.tradeoff.points[0].row).complete")) {
      assert.equal(p.run("state.tradeoff.points.some(p => p.frontier)"), false);
    }
  }
  p.context.malicious = row('<img src=x onerror="alert(1)">', 0.1, 10, {gpu: '<script>alert(1)</script>'});
  p.run("state.rows = [malicious]; renderTradeoff()");
  for (const element of [p.elements.tradeoffPlot, p.elements.tradeoffModels, p.elements.tradeoffDetail]) {
    assert.doesNotMatch(element.innerHTML, /<img|<script/);
  }
  assert.match(p.elements.tradeoffPlot.innerHTML, /role="button" tabindex="0"/);
  assert.match(p.elements.tradeoffModels.innerHTML, /aria-pressed="true"/);
});

test("missing precision or scope preserves points but prevents inferred speed superiority", () => {
  const p = page();
  for (const field of ["effective_dtype", "performance_scope", "backend"]) {
    for (const value of [null, "", "unknown"]) {
      const a = row("a", 0.1, 20), b = row("b", 0.2, 10);
      for (const r of [a, b]) {
        r.reproducibility.inference[field] = value;
        r.reproducibility.model_config = {dtype: "float16"};
      }
      p.context.rows = [a, b];
      p.run("state.rows = rows; state.tradeoff.condition = tradeoffCondition(rows[0]).key; renderTradeoff()");
      assert.equal(p.run("state.tradeoff.points.length"), 2);
      assert.equal(p.run("state.tradeoff.points.some(p => p.frontier)"), false);
    }
  }
});

test("backend inner batches partition otherwise identical server runs", () => {
  const p = page();
  p.context.rows = [row("a", 0.1, 20), row("b", 0.2, 10)];
  p.run("rows.forEach(r => {r.reproducibility.inference.batching = 'backend_chunked'; delete r.reproducibility.inference.backend_batch_size})");
  assert.equal(p.run("tradeoffCondition(rows[0]).complete"), false);
  p.run("rows[0].reproducibility.inference.backend_batch_size = 1; rows[1].reproducibility.inference.backend_batch_size = 4");
  assert.equal(p.run("tradeoffCondition(rows[0]).complete"), true);
  assert.notEqual(p.run("tradeoffCondition(rows[0]).key"), p.run("tradeoffCondition(rows[1]).key"));
});

test("speed sorting only assigns ranks within verified conditions, including Overall slices", () => {
  const p = page();
  p.context.rows = [row("a", 0.1, 20), row("b", 0.2, 10)];
  p.run("state.sortMetric = 'rtfx'");
  assert.equal(p.run("canRankSpeed(rows)"), true);
  p.run("rows[1].reproducibility.inference.effective_dtype = null");
  assert.equal(p.run("canRankSpeed(rows)"), false);
  assert.equal(p.run("canRankSpeed(rows.map(row => ({rows: [row]})))"), false);
  p.run("state.sortMetric = 'cer'");
  assert.equal(p.run("canRankSpeed(rows)"), true);
});
