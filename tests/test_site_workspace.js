const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const {test} = require("node:test");

function page() {
  const elements = {}, events = {};
  const window = {location: new URL("https://example.test/leaderboard/"),
    history: {replaceState(_state, _title, url) { window.location = new URL(url); }},
    addEventListener(name, callback) { events[name] = callback; }};
  const context = vm.createContext({URL, URLSearchParams, window, navigator: {}, document: {
    getElementById: (id) => elements[id] ||= {innerHTML: "", textContent: "", value: "", hidden: false,
      open: false, attributes: {}, listeners: {}, setAttribute(k, v) { this.attributes[k] = v; },
      addEventListener(k, fn) { this.listeners[k] = fn; }, querySelectorAll() { return []; },
      querySelector() { return {focus() {}}; }, focus() { this.focused = true; }, select() { this.selected = true; }},
    addEventListener() {}, querySelectorAll() { return []; },
  }});
  vm.runInContext(fs.readFileSync("site/app.js", "utf8"), context);
  context.server = JSON.parse(fs.readFileSync("doc/leaderboard_data.json", "utf8"));
  context.device = JSON.parse(fs.readFileSync("doc/ondevice_leaderboard_data.json", "utf8"));
  const run = (code) => vm.runInContext(code, context);
  run("state.rows = server.map(normalizeRow); state.onDeviceRows = device.map(normalizeRow); wireControls(); populateModelFilter(); render()");
  const read = (code) => JSON.parse(run(`JSON.stringify(${code})`));
  return {context, elements, window, events, run, read};
}

test("execution filters require every source condition and never guess missing precision", () => {
  const p = page();
  p.context.mixed = {gpu: "GPU", reproducibility: {source_runs: [8, 32].map((batch) =>
    ({reproducibility: {execution: {batch_size: batch}}}))}};
  p.run('state.execution.batch = "8"');
  assert.equal(p.run("matchesExecution(mixed)"), false);
  p.run("mixed.reproducibility.source_runs[1].reproducibility.execution.batch_size = 8");
  assert.equal(p.run("matchesExecution(mixed)"), true);
  p.run("delete mixed.reproducibility.source_runs[1].reproducibility.execution.batch_size");
  assert.equal(p.run("matchesExecution(mixed)"), false);
  assert.deepEqual(p.read('executionValues({model: "float16_model"}, "precision")'), ["__unknown__"]);
  assert.deepEqual(p.read('executionValues({reproducibility: {source_runs: []}}, "batch")'), ["__unknown__"]);
});

test("published speed values remain sortable without ranks across unverified or mixed conditions", () => {
  const p = page();
  // Keep the real speed values while making uncertainty explicit; newly verified
  // publications must not be required to remain historically unverified forever.
  p.run("state.rows.forEach(row => { row.reproducibility = {}; })");
  for (const tab of ["overall", "KsponSpeech", "on_device"]) {
    p.context.tab = tab;
    p.run("state.activeTab = tab; state.sortMetric = 'rtfx'; state.subsetByDataset.KsponSpeech = 'clean'; render()");
    assert.doesNotMatch(p.elements.leaderboardBody.innerHTML, /class="rank-number/);
    assert.doesNotMatch(p.elements.leaderboardBody.innerHTML, />null</);
    p.run("state.sortMetric = 'cer'; render()");
    assert.match(p.elements.leaderboardBody.innerHTML, /class="rank-number/);
  }
  p.run("state.activeTab = 'on_device'; state.execution.precision = 'float'; state.sortMetric = 'rtfx'; render()");
  assert.match(p.elements.leaderboardBody.innerHTML, /class="rank-number/);
});

test("Overall filters keep fixed representatives and all three contributors", () => {
  const p = page();
  p.run('state.rows = state.rows.filter(row => canonicalModelId(row) === "openai/whisper-base"); state.rows.forEach(row => row.reproducibility = {execution: {batch_size: 8}})');
  const before = p.read("buildOverallLeaderboard().ranked[0]");
  assert.equal(before.rows.length, 3);
  p.run('state.execution.batch = "8"');
  assert.equal(p.run("filterOverallRows(buildOverallLeaderboard().ranked).length"), 1);
  p.run('state.rows.find(row => row.run_id === buildOverallLeaderboard().ranked[0].rows[0].run_id).reproducibility.execution.batch_size = 32');
  assert.equal(p.run("filterOverallRows(buildOverallLeaderboard().ranked).length"), 0);
  // An older matching run must not replace the fixed representative.
  p.run('state.rows.push({...buildOverallLeaderboard().ranked[0].rows[0], run_id: "20000101-old", reproducibility: {execution: {batch_size: 8}}})');
  assert.equal(p.run("filterOverallRows(buildOverallLeaderboard().ranked).length"), 0);
  assert.deepEqual(p.read("buildOverallLeaderboard().ranked[0].metrics"), before.metrics);
});

test("device precision filters agree between table and chart, and reset clears conditions", () => {
  const p = page();
  p.run('state.activeTab = "on_device"; state.execution.precision = "w8a16"; render()');
  assert.equal(p.run("filterOnDeviceRows(state.onDeviceRows).length"), 1);
  assert.equal(p.run("tradeoffSourceRows().length"), 1);
  assert.equal(p.elements.batchControl.hidden, true);
  assert.match(p.elements.leaderboardBody.innerHTML, /whisper_small_quantized/);
  p.run('state.execution.precision = "float"; render()');
  assert.equal(p.run("filterOnDeviceRows(state.onDeviceRows).length"), 4);
  p.run("resetFilters()");
  assert.equal(p.run("filterOnDeviceRows(state.onDeviceRows).length"), 5);
  assert.equal(p.run("hasExecutionFilter()"), false);
});

test("comparison caps at four, preserves order through sorting, and prunes hidden selections", () => {
  const p = page();
  p.run("state.comparison = comparisonCandidates().map(comparisonKey); state.comparisonOpen = true; render()");
  assert.equal(p.run("state.comparison.length"), 4);
  assert.match(p.elements.leaderboardBody.innerHTML, / disabled/);
  const keys = p.read("state.comparison");
  p.run('state.sortMetric = "rtfx"; render()');
  assert.deepEqual(p.read("state.comparison"), keys);
  p.run('state.model = comparisonCandidates().find(row => row.key === state.comparison[0]).model_id; render()');
  assert.equal(p.run("state.comparison.length"), 1);
  assert.equal(p.elements.comparisonToggle.disabled, true);
  assert.equal(p.elements.comparisonSection.hidden, true);
});

test("float and quantized runs remain distinct comparison columns with QNN scope", () => {
  const p = page();
  p.run('state.activeTab = "on_device"; state.comparison = state.onDeviceRows.filter(row => row.model.includes("small")).map(comparisonKey); state.comparisonOpen = true; render()');
  assert.equal(p.run("state.comparison.length"), 2);
  const html = p.elements.comparisonContent.innerHTML;
  for (const label of ["w8a16", "float", "QNN Macro RTFx", "전처리·전송·토큰화 제외", "실행 조건 다름"]) assert.ok(html.includes(label), label);
  assert.equal(p.elements.comparisonSection.hidden, false);
});

test("comparison separates dataset slices, leaves absent cells blank, and escapes metadata", () => {
  const p = page();
  p.run('state.activeTab = "KsponSpeech"; state.rows = state.rows.filter(row => canonicalModelId(row) === "openai/whisper-base" && hasStandardProtocol(row) && row.dataset === "KsponSpeech"); state.rows[0].model = "<script>bad</script>"; state.comparison = state.rows.map(comparisonKey); state.comparisonOpen = true; render()');
  const html = p.elements.comparisonContent.innerHTML;
  assert.match(html, /KsponSpeech clean/);
  assert.match(html, /KsponSpeech other/);
  assert.match(html, />—<\/td>/);
  assert.match(html, /&lt;script&gt;/);
  assert.doesNotMatch(html, /<script>/);
  assert.doesNotMatch(html, /Overall/);
});

test("shared URL round-trips dataset, conditions, comparison, sort and chart selection", () => {
  const p = page();
  p.run('state.activeTab = "on_device"; state.execution.precision = "float"; state.search = "whisper"; state.sortMetric = "rtfx"; render(); state.comparison = comparisonCandidates().slice(0, 2).map(comparisonKey); state.comparisonOpen = true; state.tradeoff.selected = state.tradeoff.points[1].row.run_id; state.tradeoff.metric = "all_samples_cer"; state.tradeoff.condition = tradeoffCondition(state.tradeoff.points[1].row).key; setView("chart"); render()');
  const expected = p.read("({tab:state.activeTab,execution:state.execution,comparison:state.comparison,open:state.comparisonOpen,search:state.search,sort:state.sortMetric,chart:state.tradeoff})");
  const href = p.window.location.href;
  p.run('restoreUrlState(""); populateModelFilter(); render()');
  p.window.location = new URL(href);
  p.events.popstate();
  assert.deepEqual(p.read("({tab:state.activeTab,execution:state.execution,comparison:state.comparison,open:state.comparisonOpen,search:state.search,sort:state.sortMetric,chart:state.tradeoff})"), expected);
  assert.equal(p.window.location.href, href);
  assert.equal(p.elements.searchInput.value, "whisper");
  assert.equal(p.elements.sortMetric.value, "rtfx");
});

test("shared table URL retains a configured chart slice even when its first point is selected", () => {
  const p = page();
  p.run('setView("chart"); state.tradeoff.slice = JSON.stringify(["KsponSpeech", "other"]); render(); setView("table")');
  const search = p.window.location.search;
  p.context.sharedSearch = search;
  p.run('restoreUrlState(sharedSearch); populateModelFilter(); render()');
  assert.equal(p.run("state.tradeoff.slice"), '["KsponSpeech","other"]');
  assert.equal(p.run("state.view"), "table");
});

test("invalid and stale URL values normalize safely, preserving unrelated query/hash", () => {
  const p = page();
  assert.equal(p.window.location.search, "");
  p.window.location = new URL("https://example.test/leaderboard/?utm_source=test&tab=bad&sort=__proto__&model=missing&hardware=missing&compare=missing&compareOpen=1&metric=bad&slice=%FF#evaluation-method");
  p.events.popstate();
  assert.equal(p.run("state.activeTab"), "overall");
  assert.equal(p.run("state.sortMetric"), "cer");
  assert.equal(p.run("state.model"), "all");
  assert.equal(p.run("hasExecutionFilter()"), false);
  assert.deepEqual(p.read("state.comparison"), []);
  assert.equal(p.window.location.searchParams.get("utm_source"), "test");
  assert.equal(p.window.location.hash, "#evaluation-method");
  assert.equal(p.window.location.search.includes("missing"), false);
});

test("share copy reports success only after clipboard write and provides selectable fallback", async () => {
  const p = page();
  let copied;
  p.context.navigator.clipboard = {writeText: async (value) => { copied = value; }};
  await p.run("copyShareLink()");
  assert.equal(copied, p.window.location.href);
  assert.equal(p.elements.shareStatus.textContent, "링크 복사됨");
  p.context.navigator.clipboard.writeText = async () => { throw new Error("unavailable"); };
  await p.run("copyShareLink()");
  assert.equal(p.elements.shareUrl.value, copied);
  assert.equal(p.elements.shareUrl.hidden, false);
  assert.equal(p.elements.shareUrl.selected, true);
  p.run('state.search = "new"; render()');
  assert.equal(p.elements.shareFeedback.hidden, true);
});
