/*
 * Fraud Detection — static live demo
 * Every number is read from site/data/*.json, which src/site/build.py writes
 * from the pipeline's own outputs. Nothing here calls a model.
 */
(() => {
  "use strict";

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const SVG = "http://www.w3.org/2000/svg";

  // ── formatting (display only) ─────────────────────────────
  const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
  const gbp = (n) => `£${Math.round(n).toLocaleString("en-GB")}`;
  const int = (n) => Math.round(n).toLocaleString("en-GB");
  const thr = (t) => t.toFixed(4);
  const prob = (p) => {
    if (p >= 0.9999) return ">99.99%";
    if (p >= 0.1) return pct(p, 2);
    if (p >= 0.001) return pct(p, 2);
    if (p >= 0.0001) return pct(p, 3);
    return "<0.01%";
  };
  const sign = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);

  const bind = (key, text) => $$(`[data-bind="${key}"]`).forEach((el) => (el.textContent = text));

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    children.forEach((c) => node.append(c));
    return node;
  }

  function svg(tag, attrs = {}) {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  // count up once, final value shown at once with reduced motion
  function countUp(node, target, format, duration = 1100) {
    if (reduced) { node.textContent = format(target); return; }
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      node.textContent = format(target * eased);
      if (t < 1) requestAnimationFrame(step);
      else node.textContent = format(target);
    };
    requestAnimationFrame(step);
  }

  // grow widths from 0 on the next frame so the CSS transition runs
  function grow(node, width) {
    if (reduced) { node.style.width = width; return; }
    node.style.width = "0";
    requestAnimationFrame(() => requestAnimationFrame(() => (node.style.width = width)));
  }

  const resizers = [];
  let resizeTimer;
  const onResize = (fn) => resizers.push(fn);
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => resizers.forEach((fn) => fn()), 150);
  });

  // ── chrome ────────────────────────────────────────────────
  const topbar = $("#topbar");
  const onScroll = () => topbar.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  function revealOnScroll() {
    const items = $$(".reveal");
    if (reduced || !("IntersectionObserver" in window)) { items.forEach((n) => n.classList.add("in")); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.08 });
    items.forEach((n) => io.observe(n));
  }

  // ── hero ──────────────────────────────────────────────────
  function renderHero(o) {
    const g = o.glance;
    bind("threshold", thr(o.threshold));
    countUp($('[data-bind="g-caught"]'), g.caught, int);
    bind("g-frauds", int(g.frauds));
    bind("g-recall", `${pct(g.recall)} of test-set fraud`);
    countUp($('[data-bind="g-precision"]'), g.precision, (x) => pct(x));
    countUp($('[data-bind="g-alerts"]'), g.alerts_per_10k, int);
    countUp($('[data-bind="g-ks"]'), g.ks, (x) => x.toFixed(3));
    bind("g-auprc", g.auprc.toFixed(3));
  }

  // ── 01 score a transaction ────────────────────────────────
  const SAMPLE_HELP = {
    fraud: "A real fraud the model catches — the case the system exists for",
    legitimate: "A typical legitimate payment — most traffic looks like this, and should pass untouched",
    borderline: "The nearest approval: the highest-scoring payment still let through — where a fixed line is most fragile",
    false_alarm: "A legitimate payment the model blocks — the cost of catching more fraud, and the hardest notice to defend",
  };

  function renderSample(kind, samples, o) {
    const s = samples[kind];
    const declined = s.decision === "DECLINE";

    $$("#score .tab").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.sample === kind)));
    $("#sample-help").textContent = SAMPLE_HELP[kind];

    $("#v-meta").textContent = `Test transaction #${s.test_row} · €${s.amount_eur.toFixed(2)} · hour of day ${s.hour}`;
    const stamp = $("#v-stamp");
    stamp.textContent = s.decision;
    stamp.className = `stamp ${declined ? "decline" : "approve"}`;
    $("#v-band").textContent = `Risk band · ${s.risk_level}`;
    $("#v-prob").textContent = prob(s.fraud_probability);

    const pos = `${Math.min(100, s.fraud_probability * 100)}%`;
    $("#v-fill").style.width = pos;
    $("#v-dot").style.left = pos;
    $("#v-mark").style.left = `${o.threshold * 100}%`;
    $("#v-mark-label").textContent = `threshold ${thr(o.threshold)}`;

    const actual = $("#v-actual");
    actual.replaceChildren("Actually: ", el("strong", { text: s.actual }));

    // SHAP bars — the notice cites only positive contributions, in order
    const reasons = declined ? s.shap.filter((d) => d.value > 0).slice(0, 3).map((d) => d.feature) : [];
    const max = Math.max(...s.shap.map((d) => Math.abs(d.value)));
    const rows = s.shap.map((d) => {
      const bar = el("i", { class: d.value >= 0 ? "pos" : "neg" });
      const label = el("span", { class: "shap-label", text: d.label });
      const n = reasons.indexOf(d.feature);
      if (n >= 0) label.append(el("span", { class: "tag", text: `reason ${n + 1}` }));
      const row = el("div", { class: "shap-row" }, [
        label, el("div", { class: "shap-bar" }, [bar]), el("span", { class: "shap-val", text: sign(d.value) }),
      ]);
      grow(bar, `${(Math.abs(d.value) / max) * 50}%`);
      return row;
    });
    $("#shap").replaceChildren(...rows);
    $("#shap-foot").textContent =
      `Model baseline ${sign(s.base_value)} · ${s.other_features.count} other features ${sign(s.other_features.value)}`;

    $("#notice-text").textContent = s.notice ??
      "No notice. The payment is approved, so the API returns no decline reason.";
  }

  function initScore(samples, o) {
    $$("#score .tab").forEach((b) => b.addEventListener("click", () => renderSample(b.dataset.sample, samples, o)));
    renderSample("fraud", samples, o);
  }

  // ── 02 threshold & cost ───────────────────────────────────
  // charts are drawn at their container's pixel width so label text stays 11px
  const widthOf = (node) => Math.max(300, Math.round(node.clientWidth));

  function costChart(t, split, index) {
    const W = widthOf($("#cost-chart")), H = W < 560 ? 220 : 280, L = 50, R = 12, T = 14, B = 30;
    const n = t.threshold.length;
    const cap = 8000;
    const x = (i) => L + (i / (n - 1)) * (W - L - R);
    const y = (c) => T + (1 - Math.min(c, cap) / cap) * (H - T - B);

    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Total cost against threshold for the ${split} set` });

    [0, 2000, 4000, 6000, 8000].forEach((c) => {
      root.append(svg("line", { x1: L, x2: W - R, y1: y(c), y2: y(c), stroke: "#cfc6d1", "stroke-width": 1 }));
      const label = svg("text", { x: L - 8, y: y(c) + 4, "text-anchor": "end" });
      label.textContent = c === 0 ? "£0" : `£${c / 1000}k`;
      root.append(label);
    });
    [0.0001, 0.001, 0.01, 0.1, 1].forEach((v) => {
      let i = t.threshold.findIndex((g) => g >= v * 0.999);
      if (i < 0) i = n - 1;
      const label = svg("text", { x: x(i), y: H - 10, "text-anchor": "middle" });
      label.textContent = String(v);
      root.append(label);
    });

    const path = (costs) => costs.map((c, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(c).toFixed(1)}`).join("");
    const other = split === "validate" ? "test" : "validate";
    root.append(svg("path", { d: path(t.splits[other].cost), fill: "none", stroke: "#d3c7d5", "stroke-width": 2.5 }));
    root.append(svg("path", { d: path(t.splits[split].cost), fill: "none", stroke: "#3b2142", "stroke-width": 2.5 }));

    const f1 = x(t.marks.f1_optimal);
    root.append(svg("line", { x1: f1, x2: f1, y1: T, y2: H - B, stroke: "#110d12", "stroke-width": 1.5, "stroke-dasharray": "2 4" }));
    const co = x(t.marks.cost_optimal);
    root.append(svg("line", { x1: co, x2: co, y1: T, y2: H - B, stroke: "#ff4f12", "stroke-width": 2.5 }));

    root.append(svg("circle", { cx: x(index), cy: y(t.splits[split].cost[index]), r: 6,
      fill: "#3b2142", stroke: "#f6f3f5", "stroke-width": 2.5 }));

    $("#cost-legend").textContent =
      `${split} set in aubergine · ${other} set in lilac · orange = cost-optimal (deployment) · dotted = F1-optimal`;
    return root;
  }

  function initThreshold(t, o) {
    const range = $("#thr");
    let split = "validate";
    const n = t.threshold.length;
    range.max = String(n - 1);
    range.value = String(t.marks.cost_optimal);

    $("#formula").textContent =
      `cost = missed fraud × ${gbp(o.cost_fn)} + false alarms × ${gbp(o.cost_fp)}`;

    // marks: the deployment (cost-optimal) threshold in orange, F1-optimal dotted ink
    const marks = $("#thr-marks");
    [["cost", t.marks.cost_optimal, `cost-optimal ${thr(o.threshold)}`],
     ["f1", t.marks.f1_optimal, `F1-optimal ${thr(o.f1_threshold)}`]].forEach(([cls, i, text]) => {
      const p = i / (n - 1);
      const m = el("div", { class: `smark ${cls}${p > 0.85 ? " edge-right" : ""}` }, [el("span", { class: "mono-label", text })]);
      m.style.left = `calc(var(--thumb) / 2 + ${p} * (100% - var(--thumb)))`;
      marks.append(m);
    });

    const update = () => {
      const i = Number(range.value);
      const d = t.splits[split];
      const caught = d.tp[i], missed = d.fn[i], fp = d.fp[i];
      const alerts = caught + fp;
      $("#thr-value").textContent = thr(t.threshold[i]);
      range.setAttribute("aria-valuetext", `threshold ${thr(t.threshold[i])}`);
      $("#r-caught").textContent = `${caught} of ${d.frauds}`;
      $("#r-caught-sub").textContent = `${pct(caught / d.frauds)} caught · ${missed} missed`;
      $("#r-fp").textContent = int(fp);
      $("#r-fp-sub").textContent = alerts ? `alert precision ${pct(caught / alerts)}` : "no alerts raised";
      $("#r-cost").textContent = gbp(d.cost[i]);
      $("#r-missed").textContent = `missed fraud ${gbp(missed * o.cost_fn)}`;
      $("#cost-chart").replaceChildren(costChart(t, split, i));
    };

    $$("#threshold .tab").forEach((b) => b.addEventListener("click", () => {
      split = b.dataset.split;
      $$("#threshold .tab").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      $("#thr-split-info").textContent =
        `${o.splits[split].fraud} frauds in ${int(o.splits[split].rows)} transactions`;
      update();
    }));
    $("#thr-split-info").textContent = `${o.splits.validate.fraud} frauds in ${int(o.splits.validate.rows)} transactions`;
    range.addEventListener("input", update);
    onResize(update);
    update();

    // frozen table
    const tbody = $("#frozen-table tbody");
    const rows = [];
    for (const split of ["validate", "test"]) {
      for (const [key, name] of [["cost_optimal", `Cost-optimal ${thr(o.threshold)}`], ["f1_optimal", `F1-optimal ${thr(o.f1_threshold)}`]]) {
        const m = o.frozen[split][key];
        rows.push(el("tr", {}, [
          el("td", { text: name }), el("td", { text: split[0].toUpperCase() + split.slice(1) }),
          el("td", { class: "num", text: `${pct(m.Recall)} (${m.TP} of ${m.TP + m.FN})` }),
          el("td", { class: "num", text: pct(m.Precision) }),
          el("td", { class: "num", text: gbp(m.Cost) }),
        ]));
      }
    }
    tbody.replaceChildren(...rows);

    // honest note — the README's finding, with every number from the data
    const v = o.frozen.validate, te = o.frozen.test;
    const points = ((v.cost_optimal.Recall - v.f1_optimal.Recall) * 100).toFixed(1);
    const saving = Math.round((1 - v.cost_optimal.Cost / v.f1_optimal.Cost) * 100);
    $("#honest-text").textContent =
      `On validate, the cost-optimal threshold caught ${points} points more fraud and cut cost by ${saving}%. ` +
      `On the later test window it did not hold up: it caught ${te.cost_optimal.TP - te.f1_optimal.TP} more fraud than the F1 threshold ` +
      `for ${te.cost_optimal.FP - te.f1_optimal.FP} more false alarms, and cost ${gbp(te.cost_optimal.Cost - te.f1_optimal.Cost)} more. ` +
      `With only ${o.splits.validate.fraud} validate frauds, the cost-optimal point is unstable over time. ` +
      `That is the case for recalibrating the threshold on a rolling window and monitoring alert precision alongside KS.`;
  }

  // ── 03 champion ───────────────────────────────────────────
  const MODELS = ["XGBoost", "Random Forest", "Logistic Regression"];
  const MODEL_COLOR = { "XGBoost": "var(--key)", "Random Forest": "var(--muted)", "Logistic Regression": "var(--soft)" };

  function initChampion(o) {
    const cols = [];
    for (const split of ["validate", "test"]) for (const k of ["KS", "AUPRC", "F1"]) cols.push([split, k]);
    const best = cols.map(([s, k]) => Math.max(...MODELS.map((m) => o.ranking[s][m][k])));

    $("#rank-table tbody").replaceChildren(...MODELS.map((m) => el("tr", { class: m === "XGBoost" ? "champ" : "" }, [
      el("td", { text: m === "XGBoost" ? `${m} · champion` : m }),
      ...cols.map(([s, k], j) => {
        const v = o.ranking[s][m][k];
        return el("td", { class: `num${v === best[j] ? " best" : ""}`, text: v.toFixed(3) });
      }),
    ])));

    const bars = (target, key, fmt) => {
      const max = Math.max(...MODELS.map((m) => o.at_threshold[m][key]));
      $(target).replaceChildren(...MODELS.map((m) => {
        const fill = el("i");
        fill.style.background = MODEL_COLOR[m];
        grow(fill, `${(o.at_threshold[m][key] / max) * 100}%`);
        return el("div", { class: "hbar" }, [
          el("span", { text: m }), el("div", { class: "hbar-track" }, [fill]),
          el("span", { class: "hbar-val", text: fmt(o.at_threshold[m][key]) }),
        ]);
      }));
    };
    bars("#bars-precision", "precision", (x) => pct(x));
    bars("#bars-flag", "flag_rate", (x) => pct(x, 2));

    $("#rationale").replaceChildren(...o.rationale.map((r) => el("li", { text: r })));
  }

  // ── 04 drift ──────────────────────────────────────────────
  function initDrift(o) {
    const psi = Object.entries(o.drift.psi).sort((a, b) => b[1] - a[1]);
    const max = Math.max(0.3, ...psi.map(([, v]) => v)) * 1.08;
    const box = $("#psi");
    const rows = psi.map(([f, v]) => {
      const fill = el("i");
      grow(fill, `${(v / max) * 100}%`);
      const track = el("div", { class: "psi-track" }, [fill]);
      if (v > 0.25) {
        const flag = el("span", { class: "psi-flag", title: "significant shift" });
        flag.style.left = `${(v / max) * 100}%`;
        track.append(flag);
      }
      return el("div", { class: "psi-row" }, [el("span", { text: f }), track, el("span", { class: "psi-val", text: v.toFixed(3) })]);
    });
    box.replaceChildren(...rows);
    // dashed band lines, aligned to the track column
    [0.10, 0.25].forEach((b) => {
      const line = el("div", { class: "psi-line" });
      line.style.left = `calc(54px + (100% - 116px) * ${b / max})`;
      box.append(line);
    });

    const d = o.drift;
    $("#trigger-rule").textContent = `Retrain when test KS falls below ${d.ks_trigger.toFixed(2)}`;
    $("#trigger-now").textContent = d.ks_test.toFixed(3);
    $("#trigger-sub").textContent = `Current test KS · ${d.ks_test >= d.ks_trigger ? "above" : "below"} the trigger (validate ${d.ks_validate.toFixed(3)})`;
    const scale = (k) => `${((k - 0.5) / 0.5) * 100}%`;   // track shows KS 0.5 → 1.0
    $("#ks-mark").style.left = scale(d.ks_trigger);
    $("#ks-dot").style.left = scale(d.ks_test);

    $("#ks-scale").replaceChildren(
      el("span", { text: "0.50" }), el("span", { text: `trigger ${d.ks_trigger.toFixed(2)}` }), el("span", { text: "1.00" }));

    onResize(() => hourlyChart(o));
    hourlyChart(o);
  }

  function hourlyChart(o) {
    const h = o.drift.hourly;
    const W = widthOf($("#hourly-chart")), H = W < 560 ? 200 : 250, L = 44, R = 12, T = 22, B = 30;
    const n = h.hour.length;
    const ymax = Math.max(...h.fraud_rate) * 100 * 1.1;
    const x = (hr) => L + (hr / n) * (W - L - R);
    const y = (r) => T + (1 - (r * 100) / ymax) * (H - T - B);
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Hourly fraud rate over 48 hours" });

    const bands = [
      ["Train", 0, o.split_hours.validate_start, 0.0],
      ["Validate", o.split_hours.validate_start, o.split_hours.test_start, 0.35],
      ["Test", o.split_hours.test_start, n, 0.7],
    ];
    bands.forEach(([name, a, b, op]) => {
      if (op) root.append(svg("rect", { x: x(a), y: T, width: x(b) - x(a), height: H - T - B, fill: "#d3c7d5", "fill-opacity": op }));
      // Validate and Test bands are narrow, so their labels hang right-aligned
      const label = name === "Train"
        ? svg("text", { x: x(a) + 6, y: T - 8 })
        : svg("text", { x: x(b) - 4, y: T - 8, "text-anchor": "end" });
      label.textContent = name;
      root.append(label);
    });
    for (let tick = 0; tick <= n; tick += W < 560 ? 12 : 6) {
      const t = svg("text", { x: x(tick), y: H - 10, "text-anchor": "middle" });
      t.textContent = `${tick}h`;
      root.append(t);
    }
    const yTicks = [0, Math.round(ymax / 2 * 10) / 10, Math.floor(ymax * 10) / 10];
    yTicks.forEach((v) => {
      root.append(svg("line", { x1: L, x2: W - R, y1: y(v / 100), y2: y(v / 100), stroke: "#cfc6d1" }));
      const t = svg("text", { x: L - 6, y: y(v / 100) + 4, "text-anchor": "end" });
      t.textContent = `${v}%`;
      root.append(t);
    });
    const pts = h.fraud_rate.map((r, i) => `${x(h.hour[i] + 0.5).toFixed(1)},${y(r).toFixed(1)}`);
    root.append(svg("polyline", { points: pts.join(" "), fill: "none", stroke: "#3b2142", "stroke-width": 2.5, "stroke-linejoin": "round" }));
    h.fraud_rate.forEach((r, i) => {
      const c = svg("circle", { cx: x(h.hour[i] + 0.5), cy: y(r), r: 2.6, fill: "#3b2142" });
      const tip = svg("title");
      tip.textContent = `Hour ${h.hour[i]}: ${pct(r, 2)} (${h.fraud[i]} of ${int(h.count[i])})`;
      c.append(tip);
      root.append(c);
    });
    $("#hourly-chart").replaceChildren(root);
  }

  // ── footer scope line ─────────────────────────────────────
  function renderScope(o) {
    const rows = o.splits.train.rows + o.splits.validate.rows + o.splits.test.rows;
    const frauds = o.splits.train.fraud + o.splits.validate.fraud + o.splits.test.fraud;
    const hours = o.drift.hourly.hour.length;
    $("#scope").textContent =
      `Built on the public anonymised credit-card dataset: ${hours} hours, ${int(frauds)} frauds in ${int(rows)} transactions. ` +
      `V1–V28 are PCA components whose real-world meaning is unknown. This is a portfolio project, not a production bank system.`;
  }

  // ── figures inside the judgement notes ────────────────────
  function renderJudgement(o) {
    const frauds = o.splits.test.fraud;
    bind("cost-pair", `(${gbp(o.cost_fn)} per missed fraud, ${gbp(o.cost_fp)} per false alarm)`);
    bind("test-frauds", String(frauds));
    bind("one-case", (100 / frauds).toFixed(1));
    bind("prec-xgb", `${pct(o.at_threshold["XGBoost"].precision)} for XGBoost`);
    bind("prec-rf", `${pct(o.at_threshold["Random Forest"].precision)} for Random Forest`);
    bind("ks-now", o.drift.ks_test.toFixed(3));
    bind("ks-trig", o.drift.ks_trigger.toFixed(2));
    const [top, val] = Object.entries(o.drift.psi).sort((a, b) => b[1] - a[1])[0];
    bind("psi-top", top);
    bind("psi-top-val", val.toFixed(2));
    bind("window-hours", String(o.drift.hourly.hour.length));
  }

  // ── boot ──────────────────────────────────────────────────
  async function load(name) {
    const res = await fetch(`data/${name}.json`);
    if (!res.ok) throw new Error(`data/${name}.json: ${res.status}`);
    return res.json();
  }

  Promise.all([load("overview"), load("samples"), load("threshold")])
    .then(([o, samples, t]) => {
      renderHero(o);
      initScore(samples, o);
      initThreshold(t, o);
      initChampion(o);
      initDrift(o);
      renderScope(o);
      renderJudgement(o);
      revealOnScroll();
    })
    .catch((err) => {
      console.error(err);
      revealOnScroll();
      $("#score .section-lede").textContent =
        "The demo data could not be loaded. If you opened this file directly, serve the site folder over HTTP instead.";
    });
})();
