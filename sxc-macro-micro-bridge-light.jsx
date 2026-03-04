import React, { useState, useMemo } from "react";
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Area,
  Scatter,
} from "recharts";

// ══════════════════════════════════════════════════════════════════
// MARGIN MODEL: Regressions & Two-Tier GP
// ══════════════════════════════════════════════════════════════════

// Revenue/ton = f(Coal) — R²=0.89
const REV_A = 101.75, REV_B = 1.4335;
// GP/ton blended = f(Coal) — R²=0.56
const GP_BL_A = 58.68, GP_BL_B = 0.1185;
// Inverse demand: Coal = f(Demand)
const INV_A = 446.95, INV_B = -16.56;
// Two-tier GP
const CONTRACT_GP = 95; // $/ton contracted
const SPOT_GP_A = 17, SPOT_GP_B = 0.18; // Spot GP/ton = $17 + $0.18×Coal → $58 at $226

// Coal price scenario multipliers on naive inverse demand
const COAL_MULT = { high: 1.1, base: 0.85, low: 0.7 };

function coalForecast(demand, scen) {
  return Math.max(100, (INV_A + INV_B * demand) * (COAL_MULT[scen] || 0.85));
}

// Historical financial data for regression scatter plots
const finHist = [
  { year: 2015, coal: 130, revTon: 302, gpTon: 64 },
  { year: 2016, coal: 150, revTon: 277, gpTon: 80 },
  { year: 2017, coal: 165, revTon: 310, gpTon: 81 },
  { year: 2018, coal: 173, revTon: 324, gpTon: 81 },
  { year: 2019, coal: 156, revTon: 357, gpTon: 77 },
  { year: 2020, coal: 144, revTon: 334, gpTon: 75 },
  { year: 2021, coal: 154, revTon: 324, gpTon: 81 },
  { year: 2022, coal: 239, revTon: 461, gpTon: 91 },
  { year: 2024, coal: 260, revTon: 451, gpTon: 82 },
  { year: 2025, coal: 226, revTon: 440, gpTon: 87 },
];

// ══════════════════════════════════════════════════════════════════
// MACRO MODEL (from regression analysis)
// ══════════════════════════════════════════════════════════════════

const hist = [
  { year: 2015, yi: 0, cd: 18.0, clair: 4.30, clf: 3.20, sxc: 4.122 },
  { year: 2016, yi: 1, cd: 17.4, clair: 4.30, clf: 3.20, sxc: 3.956 },
  { year: 2017, yi: 2, cd: 18.0, clair: 4.30, clf: 3.20, sxc: 3.851 },
  { year: 2018, yi: 3, cd: 18.0, clair: 4.30, clf: 3.20, sxc: 4.033 },
  { year: 2019, yi: 4, cd: 17.4, clair: 4.15, clf: 3.20, sxc: 4.171 },
  { year: 2020, yi: 5, cd: 13.8, clair: 3.75, clf: 2.60, sxc: 3.789 },
  { year: 2021, yi: 6, cd: 16.2, clair: 4.15, clf: 2.60, sxc: 4.183 },
  { year: 2022, yi: 7, cd: 14.4, clair: 4.30, clf: 2.60, sxc: 4.031 },
  { year: 2023, yi: 8, cd: 13.8, clair: 4.30, clf: 2.60, sxc: 4.046 },
  { year: 2024, yi: 9, cd: 13.2, clair: 3.95, clf: 2.60, sxc: 4.028 },
  { year: 2025, yi: 10, cd: 12.6, clair: 3.60, clf: 2.60, sxc: 3.668 },
];

const CEIL = 14.4;
const LOG_L = 100;
const LAMBDA = 0.85;
const fcYrs = [2026, 2027, 2028, 2029, 2030, 2031, 2032];

const enriched = hist.map(d => {
  const cap = d.clair + d.clf, def = d.cd - cap;
  return { ...d, cap, def: Math.round(def * 100) / 100, share: Math.round((d.sxc / def) * 1000) / 10 };
});

// Logistic share model (L=100 fixed)
function fitLog(data) {
  const pts = data.map(d => ({ x: d.yi, y: d.share }));
  const L = LOG_L;
  const f = (x, k, t) => L / (1 + Math.exp(-k * (x - t)));
  const sse = (k, t) => pts.reduce((s, p) => s + (p.y - f(p.x, k, t)) ** 2, 0);
  let b = { e: Infinity, k: 0.3, t: 3 };
  for (let k = 0.05; k <= 1.5; k += 0.02)
    for (let t = -5; t <= 12; t += 0.25) { const e = sse(k, t); if (e < b.e) b = { e, k, t }; }
  for (let r = 0; r < 5; r++) {
    const sK = 0.01 / (r + 1), sT = 0.1 / (r + 1); let go = true;
    while (go) { go = false; for (const dK of [-sK, 0, sK]) for (const dT of [-sT, 0, sT]) { const nK = b.k + dK, nT = b.t + dT; if (nK < 0.01 || nK > 3) continue; const e = sse(nK, nT); if (e < b.e - 0.00001) { b = { e, k: nK, t: nT }; go = true; } } }
  }
  const ym = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  const sst = pts.reduce((s, p) => s + (p.y - ym) ** 2, 0);
  return { L, k: Math.round(b.k * 1000) / 1000, t0: Math.round(b.t * 10) / 10, r2: Math.round((1 - b.e / sst) * 10000) / 10000, predict: (x) => L / (1 + Math.exp(-b.k * (x - b.t))) };
}
const logF = fitLog(enriched);

// Linear share regression for comparison
function linReg(data, xk, yk) {
  const n = data.length, sx = data.reduce((s, d) => s + d[xk], 0), sy = data.reduce((s, d) => s + d[yk], 0);
  const sxy = data.reduce((s, d) => s + d[xk] * d[yk], 0), sxx = data.reduce((s, d) => s + d[xk] * d[xk], 0);
  const sl = (n * sxy - sx * sy) / (n * sxx - sx * sx), int = (sy - sl * sx) / n;
  const ym = sy / n, sst = data.reduce((s, d) => s + (d[yk] - ym) ** 2, 0);
  const ssr = data.reduce((s, d) => s + (d[yk] - (sl * d[xk] + int)) ** 2, 0);
  return { slope: sl, intercept: int, r2: 1 - ssr / sst };
}
const regSh = linReg(enriched, "yi", "share");

// Deficit-based share regression: Share = a + b × Deficit
const regShDef = linReg(enriched, "def", "share");

// EWMA time series (λ=0.85)
function buildTS() {
  const cleanDiffs = [
    { year: 2016, d: -0.6, age: 9 }, { year: 2017, d: 0.6, age: 8 },
    { year: 2018, d: 0.0, age: 7 }, { year: 2019, d: -0.6, age: 6 },
    { year: 2020, d: -1.0, age: 5 }, { year: 2021, d: -1.0, age: 4 },
    { year: 2022, d: -1.0, age: 3 }, { year: 2023, d: -0.6, age: 2 },
    { year: 2024, d: -0.6, age: 1 }, { year: 2025, d: -0.6, age: 0 },
  ];
  const wts = cleanDiffs.map(d => ({ ...d, w: Math.pow(LAMBDA, d.age) }));
  const sumW = wts.reduce((s, d) => s + d.w, 0);
  const normed = wts.map(d => ({ ...d, nw: d.w / sumW }));
  const mu = normed.reduce((s, d) => s + d.nw * d.d, 0);
  const res = normed.map(d => ({ ...d, r: d.d - mu }));
  const pos = res.filter(d => d.r >= 0), neg = res.filter(d => d.r < 0);
  const swP = pos.reduce((s, d) => s + d.nw, 0), swN = neg.reduce((s, d) => s + d.nw, 0);
  const sp = swP > 0 ? Math.sqrt(pos.reduce((s, d) => s + d.nw * d.r * d.r, 0) / swP) : 0.3;
  const sn = swN > 0 ? Math.sqrt(neg.reduce((s, d) => s + d.nw * d.r * d.r, 0) / swN) : 0.4;
  return { mu, sp, sn };
}
const ts = buildTS();

// Demand fan
function demandAtPercentile(n, zUp, zDown) {
  const last = 12.6, sq = Math.sqrt(n), c = last + ts.mu * n;
  return { c, up: Math.min(CEIL, c + zUp * ts.sp * sq), down: Math.max(0, c - zDown * ts.sn * sq) };
}

function getCap(yr) {
  return (yr <= 2027 ? 3.5 : yr <= 2029 ? 3.3 : yr <= 2031 ? 3.1 : 3.0) + (yr <= 2029 ? 2.6 : 2.4);
}

// ══════════════════════════════════════════════════════════════════
// MICRO MODEL (contract-level scenarios)
// Each scenario maps to a demand regime (z-score from fan)
// Contracted volumes are fixed inputs per scenario
// Other/Spot/Export = macro implied total at that z − contracted (the plug)
// ══════════════════════════════════════════════════════════════════

// Each scenario's demand z-score (how far from central)
const scenarioZ = {
  mgmt: 1.2,    // demand holds up, above central
  base: 0.3,    // near central, slightly above
  uss_nr: -0.3, // moderately below central
  nr_clf: -0.8, // well below central
  down: -1.4,   // deep below central
};

function macroImpliedSalesAtZ(yr, i, z) {
  const n = i + 1, yi = 11 + i;
  const sq = Math.sqrt(n);
  const cDem = 12.6 + ts.mu * n;
  const dem = z >= 0
    ? Math.min(CEIL, cDem + z * ts.sp * sq)
    : Math.max(0, cDem + z * ts.sn * sq);
  const cap = getCap(yr);
  const deficit = Math.max(0, dem - cap);
  // Deficit-based share: Share = regShDef.intercept + regShDef.slope × deficit, clamped [30,95]
  const sh = Math.max(0.30, Math.min(0.95, (regShDef.slope * deficit + regShDef.intercept) / 100));
  return { sales: Math.round(sh * deficit * 1000), demand: dem, deficit, share: sh * 100 };
}

function buildContractForecast(scenario) {
  const z = scenarioZ[scenario];
  return fcYrs.map((yr, i) => {
    let ih, hav, jew, mid, gc;

    if (scenario === "mgmt") {
      ih = 1220;
      hav = 300;
      jew = 200;
      mid = 550;
      gc = 590;
    } else if (scenario === "base") {
      ih = 1220;
      hav = yr <= 2028 ? 300 : 250;
      jew = yr <= 2028 ? 200 : 160;
      mid = 550;
      gc = yr <= 2026 ? 590 : yr <= 2028 ? 450 : yr <= 2030 ? 350 : 300;
    } else if (scenario === "uss_nr") {
      ih = 1220;
      hav = yr <= 2028 ? 300 : 250;
      jew = yr <= 2028 ? 200 : 160;
      mid = 550;
      gc = yr <= 2026 ? 590 : 0;
    } else if (scenario === "nr_clf") {
      ih = yr <= 2028 ? 1220 : yr <= 2030 ? 1150 : 1100;
      hav = yr <= 2028 ? 300 : yr <= 2030 ? 250 : 200;
      jew = yr <= 2028 ? 200 : yr <= 2030 ? 160 : 130;
      mid = yr <= 2029 ? 550 : yr <= 2031 ? 500 : 450;
      gc = yr <= 2026 ? 590 : 0;
    } else { // down
      ih = yr <= 2028 ? 1220 : yr <= 2030 ? 1100 : 1000;
      hav = yr <= 2028 ? 300 : yr <= 2030 ? 200 : 150;
      jew = yr <= 2028 ? 200 : yr <= 2030 ? 130 : 100;
      mid = yr <= 2029 ? 550 : yr <= 2031 ? 450 : 350;
      gc = yr <= 2026 ? 590 : 0;
    }

    const contracted = ih + hav + jew + mid + gc;
    const macro = macroImpliedSalesAtZ(yr, i, z);

    // Other = plug: macro implied total − contracted, floor at 0
    // If contracted > implied total, oth = 0 and total = contracted (contracts honored)
    const oth = Math.max(0, macro.sales - contracted);
    const total = contracted + oth;

    return {
      year: yr.toString(), total, contracted, ih, hav, jew, mid, gc, oth,
      impliedTotal: macro.sales, impliedDemand: macro.demand, z,
    };
  });
}

const scenarios = [
  { id: "mgmt", label: "Management", short: "Mgmt", color: "#4ade80" },
  { id: "base", label: "Base Case", short: "Base", color: "#60a5fa" },
  { id: "uss_nr", label: "USS Non-Renewal", short: "USS NR", color: "#f59e0b" },
  { id: "nr_clf", label: "NR + CLF Distress", short: "NR+CLF", color: "#f97316" },
  { id: "down", label: "Down Case", short: "Down", color: "#f87171" },
];

// ══════════════════════════════════════════════════════════════════
// BRIDGE: back-solve implied demand for each scenario
// ══════════════════════════════════════════════════════════════════

// normCDF still used for reference percentiles
function normCDF(z) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
  const p = 0.3275911;
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.sqrt(2);
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1.0 + sign * y);
}

// ══════════════════════════════════════════════════════════════════
// COMPONENT
// ══════════════════════════════════════════════════════════════════

const barConfigs = [
  { key: "ih", name: "Indiana Harbor", color: "#3b82f6" },
  { key: "hav", name: "Haverhill II", color: "#6366f1" },
  { key: "jew", name: "Jewell", color: "#8b5cf6" },
  { key: "mid", name: "Middletown", color: "#a855f7" },
  { key: "gc", name: "Granite City", color: "#22d3ee" },
  { key: "oth", name: "Other / Spot / Export", color: "#f59e0b" },
];

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const skip = ["90% Lower", "68% Lower", "range"];
  const fmt = (name, val) => {
    if (typeof val !== "number") return val;
    const n = name.toLowerCase();
    // COGS % or any percentage
    if (n.includes("cogs") || n.includes("%")) return `${val.toFixed(1)}%`;
    // Revenue or GP in $M (already in millions)
    if (n.includes("rev") || n.includes("gp")) return `$${val.toFixed(1)}M`;
    // Volume data in K tons → show as M
    if (Math.abs(val) >= 100) return `${(val / 1000).toFixed(2)}M`;
    return `${val.toFixed(2)}M`;
  };
  return (
    <div style={{ background: "rgba(255,255,255,0.98)", border: "1px solid #e5e7eb", borderRadius: 6, padding: "10px 14px", fontSize: 11, fontFamily: "'DM Sans',sans-serif", color: "#374151", boxShadow: "0 4px 20px rgba(0,0,0,0.12)", maxWidth: 300 }}>
      <div style={{ fontWeight: 700, color: "#111827", marginBottom: 6, fontSize: 12 }}>{label}</div>
      {payload.filter(p => p.value != null && !skip.includes(p.name) && p.value !== 0).map((p, i) => (
        <div key={i} style={{ display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 2 }}>
          <span style={{ color: p.color || p.stroke || p.fill, fontSize: 10 }}>{p.name}</span>
          <span style={{ color: "#111827", fontWeight: 600 }}>{fmt(p.name, p.value)}</span>
        </div>
      ))}
    </div>
  );
};

export default function MacroMicroBridge() {
  const [view, setView] = useState("bridge");
  const [selScenario, setSelScenario] = useState("base");
  const [coalScen, setCoalScen] = useState("base");
  const [shareMode, setShareMode] = useState("time"); // "time" or "deficit"
  const mn = "'JetBrains Mono',monospace", dm = "'DM Sans',sans-serif";

  // All scenario forecasts
  const allFC = useMemo(() => {
    const r = {};
    scenarios.forEach(s => { r[s.id] = buildContractForecast(s.id); });
    return r;
  }, []);

  // Bridge data: use pre-assigned z-scores and scenario implied demands
  const bridgeData = useMemo(() => {
    return fcYrs.map((yr, i) => {
      const n = i + 1;
      const d = demandAtPercentile(n, 1.645, 1.645);
      const d1 = demandAtPercentile(n, 1, 1);

      const row = {
        year: yr.toString(), central: d.c, up1: d1.up, up2: d.up, down1: d1.down, down2: d.down,
      };

      scenarios.forEach(s => {
        const fc = allFC[s.id][i];
        row[`${s.id}_demand`] = Math.round(fc.impliedDemand * 100) / 100;
        row[`${s.id}_z`] = scenarioZ[s.id];
        row[`${s.id}_pct`] = Math.round(normCDF(scenarioZ[s.id]) * 1000) / 10;
      });

      return row;
    });
  }, [allFC]);

  // Demand fan chart with scenario implied demands overlaid
  const demandOverlay = useMemo(() => {
    const lastH = hist[hist.length - 1];
    const base = hist.map(d => ({ year: d.year.toString(), actual: d.cd }));
    // 2025 bridge point
    const bridge = {
      year: "2025", actual: lastH.cd,
      central: lastH.cd, up1: lastH.cd, up2: lastH.cd, down1: lastH.cd, down2: lastH.cd,
      band2_base: lastH.cd, band2_top: 0, band1_base: lastH.cd, band1_top: 0,
    };
    scenarios.forEach(s => { bridge[`${s.id}_demand`] = lastH.cd; });

    const fc = bridgeData.map(d => {
      const row = {
        year: d.year, central: d.central, up1: d.up1, up2: d.up2, down1: d.down1, down2: d.down2,
        band2_base: d.down2, band2_top: d.up2 - d.down2,
        band1_base: d.down1, band1_top: d.up1 - d.down1,
      };
      scenarios.forEach(s => { row[`${s.id}_demand`] = d[`${s.id}_demand`]; });
      return row;
    });

    return [...base, bridge, ...fc];
  }, [bridgeData]);

  // Probability table: PDF-based normalization (asymmetric)
  const probTable = useMemo(() => {
    const lastBridge = bridgeData[bridgeData.length - 1];
    if (!lastBridge) return [];

    const normPDF = (z) => (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-z * z / 2);

    const items = scenarios.map(s => {
      const z = scenarioZ[s.id];
      const pdf = normPDF(z);
      return {
        ...s,
        demand: lastBridge[`${s.id}_demand`],
        z,
        pdf,
        sales: allFC[s.id][allFC[s.id].length - 1].total,
        contracted: allFC[s.id][allFC[s.id].length - 1].contracted,
        oth: allFC[s.id][allFC[s.id].length - 1].oth,
      };
    });

    const sumPDF = items.reduce((s, d) => s + d.pdf, 0);
    items.forEach(d => { d.prob = Math.round(d.pdf / sumPDF * 100); });

    // Ensure sums to 100
    const diff = 100 - items.reduce((s, d) => s + d.prob, 0);
    if (items.length > 0) items[Math.floor(items.length / 2)].prob += diff;

    return items.sort((a, b) => b.demand - a.demand);
  }, [bridgeData, allFC]);

  const expectedSales = probTable.reduce((s, d) => s + ((d.prob || 0) / 100) * (d.sales || 0), 0);

  // Implied SXC Sales fan (from macro model: share × (demand - captive))
  const salesFan = useMemo(() => {
    const histSales = enriched.map(d => ({ year: d.year.toString(), actual: d.sxc * 1000 }));
    const lastVal = enriched[enriched.length - 1].sxc * 1000;
    // Bridge at 2025
    const bridge = { year: "2025", actual: lastVal, central: lastVal, up1: lastVal, up2: lastVal, down1: lastVal, down2: lastVal, band2_base: lastVal, band2_top: 0, band1_base: lastVal, band1_top: 0 };
    scenarios.forEach(s => { bridge[`${s.id}_sales`] = lastVal; });

    const fc = fcYrs.map((yr, i) => {
      const n = i + 1, yi = 11 + i, cap = getCap(yr);
      const d = demandAtPercentile(n, 1.645, 1.645);
      const d1 = demandAtPercentile(n, 1, 1);
      // Deficit-based share: share depends on deficit at each demand level
      const calc = (dem) => {
        const deficit = Math.max(0, dem - cap);
        const sh = Math.max(0.30, Math.min(0.95, (regShDef.slope * deficit + regShDef.intercept) / 100));
        return Math.round(sh * deficit * 1000);
      };
      const up2v = calc(d.up), down2v = calc(d.down), up1v = calc(d1.up), down1v = calc(d1.down);
      const row = {
        year: yr.toString(),
        central: calc(d.c), up1: up1v, up2: up2v, down1: down1v, down2: down2v,
        band2_base: down2v, band2_top: up2v - down2v,
        band1_base: down1v, band1_top: up1v - down1v,
      };
      scenarios.forEach(s => { row[`${s.id}_sales`] = allFC[s.id][i].total; });
      return row;
    });

    return [...histSales, bridge, ...fc];
  }, [allFC]);

  // Contract waterfall for selected scenario
  const selFC = useMemo(() => {
    const actuals = [
      { year: "2023", total: 4032, ih: 1220, hav: 550, jew: 200, mid: 550, gc: 590, oth: 922 },
      { year: "2024", total: 4032, ih: 1220, hav: 550, jew: 200, mid: 550, gc: 590, oth: 922 },
      { year: "2025", total: 3668, ih: 1220, hav: 300, jew: 200, mid: 550, gc: 590, oth: 808 },
    ];
    return [...actuals, ...allFC[selScenario]];
  }, [selScenario, allFC]);

  // Revenue & GP fan: for each scenario, compute rev and GP per year
  const revGpFan = useMemo(() => {
    // Historical actuals (2025 is the bridge point, handled separately)
    const histRevGP = [
      { year: "2015", actualRev: 1243.6, actualGP: 264.3 },
      { year: "2016", actualRev: 1097.2, actualGP: 316.8 },
      { year: "2017", actualRev: 1195.0, actualGP: 311.4 },
      { year: "2018", actualRev: 1308.3, actualGP: 326.4 },
      { year: "2019", actualRev: 1489.1, actualGP: 322.7 },
      { year: "2020", actualRev: 1265.4, actualGP: 284.8 },
      { year: "2021", actualRev: 1354.5, actualGP: 337.2 },
      { year: "2022", actualRev: 1856.9, actualGP: 367.6 },
      { year: "2023", actualRev: 1954.0, actualGP: 338.6 },
      { year: "2024", actualRev: 1817.3, actualGP: 332.0 },
    ];
    // Add actual COGS% to historicals
    histRevGP.forEach(d => { d.actualCogsPct = Math.round((d.actualRev - d.actualGP) / d.actualRev * 1000) / 10; });

    // 2025 bridge point: last actual + starting point for all scenario lines
    const bridge = { year: "2025", actualRev: 1613.8, actualGP: 318.7, actualCogsPct: Math.round((1613.8 - 318.7) / 1613.8 * 1000) / 10 };
    scenarios.forEach(s => {
      bridge[`${s.id}_rev`] = 1613.8;
      bridge[`${s.id}_gp`] = 318.7;
      bridge[`${s.id}_cogsPct`] = bridge.actualCogsPct;
    });

    const fc = fcYrs.map((yr, i) => {
      const row = { year: yr.toString() };
      scenarios.forEach(s => {
        const f = allFC[s.id][i];
        const dem = f.impliedDemand;
        const coal = coalForecast(dem, coalScen);
        const revTon = REV_A + REV_B * coal;
        const spotGP = SPOT_GP_A + SPOT_GP_B * coal;
        const revenue = f.total * revTon / 1000;
        const gp = (f.contracted * CONTRACT_GP + f.oth * spotGP) / 1000;
        const cogsPct = revenue > 0 ? Math.round((revenue - gp) / revenue * 1000) / 10 : 0;
        row[`${s.id}_rev`] = Math.round(revenue * 10) / 10;
        row[`${s.id}_gp`] = Math.round(gp * 10) / 10;
        row[`${s.id}_cogsPct`] = cogsPct;
        row[`${s.id}_coal`] = Math.round(coal);
        row[`${s.id}_revTon`] = Math.round(revTon);
        row[`${s.id}_spotGP`] = Math.round(spotGP);
        row[`${s.id}_blendGP`] = f.total > 0 ? Math.round(gp / f.total * 1000 * 10) / 10 : 0;
        row[`${s.id}_cntPct`] = f.total > 0 ? Math.round(f.contracted / f.total * 100) : 0;
      });
      return row;
    });
    return [...histRevGP, bridge, ...fc];
  }, [allFC, coalScen]);

  // GP waterfall for selected scenario (stacked contract vs spot)
  const gpWaterfall = useMemo(() => {
    const actuals = [
      { year: "2023", contractGP: 271.7, spotGP: 46.9, totalGP: 318.6 },
      { year: "2024", contractGP: 271.7, spotGP: 60.3, totalGP: 332.0 },
      { year: "2025", contractGP: 271.7, spotGP: 47.0, totalGP: 318.7 },
    ];
    const fc = allFC[selScenario].map((f, i) => {
      const coal = coalForecast(f.impliedDemand, coalScen);
      const spotGPt = SPOT_GP_A + SPOT_GP_B * coal;
      const cGP = f.contracted * CONTRACT_GP / 1000;
      const sGP = f.oth * spotGPt / 1000;
      return {
        year: f.year, contractGP: Math.round(cGP * 10) / 10,
        spotGP: Math.round(sGP * 10) / 10, totalGP: Math.round((cGP + sGP) * 10) / 10,
      };
    });
    return [...actuals, ...fc];
  }, [selScenario, allFC, coalScen]);

  // Share regression chart data (historical + forecast)
  const shareChart = useMemo(() => {
    const histPoints = enriched.map(d => ({
      year: d.year.toString(), actual: d.share,
      projected: d.year === 2025 ? d.share : undefined,
      lin: Math.round((regSh.slope * d.yi + regSh.intercept) * 10) / 10,
      log: Math.round(logF.predict(d.yi) * 10) / 10,
    }));
    const fcPoints = fcYrs.map((yr, i) => {
      const yi = 11 + i;
      return {
        year: yr.toString(),
        projected: Math.round(logF.predict(yi) * 10) / 10,
        lin: Math.round((regSh.slope * yi + regSh.intercept) * 10) / 10,
        log: Math.round(logF.predict(yi) * 10) / 10,
      };
    });
    return [...histPoints, ...fcPoints];
  }, []);

  // Deficit-based share: scatter (deficit vs share) + scenario projections
  const defShareScatter = useMemo(() => {
    return enriched.map(d => ({
      def: d.def, share: d.share, year: d.year, sxc: d.sxc,
      fit: Math.round((regShDef.slope * d.def + regShDef.intercept) * 10) / 10,
    }));
  }, []);

  // For each scenario, compute deficit-based implied sales vs time-based
  const deficitComparison = useMemo(() => {
    return scenarios.map(s => {
      const z = scenarioZ[s.id];
      const rows = fcYrs.map((yr, i) => {
        const n = i + 1, yi = 11 + i;
        const dem = demandAtPercentile(n, Math.abs(z), Math.abs(z));
        const d = z >= 0 ? Math.min(CEIL, 12.6 + ts.mu * n + z * ts.sp * Math.sqrt(n)) : Math.max(0, 12.6 + ts.mu * n + z * ts.sn * Math.sqrt(n));
        const cap = getCap(yr);
        const deficit = Math.max(0, d - cap);
        const timeSh = logF.predict(yi);
        const defSh = Math.max(30, Math.min(95, regShDef.slope * deficit + regShDef.intercept));
        const timeSXC = Math.round(timeSh / 100 * deficit * 1000);
        const defSXC = Math.round(defSh / 100 * deficit * 1000);
        return { yr, dem: d, cap, deficit, timeSh, defSh, timeSXC, defSXC, delta: defSXC - timeSXC };
      });
      return { ...s, z, rows };
    });
  }, []);

  const btn = (id, cur, set, col, label) => (
    <button onClick={() => set(id)} style={{
      padding: "6px 12px", borderRadius: 6, fontSize: 10, fontWeight: 600, cursor: "pointer", fontFamily: dm,
      border: cur === id ? `1px solid ${col}50` : "1px solid #d1d5db",
      background: cur === id ? `${col}15` : "#f3f4f6",
      color: cur === id ? col : "#6b7280",
    }}>{label}</button>
  );

  return (
    <div style={{ minHeight: "100vh", background: "#ffffff", color: "#1f2937", fontFamily: dm, padding: "28px 24px" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 1600, margin: "0 auto" }}>

        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, fontFamily: mn, letterSpacing: 2.5, textTransform: "uppercase", color: "#6b7280", marginBottom: 8 }}>
            SunCoke Energy (SXC) · Macro-to-Micro Bridge · Probability-Weighted Scenarios
          </div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: "#111827", margin: 0 }}>Demand Distribution → Contract Scenarios → Expected Sales</h1>
          <p style={{ fontSize: 13, color: "#6b7280", margin: "6px 0 0" }}>
            Statistical demand fan (EWMA λ={LAMBDA}) · Deficit-based share (R²={regShDef?.r2?.toFixed(3) ?? "?"}) · Back-solved implied demand per scenario · Probability from fan CDF
          </p>
        </div>

        {/* Probability-weighted KPIs */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: 8, marginBottom: 20 }}>
          {probTable.map(s => (
            <div key={s.id} style={{ background: `${s.color}15`, border: `1px solid ${s.color}30`, borderRadius: 8, padding: "10px 10px" }}>
              <div style={{ fontSize: 8, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1, fontFamily: mn }}>{s.short}</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: s.color, fontFamily: mn }}>{s.prob}%</div>
              <div style={{ fontSize: 9, color: "#6b7280" }}>{((s.sales || 0) / 1000).toFixed(2)}M · z={(s.z ?? 0).toFixed(1)}</div>
            </div>
          ))}
          <div style={{ background: "#f3f4f6", border: "1px solid #d1d5db", borderRadius: 8, padding: "10px 10px" }}>
            <div style={{ fontSize: 8, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1, fontFamily: mn }}>E[Sales] '32</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: "#111827", fontFamily: mn }}>{(expectedSales / 1000).toFixed(2)}M</div>
            <div style={{ fontSize: 9, color: "#6b7280" }}>prob-weighted</div>
          </div>
        </div>

        {/* View Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 16, flexWrap: "wrap" }}>
          {btn("bridge", view, setView, "#60a5fa", "Demand × Scenarios")}
          {btn("share", view, setView, "#f97316", "Share Regression")}
          {btn("salesfan", view, setView, "#c084fc", "Implied SXC Sales")}
          {btn("waterfall", view, setView, "#60a5fa", "Contract Waterfall")}
          {btn("margins", view, setView, "#22d3ee", "Margin Model")}
          {btn("revfan", view, setView, "#c084fc", "Revenue Fan")}
          {btn("cogspct", view, setView, "#f87171", "COGS %")}
          {btn("gpfan", view, setView, "#4ade80", "Gross Profit Fan")}
          {btn("prob", view, setView, "#60a5fa", "Probability Table")}
          {(view === "waterfall" || view === "margins" || view === "gpfan" || view === "revfan" || view === "cogspct") && (
            <div style={{ marginLeft: "auto", display: "flex", gap: 4, alignItems: "center" }}>
              {(view === "waterfall" || view === "gpfan") && scenarios.map(s => btn(s.id, selScenario, setSelScenario, s.color, s.short))}
              {(view === "margins" || view === "revfan" || view === "gpfan" || view === "cogspct") && (<>
                <span style={{ fontSize: 8, color: "#6b7280", fontFamily: mn, marginLeft: 8 }}>COAL</span>
                {btn("high", coalScen, setCoalScen, "#f87171", "High")}
                {btn("base", coalScen, setCoalScen, "#f59e0b", "Base")}
                {btn("low", coalScen, setCoalScen, "#4ade80", "Low")}
              </>)}
            </div>
          )}
        </div>

        {/* ═══ DEMAND × SCENARIOS OVERLAY ═══ */}
        {view === "bridge" && (
          <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 4 }}>
              Demand Fan + Scenario Implied Demand
            </div>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 16 }}>
              Fan = statistical demand forecast · Colored lines = implied demand required for each contract scenario's sales · M tons
            </div>
            <ResponsiveContainer width="100%" height={460}>
              <ComposedChart data={demandOverlay} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="year" tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `${v}M`} domain={[4, 20]} />
                <Tooltip content={<Tip />} /><Legend wrapperStyle={{ fontSize: 9 }} />
                <ReferenceLine y={CEIL} stroke="rgba(22,163,74,0.3)" strokeDasharray="4 4" />
                {/* Fan areas */}
                <Area stackId="outer" dataKey="band2_base" stroke="none" fill="none" fillOpacity={0} connectNulls={false} legendType="none" tooltipType="none" />
                <Area stackId="outer" dataKey="band2_top" stroke="none" fill="#60a5fa" fillOpacity={0.1} connectNulls={false} legendType="none" tooltipType="none" />
                <Area stackId="inner" dataKey="band1_base" stroke="none" fill="none" fillOpacity={0} connectNulls={false} legendType="none" tooltipType="none" />
                <Area stackId="inner" dataKey="band1_top" stroke="none" fill="#60a5fa" fillOpacity={0.15} connectNulls={false} legendType="none" tooltipType="none" />
                {/* Fan lines */}
                <Line dataKey="actual" name="Actual" stroke="#111827" strokeWidth={2.5} dot={{ r: 3, fill: "#111827", stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false} />
                <Line dataKey="central" name="Central" stroke="rgba(0,0,0,0.3)" strokeWidth={1.5} strokeDasharray="4 3" dot={false} connectNulls={false} />
                <Line dataKey="up2" name="90% Band" stroke="rgba(0,0,0,0.15)" strokeWidth={1} strokeDasharray="3 3" dot={false} connectNulls={false} />
                <Line dataKey="down2" name="90% Lower" stroke="rgba(0,0,0,0.15)" strokeWidth={1} strokeDasharray="3 3" dot={false} connectNulls={false} legendType="none" />
                {/* Scenario implied demand lines */}
                {scenarios.map(s => (
                  <Line key={s.id} dataKey={`${s.id}_demand`} name={s.label} stroke={s.color} strokeWidth={2}
                    dot={{ r: 3, fill: s.color, stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false} />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* ═══ SHARE REGRESSION ═══ */}
        {view === "share" && (
          <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: 20, marginBottom: 20 }}>
            {/* Subtab buttons */}
            <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
              {btn("time", shareMode, setShareMode, "#f97316", "Share vs Time (Logistic)")}
              {btn("deficit", shareMode, setShareMode, "#22d3ee", "Share vs Deficit")}
            </div>

            {/* ── TIME-BASED ── */}
            {shareMode === "time" && (<>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 4 }}>
                Share of Deficit vs Time — Logistic Model
              </div>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 16 }}>
                Logistic: L={LOG_L}% k={logF?.k ?? "?"} t₀={logF?.t0 ? Math.round(2015 + logF.t0) : "?"} R²={logF?.r2?.toFixed(3) ?? "?"} · Assumes share rises with time as competitors exit
              </div>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={shareChart} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="year" tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} domain={[20, 100]} />
                  <Tooltip content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null;
                    return (
                      <div style={{ background: "rgba(255,255,255,0.98)", border: "1px solid #e5e7eb", borderRadius: 6, padding: "10px 14px", fontSize: 11, fontFamily: dm, color: "#374151", maxWidth: 280 }}>
                        <div style={{ fontWeight: 700, color: "#111827", marginBottom: 6, fontSize: 12 }}>{label}</div>
                        {payload.filter(p => p.value != null).map((p, i) => (
                          <div key={i} style={{ display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 2 }}>
                            <span style={{ color: p.color || p.stroke, fontSize: 10 }}>{p.name}</span>
                            <span style={{ color: "#111827", fontWeight: 600 }}>{typeof p.value === "number" ? `${p.value.toFixed(1)}%` : p.value}</span>
                          </div>
                        ))}
                      </div>
                    );
                  }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <ReferenceLine y={LOG_L} stroke="rgba(249,115,22,0.25)" strokeDasharray="3 3" />
                  <Line dataKey="lin" name="Linear (time)" stroke="#60a5fa" strokeWidth={1.5} strokeDasharray="6 3" dot={false} />
                  <Line dataKey="log" name={`Logistic (L=${LOG_L}%)`} stroke="#f97316" strokeWidth={2} strokeDasharray="4 2" dot={false} />
                  <Line dataKey="actual" name="Actual %" stroke="#60a5fa" strokeWidth={2.5} dot={{ r: 5, fill: "#60a5fa", stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false} />
                  <Line dataKey="projected" name="Projected %" stroke="#c084fc" strokeWidth={2.5} dot={{ r: 5, fill: "#c084fc", stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false} />
                </ComposedChart>
              </ResponsiveContainer>

              <div style={{ background: "#fff7ed", border: "1px solid #fed7aa", borderRadius: 8, padding: 12, marginTop: 12, fontSize: 11, color: "#6b7280", lineHeight: 1.6 }}>
                <strong style={{ color: "#f97316" }}>Limitation:</strong> This model says share rises monotonically with time. But time is just a proxy for competitive exit. If the deficit stays large (Management case), there is no reason SXC would capture 75%+ — other suppliers would remain viable. The deficit-based model addresses this directly.
              </div>
            </>)}

            {/* ── DEFICIT-BASED ── */}
            {shareMode === "deficit" && (<>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 4 }}>
                Share of Deficit vs Deficit Size — Linear Regression
              </div>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 16 }}>
                Share = {regShDef?.intercept?.toFixed(1) ?? "?"} + ({regShDef?.slope?.toFixed(2) ?? "?"}) × Deficit · R² = {regShDef?.r2?.toFixed(3) ?? "?"} · Smaller deficit → higher share (fewer alternatives)
              </div>

              {/* Scatter: Deficit (X) vs Share (Y) */}
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={defShareScatter} margin={{ top: 10, right: 20, left: 0, bottom: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="def" type="number" tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false} tickFormatter={v => `${v}M`} domain={[4, 12]} label={{ value: "Market Deficit (M tons)", position: "insideBottom", offset: -8, fontSize: 10, fill: "#6b7280" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} domain={[30, 70]} label={{ value: "SXC Share %", angle: -90, position: "insideLeft", fontSize: 10, fill: "#6b7280" }} />
                  <Tooltip content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0]?.payload;
                    if (!d) return null;
                    return (
                      <div style={{ background: "rgba(255,255,255,0.98)", border: "1px solid #e5e7eb", borderRadius: 6, padding: "10px 14px", fontSize: 11, fontFamily: dm, color: "#374151" }}>
                        <div style={{ fontWeight: 700, color: "#111827", marginBottom: 6 }}>{d.year}</div>
                        <div>Deficit: <strong style={{ color: "#22d3ee" }}>{d.def?.toFixed(2)}M</strong></div>
                        <div>SXC Sales: <strong style={{ color: "#60a5fa" }}>{d.sxc?.toFixed(3)}M</strong></div>
                        <div>Actual Share: <strong style={{ color: "#22d3ee" }}>{d.share?.toFixed(1)}%</strong></div>
                        <div>Fitted Share: <strong style={{ color: "#f59e0b" }}>{d.fit?.toFixed(1)}%</strong></div>
                      </div>
                    );
                  }} />
                  {/* Fit line */}
                  <Line dataKey="fit" stroke="#f59e0b" strokeWidth={2} strokeDasharray="6 3" dot={false} name="Linear Fit" activeDot={false} />
                  {/* Actual points as Line with no stroke, large dots */}
                  <Line dataKey="share" stroke="transparent" strokeWidth={0} dot={{ r: 6, fill: "#22d3ee", stroke: "#ffffff", strokeWidth: 2 }} activeDot={{ r: 8, fill: "#22d3ee", stroke: "#111827", strokeWidth: 2 }} name="Actual %" isAnimationActive={false} connectNulls={false} />
                </ComposedChart>
              </ResponsiveContainer>

              {/* Comparison table: Time vs Deficit for all scenarios */}
              <div style={{ fontSize: 13, fontWeight: 600, color: "#111827", marginTop: 20, marginBottom: 8 }}>
                Implied Sales Comparison: Time-Based vs Deficit-Based Share
              </div>
              <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 12 }}>
                Green delta = deficit model gives more sales (smaller deficit → higher share offsets volume loss). Red delta = deficit model gives less (large deficit → lower share penalizes optimism).
              </div>
              {deficitComparison.map(s => (
                <div key={s.id} style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: s.color, marginBottom: 4 }}>
                    {s.label} <span style={{ fontWeight: 400, color: "#6b7280" }}>z = {s.z >= 0 ? "+" : ""}{s.z.toFixed(1)}</span>
                  </div>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9, fontFamily: mn }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${s.color}30` }}>
                        {["Year", "Demand", "Deficit", "Time Sh", "Time SXC", "Def Sh", "Def SXC", "\u0394"].map((h, i) => (
                          <th key={i} style={{ padding: "4px 6px", textAlign: i === 0 ? "left" : "right", color: "#6b7280", fontWeight: 600, fontSize: 8 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(s.rows || []).map((r, ri) => (
                        <tr key={ri} style={{ borderBottom: "1px solid #f3f4f6" }}>
                          <td style={{ padding: "4px 6px", color: s.color, fontWeight: 600 }}>{r.yr}E</td>
                          <td style={{ padding: "4px 6px", textAlign: "right", color: "#6b7280" }}>{r.dem.toFixed(1)}M</td>
                          <td style={{ padding: "4px 6px", textAlign: "right", color: "#f59e0b" }}>{r.deficit.toFixed(2)}M</td>
                          <td style={{ padding: "4px 6px", textAlign: "right", color: "#f97316" }}>{r.timeSh.toFixed(1)}%</td>
                          <td style={{ padding: "4px 6px", textAlign: "right", color: "#f97316" }}>{(r.timeSXC / 1000).toFixed(2)}M</td>
                          <td style={{ padding: "4px 6px", textAlign: "right", color: "#22d3ee" }}>{r.defSh.toFixed(1)}%</td>
                          <td style={{ padding: "4px 6px", textAlign: "right", color: "#22d3ee" }}>{(r.defSXC / 1000).toFixed(2)}M</td>
                          <td style={{ padding: "4px 6px", textAlign: "right", color: r.delta >= 0 ? "#4ade80" : "#f87171", fontWeight: 600 }}>{r.delta >= 0 ? "+" : ""}{r.delta.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}

              <div style={{ background: "rgba(34,211,238,0.06)", border: "1px solid rgba(34,211,238,0.15)", borderRadius: 8, padding: 12, marginTop: 12, fontSize: 11, color: "#6b7280", lineHeight: 1.6 }}>
                <strong style={{ color: "#22d3ee" }}>Key insight:</strong> The deficit-based model penalizes the Management case (large deficit still leaves room for competitors, so ~72% share vs 75%) but rewards the Down cases (tiny deficit means SXC is effectively the last supplier, ~86% share vs 75%). The Base case is where both models converge — which makes sense, since the time trend was calibrated during the Base-like demand regime. <strong style={{ color: "#111827" }}>R² = {regShDef?.r2?.toFixed(3) ?? "?"}</strong> vs logistic R² = {logF?.r2?.toFixed(3) ?? "?"}.
              </div>
            </>)}
          </div>
        )}

        {/* ═══ IMPLIED SXC SALES FAN ═══ */}
        {view === "salesfan" && (
          <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 4 }}>
              Implied SXC Sales — Macro Fan + Contract Scenarios
            </div>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 16 }}>
              Purple fan = deficit-based share × deficit · Colored lines = contract-level scenario sales · K tons
            </div>
            <ResponsiveContainer width="100%" height={460}>
              <ComposedChart data={salesFan} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="year" tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `${(v / 1000).toFixed(1)}M`} domain={[0, 5000]} />
                <Tooltip content={<Tip />} /><Legend wrapperStyle={{ fontSize: 9 }} />
                {/* Fan areas */}
                <Area stackId="outer" dataKey="band2_base" stroke="none" fill="none" fillOpacity={0} connectNulls={false} legendType="none" tooltipType="none" />
                <Area stackId="outer" dataKey="band2_top" stroke="none" fill="#c084fc" fillOpacity={0.12} connectNulls={false} legendType="none" tooltipType="none" />
                <Area stackId="inner" dataKey="band1_base" stroke="none" fill="none" fillOpacity={0} connectNulls={false} legendType="none" tooltipType="none" />
                <Area stackId="inner" dataKey="band1_top" stroke="none" fill="#c084fc" fillOpacity={0.18} connectNulls={false} legendType="none" tooltipType="none" />
                {/* Fan lines */}
                <Line dataKey="actual" name="Actual" stroke="#111827" strokeWidth={2.5} dot={{ r: 3, fill: "#111827", stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false} />
                <Line dataKey="central" name="Central" stroke="#c084fc" strokeWidth={2} strokeDasharray="4 3" dot={{ r: 3, fill: "#c084fc", stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false} />
                <Line dataKey="up2" name="90% Band" stroke="#c084fc" strokeWidth={1} strokeDasharray="3 3" dot={false} connectNulls={false} />
                <Line dataKey="down2" name="90% Lower" stroke="#c084fc" strokeWidth={1} strokeDasharray="3 3" dot={false} connectNulls={false} legendType="none" />
                <Line dataKey="up1" name="68% Band" stroke="#d8b4fe" strokeWidth={1} strokeDasharray="2 2" dot={false} connectNulls={false} />
                <Line dataKey="down1" name="68% Lower" stroke="#d8b4fe" strokeWidth={1} strokeDasharray="2 2" dot={false} connectNulls={false} legendType="none" />
                {/* Scenario lines */}
                {scenarios.map(s => (
                  <Line key={s.id} dataKey={`${s.id}_sales`} name={s.label} stroke={s.color} strokeWidth={2}
                    dot={{ r: 3, fill: s.color, stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false} />
                ))}
              </ComposedChart>
            </ResponsiveContainer>

            {/* Sales table */}
            <div style={{ overflowX: "auto", marginTop: 16 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: mn }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
                    <th style={{ padding: "5px 8px", textAlign: "left", color: "#6b7280" }}>Year</th>
                    {scenarios.map(s => (
                      <th key={s.id} style={{ padding: "5px 8px", textAlign: "right", color: s.color, fontWeight: 600 }}>{s.short}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {fcYrs.map((yr, i) => (
                    <tr key={yr} style={{ borderBottom: "1px solid #f3f4f6" }}>
                      <td style={{ padding: "5px 8px", color: "#111827", fontWeight: 600 }}>{yr}E</td>
                      {scenarios.map(s => (
                        <td key={s.id} style={{ padding: "5px 8px", textAlign: "right", color: s.color }}>
                          {(allFC[s.id][i].total / 1000).toFixed(2)}M
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ═══ CONTRACT WATERFALL ═══ */}
        {view === "waterfall" && (
          <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#111827" }}>
                {scenarios.find(s => s.id === selScenario)?.label} — Contract Waterfall
              </div>
              <div style={{ fontSize: 13, fontWeight: 700, color: scenarios.find(s => s.id === selScenario)?.color, fontFamily: mn }}>
                P = {probTable.find(s => s.id === selScenario)?.prob || 0}%
              </div>
            </div>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 16 }}>Stacked by facility · K tons</div>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={selFC} margin={{ top: 5, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="year" tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `${(v / 1000).toFixed(1)}M`} domain={[0, 4500]} />
                <Tooltip content={<Tip />} /><Legend wrapperStyle={{ fontSize: 9 }} />
                <ReferenceLine x="2025" stroke="rgba(0,0,0,0.15)" strokeDasharray="6 4" />
                {barConfigs.map((b, i) => (
                  <Bar key={b.key} dataKey={b.key} name={b.name} stackId="s" fill={b.color}
                    radius={i === barConfigs.length - 1 ? [3, 3, 0, 0] : [0, 0, 0, 0]} />
                ))}
                <Line dataKey="total" name="Total" stroke="#111827" strokeWidth={2} strokeDasharray="4 3"
                  dot={{ r: 3, fill: "#111827", stroke: "#ffffff", strokeWidth: 2 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* ═══ MARGIN MODEL ═══ */}
        {view === "margins" && (
          <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 4 }}>
              Margin Architecture — Regressions & Two-Tier GP
            </div>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 20 }}>
              Coal price proxies market conditions · Revenue and spot margins scale with coal · Contracted margins are fixed
            </div>

            {/* Two regressions side by side */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 20 }}>
              <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, padding: 14 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#111827", marginBottom: 2 }}>Rev/ton = $102 + $1.43 × Coal</div>
                <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 10 }}>R² = 0.89 · Coal pass-through drives coke sale price</div>
                <ResponsiveContainer width="100%" height={220}>
                  <ComposedChart margin={{ top: 5, right: 30, left: -5, bottom: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis dataKey="coal" type="number" domain={[80, 300]} tick={{ fontSize: 9, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false}
                      label={{ value: "Met Coal $/ton", position: "insideBottom", offset: -6, fill: "#6b7280", fontSize: 9 }} />
                    <YAxis type="number" domain={[220, 520]} tick={{ fontSize: 9, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `$${v}`} />
                    <Tooltip content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0]?.payload;
                      return (<div style={{ background: "rgba(255,255,255,0.98)", border: "1px solid #e5e7eb", borderRadius: 5, padding: "8px 12px", fontSize: 10, fontFamily: dm, color: "#6b7280" }}>
                        <span style={{ color: "#111827", fontWeight: 600 }}>{d?.year}</span> — Coal ${d?.coal} → Rev ${d?.revTon}/t
                      </div>);
                    }} />
                    <Line data={[{ coal: 80, fit: REV_A + REV_B * 80 }, { coal: 300, fit: REV_A + REV_B * 300 }]} dataKey="fit" stroke="#c084fc" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
                    <Scatter data={finHist} dataKey="revTon" fill="#c084fc" stroke="#ffffff" strokeWidth={2} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, padding: 14 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#111827", marginBottom: 2 }}>GP/ton (blended) = $59 + $0.12 × Coal</div>
                <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 10 }}>R² = 0.56 · Stable ~$80/ton but hides two-tier structure</div>
                <ResponsiveContainer width="100%" height={220}>
                  <ComposedChart margin={{ top: 5, right: 30, left: -5, bottom: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis dataKey="coal" type="number" domain={[80, 300]} tick={{ fontSize: 9, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false}
                      label={{ value: "Met Coal $/ton", position: "insideBottom", offset: -6, fill: "#6b7280", fontSize: 9 }} />
                    <YAxis type="number" domain={[50, 100]} tick={{ fontSize: 9, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `$${v}`} />
                    <Tooltip content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0]?.payload;
                      return (<div style={{ background: "rgba(255,255,255,0.98)", border: "1px solid #e5e7eb", borderRadius: 5, padding: "8px 12px", fontSize: 10, fontFamily: dm, color: "#6b7280" }}>
                        <span style={{ color: "#111827", fontWeight: 600 }}>{d?.year}</span> — Coal ${d?.coal} → GP ${d?.gpTon}/t
                      </div>);
                    }} />
                    <Line data={[{ coal: 80, fit: GP_BL_A + GP_BL_B * 80 }, { coal: 300, fit: GP_BL_A + GP_BL_B * 300 }]} dataKey="fit" stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
                    <Scatter data={finHist} dataKey="gpTon" fill="#3b82f6" stroke="#ffffff" strokeWidth={2} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Two-tier GP cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 16 }}>
              <div style={{ background: "rgba(74,222,128,0.06)", border: "1px solid rgba(74,222,128,0.15)", borderRadius: 8, padding: 14 }}>
                <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4 }}>Contracted GP/ton</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: "#4ade80", fontFamily: mn }}>$95</div>
                <div style={{ fontSize: 10, color: "#6b7280", marginTop: 4, lineHeight: 1.5 }}>Fixed fee + operating efficiency + yield gains. Pass-through protected. Doesn't move with coal or demand.</div>
              </div>
              <div style={{ background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.15)", borderRadius: 8, padding: 14 }}>
                <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4 }}>Spot GP/ton = $17 + $0.18 × Coal</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: "#f59e0b", fontFamily: mn }}>${SPOT_GP_A + SPOT_GP_B * 226} <span style={{ fontSize: 12, fontWeight: 400, color: "#6b7280" }}>at $226 coal</span></div>
                <div style={{ fontSize: 10, color: "#6b7280", marginTop: 4, lineHeight: 1.5 }}>Market-exposed. Scales with coal (which scales with demand). No pass-through protection.</div>
              </div>
              <div style={{ background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.15)", borderRadius: 8, padding: 14 }}>
                <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4 }}>The Mix Shift Penalty</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: "#f87171", fontFamily: mn }}>$37/ton</div>
                <div style={{ fontSize: 10, color: "#6b7280", marginTop: 4, lineHeight: 1.5 }}>Every ton that moves from contracted → spot loses $37 in GP. As contracts expire, the blend compresses.</div>
              </div>
            </div>
            <div style={{ fontSize: 10, color: "#6b7280", lineHeight: 1.6 }}>
              <strong style={{ color: "#111827" }}>Total GP</strong> = (Contracted K tons × $95) + (Spot K tons × [$17 + $0.18 × Coal]). Coal forecast via inverse demand: Coal = (447 − 16.6 × Demand) × {coalScen === "high" ? "1.10" : coalScen === "low" ? "0.70" : "0.85"}.
            </div>
          </div>
        )}

        {/* ═══ REVENUE FAN ═══ */}
        {view === "revfan" && (
          <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 4 }}>
              Domestic Coke Revenue by Scenario ($M)
            </div>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 16 }}>
              Revenue = Volume × Rev/ton · Rev/ton = $102 + $1.43 × Coal · Coal from inverse demand ({coalScen})
            </div>
            <ResponsiveContainer width="100%" height={420}>
              <ComposedChart data={revGpFan} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="year" tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `$${v}M`} domain={[0, 2200]} />
                <Tooltip content={<Tip />} /><Legend wrapperStyle={{ fontSize: 9 }} />
                <ReferenceLine x="2025" stroke="rgba(0,0,0,0.15)" strokeDasharray="6 4" />
                <Line dataKey="actualRev" name="Actual Rev" stroke="#111827" strokeWidth={2.5} dot={{ r: 3, fill: "#111827", stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false} />
                {scenarios.map(s => (
                  <Line key={s.id} dataKey={`${s.id}_rev`} name={`${s.label} Rev`} stroke={s.color} strokeWidth={s.id === "base" ? 2.5 : 1.5}
                    dot={{ r: 3, fill: s.color, stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false}
                    strokeDasharray={s.id === "down" ? "6 3" : s.id === "nr_clf" ? "4 3" : "0"} />
                ))}
              </ComposedChart>
            </ResponsiveContainer>

            {/* Revenue table */}
            <div style={{ overflowX: "auto", marginTop: 16 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: mn }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
                    <th style={{ padding: "5px 8px", textAlign: "left", color: "#6b7280" }}>Year</th>
                    {scenarios.map(s => (
                      <th key={s.id} style={{ padding: "5px 8px", textAlign: "right", color: s.color, fontWeight: 600 }}>{s.short}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {fcYrs.map(yr => {
                    const row = revGpFan.find(d => d.year === yr.toString());
                    if (!row) return null;
                    return (
                      <tr key={yr} style={{ borderBottom: "1px solid #f3f4f6" }}>
                        <td style={{ padding: "5px 8px", color: "#111827", fontWeight: 600 }}>{yr}E</td>
                        {scenarios.map(s => (
                          <td key={s.id} style={{ padding: "5px 8px", textAlign: "right", color: s.color }}>${row[`${s.id}_rev`]?.toFixed(1)}M</td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ═══ COGS % ═══ */}
        {view === "cogspct" && (
          <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 4 }}>
              COGS as % of Revenue by Scenario
            </div>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 16 }}>
              COGS % = (Revenue − GP) / Revenue · Higher = worse margins · Mix shift from contracted to spot pushes COGS % up
            </div>
            <ResponsiveContainer width="100%" height={420}>
              <ComposedChart data={revGpFan} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="year" tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} domain={[70, 90]} />
                <Tooltip content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div style={{ background: "rgba(255,255,255,0.98)", border: "1px solid #e5e7eb", borderRadius: 6, padding: "10px 14px", fontSize: 11, fontFamily: dm, color: "#374151", maxWidth: 300 }}>
                      <div style={{ fontWeight: 700, color: "#111827", marginBottom: 6, fontSize: 12 }}>{label}</div>
                      {payload.filter(p => p.value != null).map((p, i) => (
                        <div key={i} style={{ display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 2 }}>
                          <span style={{ color: p.color || p.stroke, fontSize: 10 }}>{p.name}</span>
                          <span style={{ color: "#111827", fontWeight: 600 }}>{typeof p.value === "number" ? `${p.value.toFixed(1)}%` : p.value}</span>
                        </div>
                      ))}
                    </div>
                  );
                }} />
                <Legend wrapperStyle={{ fontSize: 9 }} />
                <ReferenceLine x="2025" stroke="rgba(0,0,0,0.15)" strokeDasharray="6 4" />
                <Line dataKey="actualCogsPct" name="Actual COGS %" stroke="#111827" strokeWidth={2.5} dot={{ r: 3, fill: "#111827", stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false} />
                {scenarios.map(s => (
                  <Line key={s.id} dataKey={`${s.id}_cogsPct`} name={`${s.label}`} stroke={s.color} strokeWidth={s.id === "base" ? 2.5 : 1.5}
                    dot={{ r: 3, fill: s.color, stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false}
                    strokeDasharray={s.id === "down" ? "6 3" : s.id === "nr_clf" ? "4 3" : "0"} />
                ))}
              </ComposedChart>
            </ResponsiveContainer>

            {/* COGS % table */}
            <div style={{ overflowX: "auto", marginTop: 16 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: mn }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
                    <th style={{ padding: "5px 8px", textAlign: "left", color: "#6b7280" }}>Year</th>
                    {scenarios.map(s => (
                      <th key={s.id} style={{ padding: "5px 8px", textAlign: "right", color: s.color, fontWeight: 600 }}>{s.short}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {fcYrs.map(yr => {
                    const row = revGpFan.find(d => d.year === yr.toString());
                    if (!row) return null;
                    return (
                      <tr key={yr} style={{ borderBottom: "1px solid #f3f4f6" }}>
                        <td style={{ padding: "5px 8px", color: "#111827", fontWeight: 600 }}>{yr}E</td>
                        {scenarios.map(s => (
                          <td key={s.id} style={{ padding: "5px 8px", textAlign: "right", color: s.color }}>{row[`${s.id}_cogsPct`]?.toFixed(1)}%</td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ═══ GROSS PROFIT FAN ═══ */}
        {view === "gpfan" && (
          <div>
            {/* GP fan lines */}
            <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: 20, marginBottom: 12 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 4 }}>
                Gross Profit by Scenario ($M)
              </div>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 16 }}>
                GP = (Contracted × $95) + (Spot × [$17 + $0.18×Coal]) · Coal: {coalScen}
              </div>
              <ResponsiveContainer width="100%" height={360}>
                <ComposedChart data={revGpFan} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="year" tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `$${v}M`} domain={[0, 420]} />
                  <Tooltip content={<Tip />} /><Legend wrapperStyle={{ fontSize: 9 }} />
                  <ReferenceLine x="2025" stroke="rgba(0,0,0,0.15)" strokeDasharray="6 4" />
                  <Line dataKey="actualGP" name="Actual GP" stroke="#111827" strokeWidth={2.5} dot={{ r: 3, fill: "#111827", stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false} />
                  {scenarios.map(s => (
                    <Line key={s.id} dataKey={`${s.id}_gp`} name={`${s.label} GP`} stroke={s.color} strokeWidth={s.id === "base" ? 2.5 : 1.5}
                      dot={{ r: 3, fill: s.color, stroke: "#ffffff", strokeWidth: 2 }} connectNulls={false}
                      strokeDasharray={s.id === "down" ? "6 3" : s.id === "nr_clf" ? "4 3" : "0"} />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>

              {/* GP table */}
              <div style={{ overflowX: "auto", marginTop: 16 }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: mn }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
                      <th style={{ padding: "5px 8px", textAlign: "left", color: "#6b7280" }}>Year</th>
                      {scenarios.map(s => (
                        <th key={s.id} style={{ padding: "5px 8px", textAlign: "right", color: s.color, fontWeight: 600 }}>{s.short}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {fcYrs.map(yr => {
                      const row = revGpFan.find(d => d.year === yr.toString());
                      if (!row) return null;
                      return (
                        <tr key={yr} style={{ borderBottom: "1px solid #f3f4f6" }}>
                          <td style={{ padding: "5px 8px", color: "#111827", fontWeight: 600 }}>{yr}E</td>
                          {scenarios.map(s => (
                            <td key={s.id} style={{ padding: "5px 8px", textAlign: "right", color: s.color }}>${row[`${s.id}_gp`]?.toFixed(1)}M</td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* GP stacked waterfall for selected scenario */}
            <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, padding: 20, marginBottom: 20 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 4 }}>
                {scenarios.find(s => s.id === selScenario)?.label} — Contract vs Spot GP ($M)
              </div>
              <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 16 }}>
                Green = contracted GP ($95/ton, stable) · Yellow = spot GP (variable via coal) · White line = total
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={gpWaterfall} margin={{ top: 5, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="year" tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={{ stroke: "#d1d5db" }} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "#6b7280", fontFamily: mn }} axisLine={false} tickLine={false} tickFormatter={v => `$${v}M`} />
                  <Tooltip content={<Tip />} /><Legend wrapperStyle={{ fontSize: 9 }} />
                  <ReferenceLine x="2025" stroke="rgba(0,0,0,0.15)" strokeDasharray="6 4" />
                  <Bar dataKey="contractGP" name="Contract GP ($M)" stackId="gp" fill="#4ade80" fillOpacity={0.55} />
                  <Bar dataKey="spotGP" name="Spot GP ($M)" stackId="gp" fill="#f59e0b" fillOpacity={0.5} radius={[3, 3, 0, 0]} />
                  <Line dataKey="totalGP" name="Total GP ($M)" stroke="#111827" strokeWidth={2} strokeDasharray="4 3"
                    dot={{ r: 3, fill: "#111827", stroke: "#ffffff", strokeWidth: 2 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ═══ PROBABILITY TABLE ═══ */}
        {view === "prob" && (
          <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#111827", marginBottom: 4 }}>Scenario Probability Bridge — 2032E</div>
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 16 }}>
              Scenario z-scores pre-assigned · Demand from fan at each z · Other/Spot = macro implied − contracted · PDF normalization for weights
            </div>

            {/* Year-by-year bridge */}
            <div style={{ overflowX: "auto", marginBottom: 20 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: mn }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
                    <th style={{ padding: "5px 8px", textAlign: "right", color: "#6b7280" }}>Year</th>
                    {scenarios.map(s => (
                      <th key={s.id} colSpan={4} style={{ padding: "5px 4px", textAlign: "center", color: s.color, fontWeight: 600, borderBottom: `2px solid ${s.color}30` }}>
                        {s.short} <span style={{ fontWeight: 400, color: "#6b7280", fontSize: 8 }}>z={scenarioZ[s.id] >= 0 ? "+" : ""}{scenarioZ[s.id].toFixed(1)}</span>
                      </th>
                    ))}
                  </tr>
                  <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
                    <th style={{ padding: "3px 8px" }}></th>
                    {scenarios.map(s => (
                      <React.Fragment key={`sub_${s.id}`}>
                        <th style={{ padding: "3px 4px", textAlign: "right", color: "#6b7280", fontSize: 8 }}>Dem</th>
                        <th style={{ padding: "3px 4px", textAlign: "right", color: "#f59e0b", fontSize: 8 }}>Def</th>
                        <th style={{ padding: "3px 4px", textAlign: "right", color: "#22d3ee", fontSize: 8 }}>Sh%</th>
                        <th style={{ padding: "3px 4px", textAlign: "right", color: "#6b7280", fontSize: 8 }}>Sales</th>
                      </React.Fragment>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {bridgeData.map((d, i) => {
                    return (
                      <tr key={d.year} style={{ borderBottom: "1px solid #f3f4f6" }}>
                        <td style={{ padding: "5px 8px", textAlign: "right", color: "#111827", fontWeight: 600 }}>{d.year}E</td>
                        {scenarios.map(s => {
                          const fc = allFC[s.id][i];
                          const dem = fc.impliedDemand;
                          const cap = getCap(fcYrs[i]);
                          const deficit = Math.max(0, dem - cap);
                          const share = deficit > 0 ? Math.max(30, Math.min(95, regShDef.slope * deficit + regShDef.intercept)) : 0;
                          return (
                            <React.Fragment key={s.id}>
                              <td style={{ padding: "5px 4px", textAlign: "right", color: "#6b7280" }}>{dem.toFixed(1)}</td>
                              <td style={{ padding: "5px 4px", textAlign: "right", color: "#f59e0b" }}>{deficit.toFixed(1)}</td>
                              <td style={{ padding: "5px 4px", textAlign: "right", color: "#22d3ee" }}>{share.toFixed(0)}%</td>
                              <td style={{ padding: "5px 4px", textAlign: "right", color: s.color, fontWeight: 600 }}>{(fc.total / 1000).toFixed(2)}</td>
                            </React.Fragment>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* 2032 Probability Summary */}
            <div style={{ fontSize: 12, fontWeight: 600, color: "#111827", marginBottom: 10 }}>2032E Probability Assignment (PDF-Normalized)</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 8, marginBottom: 16 }}>
              {probTable.map(s => {
                const lastFC = allFC[s.id][allFC[s.id].length - 1];
                const dem = lastFC.impliedDemand;
                const cap = getCap(2032);
                const deficit = Math.max(0, dem - cap);
                const share = deficit > 0 ? Math.max(30, Math.min(95, regShDef.slope * deficit + regShDef.intercept)) : 0;
                return (
                  <div key={s.id} style={{ background: `${s.color}0a`, border: `1px solid ${s.color}25`, borderRadius: 8, padding: 14, textAlign: "center" }}>
                    <div style={{ fontSize: 10, color: s.color, fontWeight: 600, marginBottom: 2 }}>{s.label}</div>
                    <div style={{ fontSize: 11, color: (s.z || 0) >= 0 ? "#4ade80" : "#f87171", fontFamily: mn, marginBottom: 4 }}>
                      z = {(s.z || 0) >= 0 ? "+" : ""}{(s.z || 0).toFixed(1)}
                    </div>
                    <div style={{ fontSize: 28, fontWeight: 700, color: s.color, fontFamily: mn }}>{s.prob}%</div>
                    <div style={{ fontSize: 9, color: "#6b7280", marginTop: 4 }}>
                      Dem {dem.toFixed(1)}M · Def {deficit.toFixed(1)}M
                    </div>
                    <div style={{ fontSize: 9, color: "#22d3ee" }}>
                      Share {share.toFixed(0)}%
                    </div>
                    <div style={{ fontSize: 10, color: "#111827", fontWeight: 600, marginTop: 2 }}>
                      {((s.sales || 0) / 1000).toFixed(2)}M tons
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Expected value bar */}
            <div style={{ background: "#f3f4f6", border: "1px solid #e5e7eb", borderRadius: 8, padding: 16, textAlign: "center" }}>
              <div style={{ fontSize: 10, color: "#6b7280", textTransform: "uppercase", letterSpacing: 1, fontFamily: mn, marginBottom: 4 }}>
                Probability-Weighted Expected Sales (2032E)
              </div>
              <div style={{ fontSize: 32, fontWeight: 700, color: "#111827", fontFamily: mn }}>
                {(expectedSales / 1000).toFixed(2)}M
              </div>
              <div style={{ fontSize: 11, color: "#6b7280", marginTop: 6 }}>
                = {probTable.map(s => `${s.prob}% × ${((s.sales || 0) / 1000).toFixed(2)}M`).join(" + ")}
              </div>
              {/* Visual bar */}
              <div style={{ display: "flex", height: 24, borderRadius: 6, overflow: "hidden", marginTop: 12 }}>
                {probTable.map(s => (
                  <div key={s.id} style={{
                    width: `${s.prob}%`, background: s.color, display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 8, fontWeight: 700, color: "#0a0c10", fontFamily: mn,
                  }}>
                    {s.prob > 8 ? `${s.short} ${s.prob}%` : ""}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Methodology */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 20 }}>
          <div style={{ background: "rgba(96,165,250,0.06)", border: "1px solid rgba(96,165,250,0.15)", borderRadius: 8, padding: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "#60a5fa", marginBottom: 6 }}>Top-Down (Macro)</div>
            <div style={{ fontSize: 10, color: "#6b7280", lineHeight: 1.7 }}>
              EWMA demand forecast (λ={LAMBDA}). Asymmetric σ. Deficit-based share regression (R²={regShDef?.r2?.toFixed(3) ?? "?"}). Each scenario assigned a demand z-score. Share varies by scenario deficit — smaller deficit → higher share.
            </div>
          </div>
          <div style={{ background: "#fff7ed", border: "1px solid #fed7aa", borderRadius: 8, padding: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "#f97316", marginBottom: 6 }}>Bridge Logic</div>
            <div style={{ fontSize: 10, color: "#6b7280", lineHeight: 1.7 }}>
              Contracted volumes are fixed inputs per scenario. Other/Spot/Export = macro implied total at scenario's z-score minus contracted. This is the plug — lowest-margin volume that absorbs macro uncertainty. PDF(z) normalization for probability weights.
            </div>
          </div>
          <div style={{ background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.15)", borderRadius: 8, padding: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "#f59e0b", marginBottom: 6 }}>Bottom-Up (Micro)</div>
            <div style={{ fontSize: 10, color: "#6b7280", lineHeight: 1.7 }}>
              Five contract-level scenarios. Contract volumes drive the high-margin base. Other/Spot fills the gap to macro-implied total — this is where margin compression hits. When contracts are lost, some volume rolls to spot at weaker economics.
            </div>
          </div>
        </div>

        <div style={{ padding: "12px 0", borderTop: "1px solid #e5e7eb", fontSize: 9, color: "#4b5563", fontFamily: mn, lineHeight: 1.6 }}>
          Bridge: Implied Demand = SXC Sales / Share(t) + Captive. Share(t) = {LOG_L}/(1+e^(-k(t−t₀))).
          z = (Demand − Central) / (σ±×√n). P = Φ(z). Scenario probability = band between adjacent CDF values.
          E[Sales 2032] = {(expectedSales / 1000).toFixed(3)}M tons.
        </div>
      </div>
    </div>
  );
}
