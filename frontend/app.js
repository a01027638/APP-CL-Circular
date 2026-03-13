const state = {
  segment: "todos",
  route: "todas",
  year: "todos",
  map: null,
  mapLayerGroup: null,
  mapLegend: null,
  rankingData: [],
  leadsData: [],
  closeProbExpanded: false,
};

const fmtInt = (v) => new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(v || 0);
const fmtMoney = (v) => `$${fmtInt(v)}`;
const fmtPct = (v) => `${((v || 0) * 100).toFixed(2)}%`;
const PRODUCT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"];
const CLUSTER_LABELS = {
  0: "Microexportadores",
  1: "Mid-Tier Afines",
  2: "Top-Tier",
  3: "LowMid-Tier",
};

function clusterLabelName(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "Cluster sin etiqueta";
  const id = Math.min(3, Math.max(0, Math.round(n)));
  return CLUSTER_LABELS[id] || `Cluster ${id}`;
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return await res.json();
}

function renderKPIs(kpis) {
  const grid = document.getElementById("kpiGrid");
  const containers = kpis.containers ?? Math.round((kpis.volume_tons || 0) / 20);
  const cards = [
    ["Volumen (tons)", fmtInt(kpis.volume_tons)],
    ["Valor (USD)", fmtMoney(kpis.value_usd)],
    ["Contenedores", fmtInt(containers)],
    ["Riesgo medio", fmtPct(kpis.risk)],
    ["Empresas analizadas", fmtInt(kpis.companies)],
  ];
  grid.innerHTML = cards
    .map(
      ([label, value]) =>
        `<article class="kpi-card"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div></article>`
    )
    .join("");
}

function renderTable(tableId, rows) {
  const table = document.getElementById(tableId);
  if (!rows.length) {
    table.innerHTML = "<tr><td>Sin datos</td></tr>";
    return;
  }
  const columns = Object.keys(rows[0]);
  const head = `<thead><tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;
  const body = `<tbody>${rows
    .map(
      (r) =>
        `<tr>${columns
          .map((c) => {
            const v = r[c];
            if (typeof v === "number") return `<td>${v.toFixed(2)}</td>`;
            if (v === null || v === undefined || v === "") return "<td>N/A</td>";
            return `<td>${v}</td>`;
          })
          .join("")}</tr>`
    )
    .join("")}</tbody>`;
  table.innerHTML = head + body;
}

function renderProductLegendTable(rows, colorByCode) {
  const table = document.getElementById("productLegendTable");
  if (!rows.length) {
    table.innerHTML = "<tr><td>Sin datos</td></tr>";
    return;
  }
  const head = `
    <thead>
      <tr>
        <th>codigo_arancelario</th>
        <th>nombre_producto</th>
      </tr>
    </thead>
  `;
  const body = `
    <tbody>
      ${rows
        .map((r) => {
          const color = colorByCode[r.codigo_arancelario] || "#1f77b4";
          return `
            <tr>
              <td>
                <span class="code-with-line">
                  <span class="series-line" style="background:${color};"></span>
                  <span>${r.codigo_arancelario}</span>
                </span>
              </td>
              <td>${r.nombre_producto}</td>
            </tr>
          `;
        })
        .join("")}
    </tbody>
  `;
  table.innerHTML = head + body;
}

function prettyHeader(key) {
  const map = {
    cluster: "Cluster",
    n_empresas: "Tamano",
    companies: "Empresas",
    participacion_mercado: "Participación mercado",
    ventas_anuales_musd: "Ventas anuales MUSD",
    frecuencia_exportacion_dias: "Frecuencia exportación días",
    presencia_internacional: "Presencia internacional",
    reporte_esg: "Reporte ESG",
    movimiento_cadena_frio: "Movimiento cadena de frío",
    volumen_exportaciones_musd: "Volumen exportaciones MUSD",
    mas_40_viajes_mensuales: "Mas 40 viajes mensuales",
  };
  if (map[key]) return map[key];
  return String(key)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function renderClusterProfileTable(data) {
  const table = document.getElementById("clusterProfileTable");
  if (!table) return;

  const profiles = Array.isArray(data?.profiles) ? [...data.profiles] : [];
  if (!profiles.length) {
    table.innerHTML = "<tr><td>Sin datos de clusters</td></tr>";
    return;
  }

  const companiesByCluster = new Map();
  (data?.pca || []).forEach((row) => {
    const key = String(row.cluster);
    if (!companiesByCluster.has(key)) companiesByCluster.set(key, []);
    const bucket = companiesByCluster.get(key);
    if (row.empresa && !bucket.includes(row.empresa)) bucket.push(row.empresa);
  });

  const numericKeys = Array.from(
    new Set(
      profiles.flatMap((r) =>
        Object.keys(r).filter((k) => !["cluster", "n_empresas"].includes(k))
      )
    )
  );
  const headers = ["cluster", "n_empresas", "companies", ...numericKeys];
  const head = `<thead><tr>${headers.map((h) => `<th>${prettyHeader(h)}</th>`).join("")}</tr></thead>`;

  profiles.sort((a, b) => Number(a.cluster) - Number(b.cluster));
  const body = `<tbody>${profiles
    .map((row) => {
      const clusterKey = String(row.cluster);
      const companyNames = (companiesByCluster.get(clusterKey) || []).join(", ");
      return `<tr>${headers
        .map((h) => {
          if (h === "cluster") return `<td>${clusterLabelName(clusterKey)}</td>`;
          if (h === "n_empresas") return `<td>${row.n_empresas ?? "-"}</td>`;
          if (h === "companies") {
            const needsToggle = companyNames.length > 120;
            return `<td>
              <div class="cluster-companies cluster-companies--collapsed">${companyNames || "-"}</div>
              ${needsToggle ? '<button type="button" class="cluster-more-btn">Ver mas</button>' : ""}
            </td>`;
          }
          const val = row[h];
          if (typeof val === "number") return `<td>${val.toFixed(2)}</td>`;
          const parsed = Number(val);
          return `<td>${Number.isFinite(parsed) ? parsed.toFixed(2) : val ?? "-"}</td>`;
        })
        .join("")}</tr>`;
    })
    .join("")}</tbody>`;

  table.innerHTML = head + body;
  table.querySelectorAll(".cluster-more-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const box = btn.parentElement?.querySelector(".cluster-companies");
      if (!box) return;
      const collapsed = box.classList.toggle("cluster-companies--collapsed");
      btn.textContent = collapsed ? "Ver mas" : "Ver menos";
    });
  });
}

function renderClusterLoadingsTable(rows) {
  const table = document.getElementById("clusterLoadingsTable");
  if (!table) return;
  if (!rows || !rows.length) {
    table.innerHTML = "<tr><td>Sin loadings disponibles</td></tr>";
    return;
  }

  const head = `
    <thead>
      <tr>
        <th>Variable</th>
        <th>PC1</th>
        <th>PC2</th>
        <th>PC3</th>
      </tr>
    </thead>
  `;
  const body = `
    <tbody>
      ${rows
        .map(
          (r) => `
            <tr>
              <td>${r.variable}</td>
              <td>${Number(r.pc_1 || 0).toFixed(4)}</td>
              <td>${Number(r.pc_2 || 0).toFixed(4)}</td>
              <td>${Number(r.pc_3 || 0).toFixed(4)}</td>
            </tr>
          `
        )
        .join("")}
    </tbody>
  `;
  table.innerHTML = head + body;
}

function renderForecastDetailTables(data) {
  const renderMeta = (tableId, title, detail) => {
    const table = document.getElementById(tableId);
    if (!table) return;
    if (!detail) {
      table.innerHTML = "<tr><td>Sin detalle disponible</td></tr>";
      return;
    }
    const rows = [
      { metric: "Segmento", value: title },
      { metric: "Modelo", value: `ARIMAX(${(detail.order || []).join(",")})` },
      { metric: "AIC", value: detail.aic },
      { metric: "BIC", value: detail.bic },
      { metric: "HQIC", value: detail.hqic },
      { metric: "Log-Likelihood", value: detail.llf },
      { metric: "Observaciones", value: detail.nobs },
    ];
    renderTable(tableId, rows);
  };

  const renderCoef = (tableId, detail) => {
    const table = document.getElementById(tableId);
    if (!table) return;
    const rows = detail?.coefficients || [];
    if (!rows.length) {
      table.innerHTML = "<tr><td>Sin coeficientes disponibles</td></tr>";
      return;
    }
    const coefRows = rows.map((r) => ({
      term: r.term,
      coef: r.coef,
      std_err: r.std_err,
      p_value: r.p_value,
    }));
    renderTable(tableId, coefRows);
  };

  renderMeta("forecastBovinoMetaTable", "Bovino", data?.bovino?.model_detail);
  renderCoef("forecastBovinoCoefTable", data?.bovino?.model_detail);
  renderMeta("forecastPorcinoMetaTable", "Porcino", data?.porcino?.model_detail);
  renderCoef("forecastPorcinoCoefTable", data?.porcino?.model_detail);
}

function renderRouteRiskScatterChart(targetId, scatterData, title) {
  Plotly.newPlot(
    targetId,
    [
      {
        x: scatterData.map((r) => r.average_yearly_tons),
        y: scatterData.map((r) => r.risk_index),
        text: scatterData.map((r) => r.route_name || r.route_id || "Ruta"),
        mode: "markers",
        type: "scatter",
        marker: {
          size: 11,
          color: scatterData.map((r) => r.risk_index),
          colorscale: [
            [0.0, "#FDD835"],
            [0.5, "#FB8C00"],
            [1.0, "#C62828"],
          ],
          cmin: 0,
          cmax: 100,
          showscale: true,
          colorbar: { title: "Riesgo" },
        },
        hovertemplate:
          "<b>%{text}</b><br>Average yearly tons: %{x:,.0f}<br>Risk index: %{y:.2f}<extra></extra>",
      },
    ],
    {
      title,
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
      xaxis: { title: "Toneladas promedio" },
      yaxis: { title: "Índice de riesgo", range: [0, 100] },
    },
    { responsive: true }
  );
}

function renderTopRiskRoutesBarChart(targetId, rows, title) {
  const top5 = [...rows]
    .sort((a, b) => Number(b.risk_index || 0) - Number(a.risk_index || 0))
    .slice(0, 5);

  Plotly.newPlot(
    targetId,
    [
      {
        x: top5.map((r) => Number(r.risk_index || 0)),
        y: top5.map((r) => r.route_name || r.route_id || "Ruta"),
        type: "bar",
        orientation: "h",
        marker: {
          color: "#0f3f73",
        },
        hovertemplate: "<b>%{y}</b><br>Índice de riesgo: %{x:.2f}<extra></extra>",
      },
    ],
    {
      title,
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
      margin: { l: 170, r: 20, t: 60, b: 50 },
      xaxis: { title: "Índice de riesgo", range: [0, 100] },
      yaxis: { autorange: "reversed" },
    },
    { responsive: true }
  );
}

function setupTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("tab--active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("tab-content--active"));
      tab.classList.add("tab--active");
      document.getElementById(`tab-${tab.dataset.tab}`).classList.add("tab-content--active");
      if (tab.dataset.tab === "map" && state.map) {
        // Leaflet requiere recalcular el tamanio cuando el contenedor pasa de hidden a visible.
        setTimeout(() => {
          state.map.invalidateSize();
          ["mapRiskTonsScatterChart", "mapTopRiskRoutesChart"].forEach((id) => {
            const el = document.getElementById(id);
            if (el && typeof Plotly !== "undefined" && el.data) Plotly.Plots.resize(el);
          });
        }, 80);
      }
      if (tab.dataset.tab === "priority") {
        setTimeout(() => {
          ["closeProbFeatureImportanceChart", "closeProbClusterProbChart"].forEach((id) => {
            const el = document.getElementById(id);
            if (el && typeof Plotly !== "undefined" && el.data) Plotly.Plots.resize(el);
          });
        }, 80);
      }
    });
  });
}

function openTab(tabName) {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((t) => t.classList.remove("tab--active"));
  document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("tab-content--active"));
  const tab = document.querySelector(`.tab[data-tab="${tabName}"]`);
  if (!tab) return;
  tab.classList.add("tab--active");
  const pane = document.getElementById(`tab-${tabName}`);
  if (pane) pane.classList.add("tab-content--active");

  if (tabName === "map" && state.map) {
    setTimeout(() => {
      state.map.invalidateSize();
      ["mapRiskTonsScatterChart", "mapTopRiskRoutesChart"].forEach((id) => {
        const el = document.getElementById(id);
        if (el && typeof Plotly !== "undefined" && el.data) Plotly.Plots.resize(el);
      });
    }, 80);
  }

  if (tabName === "priority") {
    setTimeout(() => {
      ["closeProbFeatureImportanceChart", "closeProbClusterProbChart"].forEach((id) => {
        const el = document.getElementById(id);
        if (el && typeof Plotly !== "undefined" && el.data) Plotly.Plots.resize(el);
      });
    }, 80);
  }
}

function syncClusterKValue() {
  const slider = document.getElementById("clusterK");
  const out = document.getElementById("clusterKValue");
  if (!slider || !out) return;
  out.textContent = slider.value;
}

async function loadFilters() {
  const data = await api("/api/filters");
  const segmentSelect = document.getElementById("segmentSelect");
  const routeSelect = document.getElementById("routeSelect");
  const yearSelect = document.getElementById("yearSelect");
  segmentSelect.innerHTML = `<option value="todos">todos</option>${data.segments
    .map((s) => `<option value="${s}">${s}</option>`)
    .join("")}`;
  routeSelect.innerHTML = `<option value="todas">todas</option>${data.routes
    .map((s) => `<option value="${s}">${s}</option>`)
    .join("")}`;
  yearSelect.innerHTML = `<option value="todos" selected>Todos</option>${(data.years || [])
    .map((y) => `<option value="${y}">${y}</option>`)
    .join("")}`;
}

async function loadMarket() {
  const liveSegment = document.getElementById("segmentSelect")?.value || state.segment;
  const liveRoute = document.getElementById("routeSelect")?.value || state.route;
  const liveYear = document.getElementById("yearSelect")?.value || state.year;
  state.segment = liveSegment;
  state.route = liveRoute;
  state.year = liveYear;

  const params = new URLSearchParams({
    segment: state.segment,
    route: state.route,
    year: state.year,
  });
  const data = await api(`/api/kpis?${params.toString()}`);
  renderKPIs(data.kpis);

  const productByCode = {};
  data.product_labels.forEach((item) => {
    productByCode[item.codigo_arancelario] = item.nombre_producto;
  });

  const allowedCodesBySegment = {
    bovino: new Set(["0201", "0202"]),
    porcino: new Set(["0203"]),
  };
  const allowedCodes = allowedCodesBySegment[state.segment] || null;
  const filteredTrend = (data.product_trend || []).filter((row) =>
    allowedCodes ? allowedCodes.has(String(row.codigo)) : true
  );

  const grouped = {};
  filteredTrend.forEach((row) => {
    if (!grouped[row.codigo]) grouped[row.codigo] = [];
    grouped[row.codigo].push(row);
  });
  const codes = Object.keys(grouped).sort();
  const colorByCode = {};
  codes.forEach((code, idx) => {
    colorByCode[code] = PRODUCT_COLORS[idx % PRODUCT_COLORS.length];
  });
  const traces = codes.map((code) => {
    const rows = grouped[code].sort((a, b) => a.year - b.year);
    const productName = productByCode[code] || "Producto";
    const shortName = productName.length > 42 ? `${productName.slice(0, 42)}...` : productName;
    return {
      x: rows.map((r) => String(r.year)),
      y: rows.map((r) => r.valor_exportado),
      type: "scatter",
      mode: "lines+markers",
      name: `${code} - ${shortName}`,
      line: { color: colorByCode[code], width: 2.6 },
      marker: { color: colorByCode[code], size: 6 },
      hovertemplate: `<b>${code}</b><br>${productName}<br>Año: %{x}<br>Valor: %{y:,.0f}<extra></extra>`,
    };
  });

  Plotly.newPlot(
    "trendChart",
    traces,
    {
      title: "Serie de tiempo por producto exportado (anual)",
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
      xaxis: { title: "Año" },
      yaxis: { title: "Valor exportado" },
      showlegend: false,
    },
    { responsive: true }
  );

  const filteredLabels = (data.product_labels || []).filter((row) =>
    allowedCodes ? allowedCodes.has(String(row.codigo_arancelario)) : true
  );
  renderProductLegendTable(filteredLabels, colorByCode);

  const states = data.top_states || [];
  const normalizeStateName = (name) =>
    name === "Veracruz de Ignacio de la Llave" ? "Veracruz" : name;
  const segmentLabel =
    state.segment === "bovino" ? "Bovino" : state.segment === "porcino" ? "Porcino" : "Bovino + Porcino";
  Plotly.newPlot(
    "topStatesChart",
    [
      {
        x: states.map((r) => r.trade_value),
        y: states.map((r) => normalizeStateName(r.state)),
        type: "bar",
        orientation: "h",
        marker: {
          color: state.segment === "porcino" ? "#15a35b" : "#0f3f73",
        },
      },
    ],
    {
      title: `Top 5 estados exportadores (${segmentLabel})`,
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
      margin: { l: 140, r: 20, t: 60, b: 50 },
      xaxis: { title: "Trade Value" },
      yaxis: { autorange: "reversed" },
    },
    { responsive: true }
  );

  Plotly.newPlot(
    "routeRiskChart",
    [
      {
        labels: (data.company_share || []).map((r) => r.label),
        values: (data.company_share || []).map((r) => r.participacion_mercado),
        type: "pie",
        textinfo: "none",
        hovertemplate: "<b>%{label}</b><br>Participación: %{value:.2f}%<extra></extra>",
        marker: {
          line: { color: "#ffffff", width: 1 },
        },
      },
    ],
    {
      title: "Top Empresas Exportadoras (MX/USA)",
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
      showlegend: true,
      legend: { orientation: "v", x: 1.02, y: 0.5 },
    },
    { responsive: true }
  );

  const scatterData = data.route_tons_risk || [];
  renderRouteRiskScatterChart(
    "riskTonsScatterChart",
    scatterData,
    "Toneladas promedio vs indice de riesgo por ruta"
  );
  renderRouteRiskScatterChart(
    "mapRiskTonsScatterChart",
    scatterData,
    "Volumen vs índice de riesgo por ruta"
  );
  renderTopRiskRoutesBarChart(
    "mapTopRiskRoutesChart",
    scatterData,
    "Top 5 rutas con mayor índice de riesgo"
  );
}

async function loadRanking() {
  const params = new URLSearchParams({
    lookback_months: "12",
    year: state.year,
    w1: document.getElementById("w1").value,
    w2: document.getElementById("w2").value,
    w3: document.getElementById("w3").value,
    w4: document.getElementById("w4").value,
    w5: document.getElementById("w5").value,
    w6: document.getElementById("w6").value,
    top_n: "30",
  });
  const data = await api(`/api/ranking?${params.toString()}`);
  state.rankingData = data.data;

  renderTable(
    "rankingTable",
    data.data.map((r) => ({
      empresa: r.empresa,
      segmento: r.segmento,
      score: r.opportunity_score,
      valor_usd: r.value_usd,
      riesgo: r.incidence_probability,
      drivers: r.drivers_score,
    }))
  );

  Plotly.newPlot(
    "rankingScatter",
    [
      {
        x: data.data.map((r) => r.value_usd),
        y: data.data.map((r) => r.incidence_probability),
        text: data.data.map((r) => r.empresa),
        mode: "markers",
        marker: {
          size: data.data.map((r) => Math.max(8, Math.min(30, r.volume_tons / 40))),
          color: data.data.map((r) => r.opportunity_score),
          colorscale: "GnBu",
        },
      },
    ],
    {
      title: "Matriz valor-riesgo-oportunidad",
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
      xaxis: { title: "Valor USD" },
      yaxis: { title: "Riesgo incidencia" },
    },
    { responsive: true }
  );
}

async function loadMap() {
  const params = new URLSearchParams({
    segment: state.segment,
    route: state.route,
    year: state.year,
  });
  const data = await api(`/api/map-flows?${params.toString()}`);
  if (!state.map) {
    state.map = L.map("flowMap").setView([30, -104], 4.4);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: "&copy; OpenStreetMap",
    }).addTo(state.map);
    state.mapLayerGroup = L.layerGroup().addTo(state.map);
    // El mapa se inicializa durante el primer refresh, aun cuando el tab puede estar oculto.
    setTimeout(() => state.map.invalidateSize(), 120);
  }

  state.mapLayerGroup.clearLayers();
  if (state.mapLegend) {
    state.map.removeControl(state.mapLegend);
    state.mapLegend = null;
  }

  const rows = data.data || [];
  if (!rows.length) return;

  const vols = rows.map((r) => Number(r.average_yearly_tons ?? r.volumen ?? 0)).filter((v) => Number.isFinite(v));
  const vMin = vols.length ? Math.min(...vols) : 0;
  const vMax = vols.length ? Math.max(...vols) : 1;
  const volNorm = (v) => {
    if (vMax <= vMin) return 0.5;
    return (v - vMin) / (vMax - vMin);
  };

  const flowColor = (n) => {
    if (n < 0.2) return "#9fd3fb";
    if (n < 0.4) return "#77bdf4";
    if (n < 0.6) return "#4ea5eb";
    if (n < 0.8) return "#2f84d2";
    return "#1f5fa7";
  };

  const curvedPath = (lat1, lon1, lat2, lon2, bend = 0.18, points = 28) => {
    const mx = (lon1 + lon2) / 2;
    const my = (lat1 + lat2) / 2;
    const dx = lon2 - lon1;
    const dy = lat2 - lat1;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    const cx = mx + nx * len * bend;
    const cy = my + ny * len * bend;
    const out = [];
    for (let i = 0; i <= points; i += 1) {
      const t = i / points;
      const x = (1 - t) * (1 - t) * lon1 + 2 * (1 - t) * t * cx + t * t * lon2;
      const y = (1 - t) * (1 - t) * lat1 + 2 * (1 - t) * t * cy + t * t * lat2;
      out.push([y, x]);
    }
    return out;
  };

  const ordered = [...rows].sort(
    (a, b) => Number(a.average_yearly_tons ?? a.volumen ?? 0) - Number(b.average_yearly_tons ?? b.volumen ?? 0)
  );
  const bounds = [];

  ordered.forEach((r, idx) => {
    const originLat = Number(r.origin_lat);
    const originLon = Number(r.origin_lon);
    const destLat = Number(r.dest_lat);
    const destLon = Number(r.dest_lon);
    if (![originLat, originLon, destLat, destLon].every(Number.isFinite)) return;

    const vol = Number(r.average_yearly_tons ?? r.volumen ?? 0);
    const n = volNorm(vol);
    const color = flowColor(n);
    const companies = Array.isArray(r.companies) ? r.companies.slice(0, 8) : [];
    const destinations = Array.isArray(r.destinations) ? r.destinations.slice(0, 8) : [];
    const states = Array.isArray(r.states) ? r.states.slice(0, 8) : [];
    const companiesText = companies.length ? companies.join(", ") : r.empresa || "N/A";
    const destinationText = destinations.length ? destinations.join(", ") : "N/A";
    const statesText = states.length ? states.join(", ") : "N/A";
    const routeLabel = r.route_name || r.ruta_logistica || "Ruta";

    const bend = ((idx % 5) - 2) * 0.03;
    const linePoints = curvedPath(originLat, originLon, destLat, destLon, 0.14 + bend, 24);
    const polyline = L.polyline(linePoints, {
      color,
      weight: Math.max(2, Math.min(10, 1.8 + n * 8)),
      opacity: 0.7,
      lineCap: "round",
      lineJoin: "round",
    });
    polyline.bindPopup(
      `<strong>${routeLabel}</strong><br/>Volumen transportado (tons/año): ${fmtInt(
        vol
      )}<br/>Empresas (${r.company_count ?? companies.length}): ${companiesText}<br/>Destinos: ${destinationText}<br/>Estados destino: ${statesText}`
    );
    polyline.addTo(state.mapLayerGroup);
    bounds.push([originLat, originLon], [destLat, destLon]);
  });

  const destAgg = new Map();
  rows.forEach((r) => {
    const destLat = Number(r.dest_lat);
    const destLon = Number(r.dest_lon);
    if (!Number.isFinite(destLat) || !Number.isFinite(destLon)) return;
    const key = `${destLat.toFixed(4)},${destLon.toFixed(4)}`;
    if (!destAgg.has(key)) {
      destAgg.set(key, { lat: destLat, lon: destLon, vol: 0, routes: 0 });
    }
    const item = destAgg.get(key);
    item.vol += Number(r.average_yearly_tons ?? r.volumen ?? 0);
    item.routes += 1;
  });

  const destVols = [...destAgg.values()].map((d) => d.vol);
  const dMin = destVols.length ? Math.min(...destVols) : 0;
  const dMax = destVols.length ? Math.max(...destVols) : 1;
  const dNorm = (v) => (dMax <= dMin ? 0.5 : (v - dMin) / (dMax - dMin));

  [...destAgg.values()].forEach((d) => {
    const n = dNorm(d.vol);
    const marker = L.circleMarker([d.lat, d.lon], {
      radius: 5 + n * 13,
      color: "#0b3558",
      weight: 1,
      fillColor: "#1abc73",
      fillOpacity: 0.5 + n * 0.35,
    });
    marker.bindPopup(
      `<strong>Nodo de destino</strong><br/>Volumen total (tons/año): ${fmtInt(d.vol)}<br/>Rutas conectadas: ${d.routes}`
    );
    marker.addTo(state.mapLayerGroup);
  });

  if (bounds.length) {
    state.map.fitBounds(bounds, { padding: [24, 24], maxZoom: 6 });
  }

  state.mapLegend = L.control({ position: "bottomright" });
  state.mapLegend.onAdd = function onAdd() {
    const div = L.DomUtil.create("div", "map-volume-legend");
    div.style.background = "rgba(255,255,255,0.92)";
    div.style.border = "1px solid #d5e1ed";
    div.style.borderRadius = "10px";
    div.style.padding = "8px 10px";
    div.style.fontSize = "12px";
    div.style.color = "#0b3558";
    div.style.lineHeight = "1.3";
    div.innerHTML = `
      <strong>Volumen por ruta</strong><br/>
      <span style="display:inline-block;width:12px;height:4px;background:#9fd3fb;margin-right:6px;"></span>Bajo<br/>
      <span style="display:inline-block;width:12px;height:4px;background:#4ea5eb;margin-right:6px;"></span>Medio<br/>
      <span style="display:inline-block;width:12px;height:4px;background:#1f5fa7;margin-right:6px;"></span>Alto
    `;
    return div;
  };
  state.mapLegend.addTo(state.map);
}

async function loadForecast() {
  const params = new URLSearchParams({
    horizon_q: document.getElementById("forecastHorizonQ").value,
  });
  const data = await api(`/api/forecast-arimax-quarterly?${params.toString()}`);
  renderForecastDetailTables(data);

  const bM = data?.bovino?.metrics || {};
  const pM = data?.porcino?.metrics || {};
  const avg = (a, b) => {
    const vals = [a, b].filter((v) => Number.isFinite(v));
    if (!vals.length) return null;
    return vals.reduce((x, y) => x + y, 0) / vals.length;
  };
  const mapeAvg = avg(bM.mape, pM.mape);
  const rmseAvg = avg(bM.rmse, pM.rmse);
  const maeAvg = avg(bM.mae, pM.mae);

  document.getElementById("mapeCard").innerHTML = `<strong>MAPE promedio</strong><div>${
    mapeAvg !== null ? mapeAvg.toFixed(2) + "%" : "N/A"
  }</div>`;
  document.getElementById("rmseCard").innerHTML = `<strong>RMSE promedio</strong><div>${
    rmseAvg !== null ? rmseAvg.toFixed(2) : "N/A"
  }</div>`;
  document.getElementById("maeCard").innerHTML = `<strong>MAE promedio</strong><div>${
    maeAvg !== null ? maeAvg.toFixed(2) : "N/A"
  }</div>`;

  const modelRows = [
    {
      segmento: "bovino",
      modelo: `ARIMAX(${(bM.order || []).join(",")})`,
      mape: bM.mape ?? null,
      rmse: bM.rmse ?? null,
      mae: bM.mae ?? null,
    },
    {
      segmento: "porcino",
      modelo: `ARIMAX(${(pM.order || []).join(",")})`,
      mape: pM.mape ?? null,
      rmse: pM.rmse ?? null,
      mae: pM.mae ?? null,
    },
  ];
  renderTable("forecastModelTable", modelRows);

  const bovHist = data?.bovino?.history || [];
  const bovFc = data?.bovino?.forecast || [];
  const porHist = data?.porcino?.history || [];
  const porFc = data?.porcino?.forecast || [];

  Plotly.newPlot(
    "forecastChart",
    [
      {
        x: bovHist.map((r) => r.date),
        y: bovHist.map((r) => r.value),
        type: "scatter",
        mode: "lines",
        name: "Bovino historico",
        line: { color: "#0F3F73", width: 2.4 },
      },
      {
        x: bovFc.map((r) => r.date),
        y: bovFc.map((r) => r.forecast),
        type: "scatter",
        mode: "lines",
        name: "Bovino forecast",
        line: { color: "#2D6FA3", width: 2.6, dash: "dash" },
      },
      {
        x: [...bovFc.map((r) => r.date), ...bovFc.map((r) => r.date).reverse()],
        y: [...bovFc.map((r) => r.upper), ...bovFc.map((r) => r.lower).reverse()],
        fill: "toself",
        type: "scatter",
        mode: "lines",
        line: { color: "rgba(0,0,0,0)" },
        fillcolor: "rgba(45,111,163,0.18)",
        name: "Bovino 95% CI",
      },
      {
        x: porHist.map((r) => r.date),
        y: porHist.map((r) => r.value),
        type: "scatter",
        mode: "lines",
        name: "Porcino historico",
        line: { color: "#15A35B", width: 2.4 },
      },
      {
        x: porFc.map((r) => r.date),
        y: porFc.map((r) => r.forecast),
        type: "scatter",
        mode: "lines",
        name: "Porcino forecast",
        line: { color: "#26B36E", width: 2.6, dash: "dash" },
      },
      {
        x: [...porFc.map((r) => r.date), ...porFc.map((r) => r.date).reverse()],
        y: [...porFc.map((r) => r.upper), ...porFc.map((r) => r.lower).reverse()],
        fill: "toself",
        type: "scatter",
        mode: "lines",
        line: { color: "rgba(0,0,0,0)" },
        fillcolor: "rgba(38,179,110,0.14)",
        name: "Porcino 95% CI",
      },
    ],
    {
      title: "ARIMAX trimestral con exógena tipo de cambio (lag 1 trimestre)",
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
      xaxis: { title: "Fecha trimestral" },
      yaxis: { title: "Valor exportado" },
    },
    { responsive: true }
  );
}

async function loadScenarios() {
  const params = new URLSearchParams({
    adoption_rate: document.getElementById("adoptionRate").value,
    adoption_effectiveness: document.getElementById("adoptionEff").value,
    risk_multiplier: document.getElementById("riskMul").value,
    expansion_growth: document.getElementById("expansion").value,
    year: state.year,
  });
  const data = await api(`/api/scenarios?${params.toString()}`);
  Plotly.newPlot(
    "scenarioChart",
    [
      {
        x: data.summary.map((r) => r.escenario),
        y: data.summary.map((r) => r.perdida_esperada_usd),
        type: "bar",
        marker: { color: data.summary.map((r) => r.loss_rate), colorscale: "Reds" },
      },
    ],
    {
      title: "Pérdida esperada por escenario",
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
    },
    { responsive: true }
  );
}

async function loadClusters() {
  const params = new URLSearchParams({
    k: document.getElementById("clusterK").value,
    max_k: "8",
  });
  const data = await api(`/api/clustering?${params.toString()}`);
  renderClusterProfileTable(data);
  renderClusterLoadingsTable(data.loadings || []);

  Plotly.newPlot(
    "elbowChart",
    [
      {
        x: data.diagnostics.map((d) => d.k),
        y: data.diagnostics.map((d) => d.inertia),
        mode: "lines+markers",
        name: "Inercia",
      },
      {
        x: data.diagnostics.map((d) => d.k),
        y: data.diagnostics.map((d) => d.silhouette),
        mode: "lines+markers",
        yaxis: "y2",
        name: "Silhouette",
      },
    ],
    {
      title: "Método del codo + silhouette",
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
      yaxis2: { overlaying: "y", side: "right" },
    },
    { responsive: true }
  );

  const brandClusterPalette = [
    "#0F3F73", // brand blue
    "#15A35B", // brand green
    "#7D3ECF", // purple accent
    "#0F874B", // dark green
    "#2D6FA3", // medium blue
    "#26B36E", // vivid green
    "#1D4E89", // deep blue
    "#5B2CA5", // deep purple
  ];
  const clusterIds = [...new Set((data.pca || []).map((r) => String(r.cluster)))].sort((a, b) => Number(a) - Number(b));
  const pcaTraces = clusterIds.map((clusterId, idx) => {
    const rows = (data.pca || []).filter((r) => String(r.cluster) === clusterId);
    return {
      x: rows.map((r) => r.pca_1),
      y: rows.map((r) => r.pca_2),
      text: rows.map((r) => r.empresa),
      mode: "markers",
      type: "scatter",
      name: clusterLabelName(clusterId),
      marker: {
        color: brandClusterPalette[idx % brandClusterPalette.length],
        size: 12,
        opacity: 0.95,
        line: { color: "#ffffff", width: 1.2 },
      },
      hovertemplate: `<b>%{text}</b><br>${clusterLabelName(clusterId)}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>`,
    };
  });

  Plotly.newPlot(
    "pcaChart",
    pcaTraces,
    {
      title: "PCA 2D de clusters",
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
      showlegend: true,
      legend: { orientation: "h", y: 1.1, x: 0 },
    },
    { responsive: true }
  );

  const pca3dTraces = clusterIds.map((clusterId, idx) => {
    const rows = (data.pca || []).filter((r) => String(r.cluster) === clusterId);
    const safeNum = (v) => {
      const n = Number(v);
      return Number.isFinite(n) ? n : 0;
    };
    return {
      x: rows.map((r) => safeNum(r.pca_1)),
      y: rows.map((r) => safeNum(r.pca_2)),
      z: rows.map((r) => safeNum(r.pca_3)),
      text: rows.map((r) => r.empresa),
      mode: "markers",
      type: "scatter3d",
      name: clusterLabelName(clusterId),
      marker: {
        color: brandClusterPalette[idx % brandClusterPalette.length],
        size: 6,
        opacity: 0.95,
        line: { color: "#ffffff", width: 0.8 },
      },
      hovertemplate: `<b>%{text}</b><br>${clusterLabelName(clusterId)}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>`,
    };
  });

  const cubeFill = "rgba(220, 228, 240, 0.92)";

  Plotly.newPlot(
    "pca3dChart",
    pca3dTraces,
    {
      title: "PCA 3D de clusters",
      height: 620,
      paper_bgcolor: "rgba(255,255,255,0)",
      plot_bgcolor: "rgba(255,255,255,0)",
      font: { color: "#0b3558" },
      margin: { l: 35, r: 35, t: 95, b: 50 },
      showlegend: true,
      legend: { orientation: "h", y: 1.06, x: 0.02 },
      scene: {
        aspectmode: "auto",
        domain: { x: [0.02, 0.98], y: [0.02, 0.94] },
        xaxis: {
          title: "PC1",
          showbackground: true,
          backgroundcolor: cubeFill,
          gridcolor: "rgba(255,255,255,0.95)",
          zerolinecolor: "rgba(255,255,255,0.98)",
        },
        yaxis: {
          title: "PC2",
          showbackground: true,
          backgroundcolor: cubeFill,
          gridcolor: "rgba(255,255,255,0.95)",
          zerolinecolor: "rgba(255,255,255,0.98)",
        },
        zaxis: {
          title: "PC3",
          showbackground: true,
          backgroundcolor: cubeFill,
          gridcolor: "rgba(255,255,255,0.95)",
          zerolinecolor: "rgba(255,255,255,0.98)",
        },
        camera: {
          eye: { x: 1.45, y: 1.4, z: 0.72 },
        },
      },
    },
    { responsive: true }
  );
}

async function loadLeads() {
  const params = new URLSearchParams({ top_n: document.getElementById("topLeads").value });
  const data = await api(`/api/leads?${params.toString()}`);
  state.leadsData = data.data;
  renderTable(
    "leadsTable",
    data.data.map((r) => ({
      empresa: r.empresa,
      segmento: r.segmento,
      score: r.score,
      ruta: r.ruta_principal,
      drivers: r.drivers_score,
      pitch: r.pitch_comercial_sugerido,
    }))
  );
}

async function loadCloseProbabilityTable() {
  try {
    const source = document.getElementById("closeProbSource")?.value || "real";
    const includeSimulated = source === "all" ? "true" : "false";
    const topN = state.closeProbExpanded ? 200 : 10;
    const cacheBust = Date.now();
    const data = await api(
      `/api/close-probability-model?include_simulated=${includeSimulated}&top_n=${topN}&_ts=${cacheBust}`
    );
    const rows = (data.data || []).map((r) => ({
      empresa: r.empresa,
      reporte_esg: r.reporte_esg,
      cluster_label: clusterLabelName(r.cluster_label),
      frecuencia_exportacion: r.frecuencia_exportacion_dias,
      mas_40_viajes_mensuales: r.mas_40_viajes_mensuales,
      ventas_anuales_musd: r.ventas_anuales_musd,
      probabilidad_cierre_pct: Number(r.probabilidad_cierre_modelo || 0) * 100,
    }));
    renderTable("closeProbTable", rows);

    const fi = data.feature_importance || {};
    const fiRows = Object.entries(fi)
      .filter(([, v]) => Number.isFinite(Number(v)))
      .map(([feature, importance]) => ({ feature, importance: Number(importance) }))
      .sort((a, b) => b.importance - a.importance);

    Plotly.newPlot(
      "closeProbFeatureImportanceChart",
      [
      {
        x: fiRows.map((r) => r.feature),
        y: fiRows.map((r) => r.importance),
        type: "bar",
        marker: { color: "#0f3f73" },
          hovertemplate: "<b>%{x}</b><br>Importancia: %{y:.4f}<extra></extra>",
        },
      ],
      {
        title: "Feature Importance del Modelo",
        paper_bgcolor: "rgba(255,255,255,0)",
        plot_bgcolor: "rgba(255,255,255,0)",
        font: { color: "#0b3558" },
        margin: { l: 50, r: 20, t: 60, b: 90 },
        xaxis: { tickangle: -25, automargin: true },
        yaxis: { title: "Importancia" },
      },
      { responsive: true }
    );

    const clusterRows = (data.cluster_probabilities || [])
      .map((r) => ({
        cluster: Number(r.cluster),
        cluster_name: clusterLabelName(r.cluster),
        prob: Number(r.probabilidad_media_cierre || 0) * 100,
      }))
      .sort((a, b) => a.cluster - b.cluster);

    Plotly.newPlot(
      "closeProbClusterProbChart",
      [
        {
          x: clusterRows.map((r) => r.prob),
          y: clusterRows.map((r) => r.cluster_name),
          type: "bar",
          orientation: "h",
          marker: { color: "#15a35b" },
          hovertemplate: "<b>%{y}</b><br>Prob. cierre media: %{x:.2f}%<extra></extra>",
        },
      ],
      {
        title: "Probabilidad de Cierre por Cluster",
        paper_bgcolor: "rgba(255,255,255,0)",
        plot_bgcolor: "rgba(255,255,255,0)",
        font: { color: "#0b3558" },
        margin: { l: 110, r: 20, t: 60, b: 50 },
        xaxis: { title: "Probabilidad media (%)", range: [0, 100] },
        yaxis: { autorange: "reversed" },
      },
      { responsive: true }
    );

    const m = data?.model?.metrics || {};
    const cm = data?.model?.class_metrics || data?.class_metrics || {};
    const asPct = (v) => (Number.isFinite(Number(v)) ? `${(Number(v) * 100).toFixed(2)}%` : "N/A");
    document.getElementById("closeProbAccuracyCard").innerHTML = `<strong>Accuracy</strong><div>${asPct(m.accuracy)}</div>`;
    document.getElementById("closeProbPrecisionCard").innerHTML = `<strong>Precision</strong><div>${asPct(
      m.precision
    )}</div>`;
    document.getElementById("closeProbRecallCard").innerHTML = `<strong>Recall</strong><div>${asPct(m.recall)}</div>`;
    document.getElementById("closeProbF1Card").innerHTML = `<strong>F1 Score</strong><div>${asPct(m.f1)}</div>`;
    document.getElementById("closeProbRocAucCard").innerHTML = `<strong>ROC AUC</strong><div>${asPct(m.roc_auc)}</div>`;

    const numOrNull = (v) => {
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    };
    const classRows = ["0", "1"].map((label) => ({
      clase: label,
      precision: numOrNull(cm?.[label]?.precision) !== null ? numOrNull(cm?.[label]?.precision) * 100 : null,
      recall: numOrNull(cm?.[label]?.recall) !== null ? numOrNull(cm?.[label]?.recall) * 100 : null,
      f1: numOrNull(cm?.[label]?.f1) !== null ? numOrNull(cm?.[label]?.f1) * 100 : null,
      soporte: numOrNull(cm?.[label]?.support),
    }));
    renderTable("closeProbClassMetricsTable", classRows);

    const moreBtn = document.getElementById("closeProbMoreBtn");
    if (moreBtn) moreBtn.textContent = state.closeProbExpanded ? "Ver Menos" : "Ver Más";
  } catch (err) {
    const table = document.getElementById("closeProbTable");
    const detail = err?.message ? String(err.message) : "Error desconocido";
    if (table) table.innerHTML = `<tr><td>Error al cargar modelo de cierre: ${detail}</td></tr>`;
    console.error(err);
  }
}

function downloadLeadsCSV() {
  if (!state.leadsData.length) return;
  const keys = Object.keys(state.leadsData[0]);
  const csv = [keys.join(",")]
    .concat(state.leadsData.map((r) => keys.map((k) => `"${String(r[k] ?? "").replace(/"/g, '""')}"`).join(",")))
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "leads_priorizados_cl_circular.csv";
  link.click();
  URL.revokeObjectURL(url);
}

async function fullRefresh() {
  await loadMarket();
  await loadRanking();
  await loadMap();
  await loadForecast();
  await loadScenarios();
  await loadClusters();
  await loadLeads();
  await loadCloseProbabilityTable();
}

function wireEvents() {
  document.getElementById("segmentSelect").addEventListener("change", (e) => (state.segment = e.target.value));
  document.getElementById("routeSelect").addEventListener("change", (e) => (state.route = e.target.value));
  document.getElementById("yearSelect").addEventListener("change", (e) => (state.year = e.target.value));

  document.getElementById("refreshBtn").addEventListener("click", async () => {
    await loadMarket();
    await loadMap();
  });
  document.getElementById("discoverBtn").addEventListener("click", () => {
    const target = document.querySelector("main.layout");
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  document.getElementById("goMapBtn").addEventListener("click", () => {
    openTab("map");
    const target = document.getElementById("tab-map");
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  document.getElementById("toggleProductDetailBtn").addEventListener("click", () => {
    const wrap = document.getElementById("marketDetailWrap");
    const btn = document.getElementById("toggleProductDetailBtn");
    const nowHidden = wrap.classList.toggle("is-hidden");
    btn.textContent = nowHidden ? "Ver Detalle" : "Ocultar Detalle";
  });
  document.getElementById("rankingBtn").addEventListener("click", loadRanking);
  document.getElementById("forecastBtn").addEventListener("click", loadForecast);
  document.getElementById("toggleForecastDetailBtn").addEventListener("click", () => {
    const wrap = document.getElementById("forecastDetailWrap");
    const btn = document.getElementById("toggleForecastDetailBtn");
    const nowHidden = wrap.classList.toggle("is-hidden");
    btn.textContent = nowHidden ? "Ver Detalle" : "Ocultar Detalle";
  });
  document.getElementById("scenarioBtn").addEventListener("click", loadScenarios);
  document.getElementById("clusterBtn").addEventListener("click", loadClusters);
  document.getElementById("toggleLoadingsBtn").addEventListener("click", () => {
    const wrap = document.getElementById("clusterLoadingsWrap");
    const btn = document.getElementById("toggleLoadingsBtn");
    const nowHidden = wrap.classList.toggle("is-hidden");
    btn.textContent = nowHidden ? "Ver Detalle" : "Ocultar Detalle";
  });
  document.getElementById("clusterK").addEventListener("input", syncClusterKValue);
  syncClusterKValue();
  document.getElementById("leadsBtn").addEventListener("click", loadLeads);
  document.getElementById("downloadLeads").addEventListener("click", downloadLeadsCSV);
  document.getElementById("closeProbBtn").addEventListener("click", loadCloseProbabilityTable);
  document.getElementById("closeProbMoreBtn").addEventListener("click", async () => {
    state.closeProbExpanded = !state.closeProbExpanded;
    await loadCloseProbabilityTable();
  });
}

async function init() {
  setupTabs();
  await loadFilters();
  wireEvents();
  await fullRefresh();
}

init().catch((err) => {
  console.error(err);
  alert("Error cargando la aplicacion. Revisa la consola.");
});



