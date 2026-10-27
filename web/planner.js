// Solar panel planner: lay out modules on a rooftop or a drawn patch of open land, drag
// them around, and estimate their output with SolarSiteService (physics only - no equity data).
//
// Geometry runs here, in local metres around the surface, so dragging stays instant; the
// module size and spacing come from /site/panel so the count suggested here and the kW
// estimated by the service always agree. Uses globals from index.html: map, streets,
// satellite, layer (tract layer), esc, lastSearch.

const planner = (() => {
  const M_PER_DEG_LAT = 110574, M_PER_DEG_LON = 111320;
  const MAX_DRAWN = 6000;   // Leaflet draws this many on a canvas comfortably
  const COLORS = {panel: "#1d3557", edge: "#a8dadc", bad: "#b2182b", roof: "#ffd400", land: "#ffffff"};

  let spec = null, active = false;
  let surface = null;          // {pts: [[lat, lon]...], kind: "roof"|"ground", label}
  let surfaceLayer = null, panels = [], layout = [], theta = 0, origin = null;
  let drawing = null, estimateTimer = null, estimateAbort = null;
  const renderer = L.canvas({padding: 0.5});
  const fmtN = (v, d = 0) => Number(v).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});

  // ---- local metric frame ----------------------------------------------------------
  const toXY = ([lat, lon]) => [(lon - origin[1]) * Math.cos(origin[0] * Math.PI / 180) * M_PER_DEG_LON,
                                (lat - origin[0]) * M_PER_DEG_LAT];
  const toLL = ([x, y]) => [origin[0] + y / M_PER_DEG_LAT,
                            origin[1] + x / (Math.cos(origin[0] * Math.PI / 180) * M_PER_DEG_LON)];
  const rot = ([x, y], a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];

  function inside([x, y], poly) {
    let hit = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  }
  function edgeDist([x, y], poly) {
    let best = Infinity;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [ax, ay] = poly[j], [bx, by] = poly[i], dx = bx - ax, dy = by - ay;
      const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1)));
      best = Math.min(best, Math.hypot(x - ax - t * dx, y - ay - t * dy));
    }
    return best;
  }
  const area = poly => Math.abs(poly.reduce((s, [x, y], i) => {
    const [x2, y2] = poly[(i + 1) % poly.length]; return s + x * y2 - x2 * y; }, 0)) / 2;

  // ---- module footprint and spacing for the current mount ------------------------
  function dims() {
    const m = spec.mounts[mount()], p = spec.panel;
    const depth = p.length_m * Math.cos(m.tilt * Math.PI / 180);   // plan-view depth of a tilted module
    return {w: p.width_m, d: depth, setback: m.setback_m,
            colPitch: p.width_m + p.gap_m,
            rowPitch: m.gcr ? p.length_m / m.gcr : depth + p.gap_m};
  }
  const mount = () => document.getElementById("pl-mount").value;

  function corners(cxy, a) {
    const {w, d} = dims();
    return [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]]
      .map(c => rot(c, a)).map(([x, y]) => [x + cxy[0], y + cxy[1]]);
  }

  // Every panel position that fits the surface, rows parallel to angle a. A few grid
  // offsets are tried and the fullest kept - the first offset alone wastes edge space.
  function computeLayout() {
    if (!surface) return [];
    const {w, d, setback, colPitch, rowPitch} = dims();
    const poly = surface.pts.map(toXY).map(p => rot(p, -theta));
    const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    // A pitched roof has two slopes either side of a ridge along its long axis, and only the
    // one facing nearer south is worth covering. Rows run parallel to the ridge, so in this
    // frame that slope is one side of a horizontal line. The ridge is the midpoint of the
    // roof's width - not the mean of its corners, which an outline with a jog drags off
    // centre - and a module must lie wholly on its slope, not merely be centred there.
    let slope = null;
    if (mount() === "roof") {
      const a = (90 - azimuth()) * Math.PI / 180;
      slope = {ridge: (minY + maxY) / 2, sign: Math.sign(rot([Math.cos(a), Math.sin(a)], -theta)[1]) || 1};
    }
    const fits = (x, y) => {
      if (slope && slope.sign * (y - slope.ridge) < d / 2) return false;
      const cs = [[x - w / 2, y - d / 2], [x + w / 2, y - d / 2], [x + w / 2, y + d / 2], [x - w / 2, y + d / 2]];
      if (!cs.every(c => inside(c, poly) && edgeDist(c, poly) >= setback)) return false;
      // a concave notch can poke between corners: no surface vertex may sit inside the module
      return !poly.some(([px, py]) => Math.abs(px - x) < w / 2 && Math.abs(py - y) < d / 2);
    };
    // Grid offsets are tried with the slope rule already applied, so the alignment chosen
    // is the one that fills the usable slope best, not the whole outline.
    let best = [];
    for (const fx of [0, 0.25, 0.5, 0.75]) for (const fy of [0, 0.2, 0.4, 0.6, 0.8]) {
      const cells = [];
      for (let y = minY + setback + d / 2 + fy * rowPitch; y <= maxY - setback - d / 2; y += rowPitch)
        for (let x = minX + setback + w / 2 + fx * colPitch; x <= maxX - setback - w / 2; x += colPitch)
          if (fits(x, y)) cells.push([x, y]);
      if (cells.length > best.length) best = cells;
    }
    return best.map(c => rot(c, theta));
  }

  // Flush panels follow a pitched roof's longest edge, the way installers lay them; racks on
  // a flat roof or open land run east-west so they face due south.
  function rowAngle(pts, kind) {
    if (kind !== "roof") return 0;
    const xy = pts.map(toXY);
    let best = 0, bestLen = 0;
    xy.forEach((p, i) => {
      const q = xy[(i + 1) % xy.length], len = Math.hypot(q[0] - p[0], q[1] - p[1]);
      if (len > bestLen) { bestLen = len; best = Math.atan2(q[1] - p[1], q[0] - p[0]); }
    });
    if (best > Math.PI / 2) best -= Math.PI;
    if (best < -Math.PI / 2) best += Math.PI;
    return best;
  }
  // Panels face whichever side of their rows is nearer south (compass 180).
  function azimuth() {
    if (mount() !== "roof") return 180;
    const compass = a => ((90 - a * 180 / Math.PI) % 360 + 360) % 360;
    const opts = [compass(theta + Math.PI / 2), compass(theta - Math.PI / 2)];
    return opts.reduce((b, a) => Math.abs(a - 180) < Math.abs(b - 180) ? a : b);
  }

  // ---- panels on the map -------------------------------------------------------------
  function addPanel(cxy, checked = false) {
    const lyr = L.polygon(corners(cxy, theta).map(toLL), {renderer, weight: 0.6, color: COLORS.edge,
      fillColor: COLORS.panel, fillOpacity: 0.92, bubblingMouseEvents: false}).addTo(map);
    const p = {c: cxy, lyr};
    lyr.on("mousedown", e => startDrag(p, e));
    lyr.on("contextmenu", e => { L.DomEvent.preventDefault(e.originalEvent); removePanel(p); });
    panels.push(p);
    if (checked) p.bad = false; else check(p);   // auto-layout cells are valid by construction
    return p;
  }
  function removePanel(p) { map.removeLayer(p.lyr); panels = panels.filter(q => q !== p); changed(); }
  function clearPanels() { panels.forEach(p => map.removeLayer(p.lyr)); panels = []; }

  // A dragged module that leaves the surface or overlaps another turns red: still counted
  // (it is a real panel the user placed) but flagged, and the summary says how many.
  function check(p) {
    let bad = false;
    if (surface) {
      const poly = surface.pts.map(toXY);
      bad = !corners(p.c, theta).every(c => inside(c, poly));
    }
    // Both panels of an overlapping pair are flagged, not just the one that moved.
    const {w, d} = dims();
    const hits = panels.filter(q => {
      if (q === p) return false;
      const [dx, dy] = rot([q.c[0] - p.c[0], q.c[1] - p.c[1]], -theta);
      return Math.abs(dx) < w - 0.01 && Math.abs(dy) < d - 0.01;
    });
    hits.forEach(q => paint(q, true));
    paint(p, bad || hits.length > 0);
  }
  function paint(p, bad) {
    p.bad = bad;
    p.lyr.setStyle({fillColor: bad ? COLORS.bad : COLORS.panel});
  }

  function startDrag(p, e) {
    L.DomEvent.stop(e);
    map.dragging.disable();
    const start = toXY([e.latlng.lat, e.latlng.lng]), c0 = p.c.slice();
    const move = ev => {
      const now = toXY([ev.latlng.lat, ev.latlng.lng]);
      p.c = [c0[0] + now[0] - start[0], c0[1] + now[1] - start[1]];
      p.lyr.setLatLngs(corners(p.c, theta).map(toLL));
    };
    map.on("mousemove", move);
    map.once("mouseup", () => {
      map.off("mousemove", move);
      map.dragging.enable();
      // Only the moved panel and ones already flagged can have changed state; re-checking
      // all of them is quadratic and stalls on a filled field of thousands.
      check(p);
      panels.filter(q => q.bad && q !== p).forEach(check);
      changed();
    });
  }

  // A partial array grows outward from the middle of the surface, so 200 panels on a field
  // read as one compact array rather than a thin strip along its southern edge.
  function place(n) {
    clearPanels();
    layout = computeLayout();
    const k = Math.min(n, layout.length, MAX_DRAWN);
    const mid = layout.reduce((s, c) => [s[0] + c[0] / layout.length, s[1] + c[1] / layout.length], [0, 0]);
    const cells = k < layout.length
      // square rings (Chebyshev distance), so a partial array is a rectangular block, not a disc
      ? layout.slice().sort((a, b) => Math.max(Math.abs(a[0] - mid[0]), Math.abs(a[1] - mid[1]))
                                    - Math.max(Math.abs(b[0] - mid[0]), Math.abs(b[1] - mid[1])))
      : layout;
    cells.slice(0, k).forEach(c => addPanel(c, true));
    if (k && k < layout.length)
      map.fitBounds(L.latLngBounds(panels.map(p => p.lyr.getBounds())), {maxZoom: 21, padding: [80, 80], animate: false});
    changed();
    return k;
  }

  // ---- surfaces: the searched rooftop, or a drawn patch of land -------------------
  // OpenStreetMap does not record roof shape, so guess it: houses are pitched, and large or
  // non-residential buildings are almost always flat. The mount menu overrides the guess.
  const PITCHED = new Set(["house", "detached", "semidetached_house", "terrace", "residential",
                           "bungalow", "cabin", "garage", "shed", "farm"]);
  function roofKind(tag, pts) {
    if (PITCHED.has(tag)) return "roof";
    return tag !== "yes" || area(pts.map(toXY)) > 250 ? "flat" : "roof";
  }

  function setSurface(pts, kind, label, tag) {
    if (surfaceLayer) map.removeLayer(surfaceLayer);
    clearPanels();
    origin = [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];
    const mountKind = kind === "ground" ? "ground" : roofKind(tag, pts);
    surface = {pts, kind, label};
    document.getElementById("pl-mount").value = mountKind;
    theta = rowAngle(pts, mountKind);
    // Same canvas as the panels and added before them, so panels draw on top of the outline.
    surfaceLayer = L.polygon(pts, {renderer, interactive: false, color: kind === "roof" ? COLORS.roof : COLORS.land,
      weight: 3, dashArray: kind === "roof" ? null : "6 5", fillOpacity: 0.08}).addTo(map);
    map.fitBounds(surfaceLayer.getBounds(), {maxZoom: 21, padding: [60, 60], animate: false});
    layout = computeLayout();
    renderSurface();
    loadPotential();
    changed();
  }

  function startDrawing() {
    stopDrawing();
    const pts = [], line = L.polyline([], {color: COLORS.land, weight: 2, dashArray: "6 5"}).addTo(map);
    const guide = L.polyline([], {color: COLORS.land, weight: 1, opacity: 0.7}).addTo(map);
    map.doubleClickZoom.disable();
    map.getContainer().style.cursor = "crosshair";
    const finish = () => {
      // the two clicks of a double-click each added a corner; drop repeats
      const done = pts.filter((p, i) => i === 0 || map.distance(p, pts[i - 1]) > 0.3);
      stopDrawing();
      if (done.length >= 3) setSurface(done, "ground", "Drawn area");
    };
    const click = e => {
      // Clicking near the first vertex closes the shape, the way drawing tools usually do.
      if (pts.length >= 3 && map.latLngToContainerPoint(pts[0]).distanceTo(e.containerPoint) < 12) return finish();
      pts.push([e.latlng.lat, e.latlng.lng]); line.setLatLngs(pts);
    };
    const hover = e => guide.setLatLngs(pts.length ? [pts[pts.length - 1], e.latlng] : []);
    map.on("click", click); map.on("mousemove", hover); map.on("dblclick", finish);
    drawing = {line, guide, click, hover, finish};
    setHint("Click to add corners of the land. Double-click, or click the first corner, to finish.");
  }
  function stopDrawing() {
    if (!drawing) return;
    map.off("click", drawing.click); map.off("mousemove", drawing.hover); map.off("dblclick", drawing.finish);
    map.removeLayer(drawing.line); map.removeLayer(drawing.guide);
    map.doubleClickZoom.enable();
    map.getContainer().style.cursor = "";
    drawing = null;
    setHint("");
  }

  // ---- side panel ----------------------------------------------------------------------
  const el = id => document.getElementById(id);
  const setHint = t => { el("pl-hint").textContent = t; el("pl-hint").hidden = !t; };

  function renderSurface() {
    if (!surface) { el("pl-surface").innerHTML = `<p class="empty">Search an address and use its rooftop, draw a patch
      of open land, or just drag panels onto the map.</p>`; el("pl-fit").innerHTML = ""; return; }
    const m2 = area(surface.pts.map(toXY));
    const kw = layout.length * spec.panel.watts / 1000;
    el("pl-surface").innerHTML = `<div class="pair"><dt>${esc(surface.label)}</dt><dd>${fmtN(m2)} m²</dd></div>
      <div class="pair"><dt>${surface.kind === "roof" ? "Rooftop" : "Open land"}</dt>
      <dd>${fmtN(m2 * 10.7639)} ft²${m2 > 4047 ? ` · ${fmtN(m2 / 4046.86, 1)} acres` : ""}</dd></div>`;
    el("pl-fit").innerHTML = layout.length
      ? `<b>Fits up to ${fmtN(layout.length)} panels</b> (${fmtN(kw, kw < 100 ? 1 : 0)} kW)
         <div class="sub">${{roof: "Flush on the roof slope facing nearer south (the other slope is left bare), 0.9 m from every edge for fire access.",
             flat: "Flat roof: low racks at 10° facing south, 1.2 m from the edges.",
             ground: "Racks at 30° facing south, rows spaced so they don't shade each other."}[mount()]}
         ${layout.length > MAX_DRAWN ? ` The map shows at most ${fmtN(MAX_DRAWN)}.` : ""}</div>`
      : `<b>No full panel fits</b><div class="sub">The surface is smaller than one module plus the edge setback.</div>`;
    el("pl-count").max = Math.min(layout.length || MAX_DRAWN, MAX_DRAWN);
  }

  function renderCount() {
    const bad = panels.filter(p => p.bad).length;
    el("pl-placed").innerHTML = `<b>${fmtN(panels.length)}</b> panel${panels.length === 1 ? "" : "s"} placed
      (${fmtN(panels.length * spec.panel.watts / 1000, 1)} kW)` +
      (bad ? `<div class="sub" style="color:var(--hi)">${bad} shown in red: off the surface or overlapping.</div>` : "");
  }

  async function loadPotential() {
    const [lat, lon] = origin;
    try {
      const r = await fetch(`../site/potential?lat=${lat}&lon=${lon}`);
      const x = await r.json();
      if (!r.ok) throw new Error(x.detail || r.status);
      el("pl-sun").innerHTML = `<div class="pair"><dt>Sunlight here</dt><dd>${x.ghi_kwh_m2_day} kWh/m²/day</dd></div>
        <div class="pair"><dt>1 kW of panels makes</dt><dd>${fmtN(x.specific_yield_kwh_kw[mount()])} kWh/yr</dd></div>`;
    } catch (e) { el("pl-sun").innerHTML = `<div class="sub">Sun data unavailable (${esc(e.message)}).</div>`; }
  }

  function changed() {
    renderCount();
    clearTimeout(estimateTimer);
    estimateTimer = setTimeout(estimate, 250);
  }

  async function estimate() {
    const out = el("pl-results");
    if (!panels.length) { out.innerHTML = `<p class="empty">Place panels to see how much electricity they make.</p>`; return; }
    if (estimateAbort) estimateAbort.abort();
    estimateAbort = new AbortController();
    // Estimate the layout at its own centre, so panels dragged onto open land with no drawn
    // surface still get the sunlight of where they actually are.
    const ll = panels.map(p => toLL(p.c));
    const q = new URLSearchParams({
      lat: (ll.reduce((s, p) => s + p[0], 0) / ll.length).toFixed(5),
      lon: (ll.reduce((s, p) => s + p[1], 0) / ll.length).toFixed(5),
      panels: panels.length, mount: mount(), azimuth: Math.round(azimuth()),
    });
    try {
      // ROIService prices the layout (endterm cost layer) and carries the energy figures with
      // it. Without the gateway or ROI running, fall back to energy alone from SolarSiteService.
      const money = new URLSearchParams({...Object.fromEntries(q), price_per_kwh: el("pl-price").value,
        cost_per_w: el("pl-cost").value, tax_credit: (Number(el("pl-credit").value) / 100).toString()});
      let r = await fetch(`../roi/layout?${money}`, {signal: estimateAbort.signal}).catch(() => null);
      if (!r || !r.ok) r = await fetch(`../site/estimate?${q}`, {signal: estimateAbort.signal});
      const x = await r.json();
      if (!r.ok) throw new Error(x.detail?.[0]?.msg || x.detail || r.status);
      const usd = v => "$" + fmtN(v);
      const cost = x.annual_savings == null
        ? `<h2>Cost and savings</h2><div class="sub">Unavailable here: open the dashboard through the
           gateway on port 8000 with ROIService running.</div>`
        : `<h2>Cost and savings</h2>
           <div class="big" style="color:var(--ink)">${usd(x.annual_savings)}<span style="font-size:13px;font-weight:500"> / year</span></div>
           <div class="sub">saved at ${(x.assumptions.price_per_kwh * 100).toFixed(2)}¢ per kWh</div>
           <dl style="margin-top:10px">
             ${[["Installed cost", usd(x.net_cost)],
                ["Simple payback", `${x.simple_payback_years} years`],
                [`Net savings over ${x.lifetime_years} years`, usd(x.lifetime_net_savings)]]
               .map(([k, v]) => `<div class="pair"><dt>${k}</dt><dd>${v}</dd></div>`).join("")}
           </dl>`;
      out.innerHTML = `
        <div class="big" style="color:var(--ink)">${fmtN(x.annual_kwh)}<span style="font-size:13px;font-weight:500"> kWh / year</span></div>
        <div class="sub">electricity generated by ${fmtN(x.panels)} panel${x.panels === 1 ? "" : "s"}</div>
        <dl style="margin-top:10px">
          ${[["System size", `${fmtN(x.system_kw, 1)} kW`],
             ["Each kW of panels makes", `${fmtN(x.specific_yield_kwh_kw)} kWh/yr`],
             ["Capacity factor", `${(x.capacity_factor * 100).toFixed(1)}%`],
             ["CO₂ avoided", `${fmtN(x.co2_avoided_kg_year / 1000, 1)} t/yr`],
             ["Facing", `${Math.round(x.azimuth)}° (${compassName(x.azimuth)}), ${x.tilt}° tilt`]]
            .map(([k, v]) => `<div class="pair"><dt>${k}</dt><dd>${v}</dd></div>`).join("")}
        </dl>
        ${cost}
        <h2>Monthly output (kWh)</h2>
        ${chart(x.monthly_kwh)}
        <p class="note">Sunlight: NASA POWER long-term average at this spot, converted with pvlib
          using NREL PVWatts losses. CO₂: EPA eGRID 2023, RFC East grid. No federal homeowner
          credit after 2025. Estimates, not a quote.</p>`;
    } catch (e) {
      if (e.name !== "AbortError") out.innerHTML = `<div class="sub" style="color:var(--hi)">Estimate unavailable: ${esc(e.message)}</div>`;
    }
  }
  const compassName = a => ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round(a / 45) % 8];

  // One series, so no legend: the heading names it. Columns grow from one baseline with
  // 4px rounded tops; only the peak month is labelled; every column has a hover target.
  function chart(monthly) {
    const vals = Object.values(monthly), names = "JFMAMJJASOND".split("");
    const full = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const W = 276, H = 128, top = 16, base = H - 18, band = W / 12, bw = 14, max = Math.max(...vals);
    const peak = vals.indexOf(max);
    const cols = vals.map((v, i) => {
      const h = Math.max(1, (v / max) * (base - top)), x = i * band + (band - bw) / 2, y = base - h, r = Math.min(4, h);
      const path = `M${x},${base} V${y + r} Q${x},${y} ${x + r},${y} H${x + bw - r} Q${x + bw},${y} ${x + bw},${y + r} V${base} Z`;
      return `<g class="col" data-tip="${full[i]}: ${fmtN(v)} kWh">
        <rect x="${i * band}" y="0" width="${band}" height="${H}" fill="transparent"/>
        <path d="${path}" fill="#2a78d6"/>
        <text x="${i * band + band / 2}" y="${H - 4}" text-anchor="middle" class="ax">${names[i]}</text>
        ${i === peak ? `<text x="${i * band + band / 2}" y="${y - 4}" text-anchor="middle" class="pk">${fmtN(v)}</text>` : ""}
      </g>`;
    }).join("");
    return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" width="100%" role="img"
      aria-label="Monthly output, peak ${full[peak]} at ${fmtN(max)} kWh">
      <line x1="0" x2="${W}" y1="${base}" y2="${base}" stroke="var(--line)"/>${cols}</svg>
      <div class="tip" hidden></div></div>`;
  }
  document.addEventListener("mouseover", e => {
    const g = e.target.closest?.(".chart .col"), tip = e.target.closest?.(".chart")?.querySelector(".tip");
    document.querySelectorAll(".chart .col.on").forEach(c => c.classList.remove("on"));
    if (!g || !tip) { document.querySelectorAll(".chart .tip").forEach(t => t.hidden = true); return; }
    g.classList.add("on");
    tip.textContent = g.dataset.tip; tip.hidden = false;
    const box = g.getBoundingClientRect(), host = tip.parentElement.getBoundingClientRect();
    tip.style.left = `${box.left - host.left + box.width / 2}px`;
  });

  // ---- entering and leaving -----------------------------------------------------------
  async function open(fromSearch) {
    if (!spec) spec = await (await fetch("../site/panel")).json();
    active = true;
    document.body.classList.add("planning");
    if (layer) map.removeLayer(layer);
    // the search's pin and roof highlight would sit on top of, or tint, the panels
    if (pin) map.removeLayer(pin);
    if (roof) map.removeLayer(roof);
    if (!map.hasLayer(satellite)) { map.removeLayer(streets); satellite.addTo(map); }
    if (fromSearch && lastSearch?.building) {
      const ring = lastSearch.building.geometry.coordinates[0].map(([lon, lat]) => [lat, lon]);
      setSurface(ring.slice(0, -1), "roof", lastSearch.location.matched_address.split(",")[0],
                 lastSearch.building.properties.building);
    } else {
      document.getElementById("pl-mount").value = "ground";   // no roof: assume open land
      renderSurface(); changed();
    }
  }
  function close() {
    stopDrawing(); clearPanels();
    armed = false; el("pl-tile").classList.remove("armed");
    if (surfaceLayer) map.removeLayer(surfaceLayer);
    surface = surfaceLayer = null; layout = []; origin = null;
    active = false;
    document.body.classList.remove("planning");
    if (layer) layer.addTo(map).bringToBack();
    if (pin) pin.addTo(map);
    if (roof) roof.addTo(map);
  }

  // One loose panel, dragged or clicked onto the map. With no surface yet (open land), the
  // first one sets the local frame, and the map zooms in: a 1 m panel is a speck below z20.
  function dropAt(ll) {
    if (!origin) { origin = [ll.lat, ll.lng]; loadPotential(); }
    addPanel(toXY([ll.lat, ll.lng]));
    if (!surface && map.getZoom() < 20) map.setView(ll, 20, {animate: false});
    changed();
  }

  // Drag a panel from the tray onto the map. With no surface yet (open land), the first
  // drop sets the local frame and rows run east-west.
  const mapEl = document.getElementById("map");
  mapEl.addEventListener("dragover", e => { if (active) e.preventDefault(); });
  mapEl.addEventListener("drop", e => {
    if (!active) return;
    e.preventDefault();
    dropAt(map.mouseEventToLatLng(e));
  });
  el("pl-tile").addEventListener("dragstart", e => e.dataTransfer.setData("text/plain", "panel"));
  // Click-to-place as well as drag: touchpads and touch screens make HTML drag-and-drop
  // unreliable, and a demo should not depend on it. Click the tile, then the map.
  let armed = false;
  el("pl-tile").addEventListener("click", () => {
    armed = !armed;
    el("pl-tile").classList.toggle("armed", armed);
    map.getContainer().style.cursor = armed ? "copy" : "";
    setHint(armed ? "Click the map to drop a panel. Click the panel tile again to stop." : "");
  });
  map.on("click", e => {
    if (!armed || drawing) return;
    dropAt(e.latlng);
  });

  el("pl-draw").addEventListener("click", startDrawing);
  el("pl-fill").addEventListener("click", () => {
    if (!surface) return setHint("Pick a rooftop or draw an area first, so there is something to fill.");
    const k = place(layout.length);
    el("pl-count").value = k;
  });
  el("pl-place").addEventListener("click", () => {
    const n = Math.max(1, Math.floor(Number(el("pl-count").value) || 0));
    if (!surface) return setHint("Pick a rooftop or draw an area first, or drag panels on yourself.");
    const k = place(n);
    setHint(k < n ? `Only ${fmtN(k)} fit on this surface; placed them all.` : "");
  });
  el("pl-clear").addEventListener("click", () => { clearPanels(); changed(); });
  el("pl-mount").addEventListener("change", () => {
    if (surface) {
      theta = rowAngle(surface.pts, mount());
      layout = computeLayout(); renderSurface();
      const n = panels.length; if (n) place(n);
    }
    loadPotentialIfAny(); changed();
  });
  const loadPotentialIfAny = () => { if (origin) loadPotential(); };
  ["pl-price", "pl-cost", "pl-credit"].forEach(id => el(id).addEventListener("input", changed));
  el("pl-exit").addEventListener("click", close);

  return {open, close, get active() { return active; }};
})();
