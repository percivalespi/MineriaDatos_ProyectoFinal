(() => {
  "use strict";

  const EARTH_RADIUS_KM = 6371.0;

  const state = {
    metadata: null,
    activeSection: 1,
    lastEta: null,
    form: {
      ciudad_codigo: "HYD",
      city: "Metropolitan",
      use_coords: false,
      dist_haversine_km: 11.2,
      rest_lat: 17.385,
      rest_lon: 78.486,
      del_lat: 17.456,
      del_lon: 78.557,
      time_orderd: "19:45",
      tiempo_preparacion_min: 10,
      road_traffic_density: "Medium",
      weather_conditions: "Cloudy",
      festival: "No",
      delivery_person_age: 29,
      delivery_person_ratings: 4.7,
      vehicle_condition: 1,
      multiple_deliveries: 1,
      type_of_vehicle: "scooter",
      type_of_order: "Meal",
    },
  };

  const els = {
    form: document.getElementById("etaForm"),
    ciudadSelect: document.getElementById("ciudadSelect"),
    cityTypeSelect: document.getElementById("cityTypeSelect"),
    useCoordsToggle: document.getElementById("useCoordsToggle"),
    distanceBlock: document.getElementById("distanceBlock"),
    coordsBlock: document.getElementById("coordsBlock"),
    distRange: document.getElementById("distRange"),
    distValueBadge: document.getElementById("distValueBadge"),
    restLatInput: document.getElementById("restLatInput"),
    restLonInput: document.getElementById("restLonInput"),
    delLatInput: document.getElementById("delLatInput"),
    delLonInput: document.getElementById("delLonInput"),
    timeInput: document.getElementById("timeInput"),
    prepRange: document.getElementById("prepRange"),
    prepValueBadge: document.getElementById("prepValueBadge"),
    trafficSeg: document.getElementById("trafficSeg"),
    weatherSelect: document.getElementById("weatherSelect"),
    festivalSelect: document.getElementById("festivalSelect"),
    ageRange: document.getElementById("ageRange"),
    ageValueBadge: document.getElementById("ageValueBadge"),
    ratingRange: document.getElementById("ratingRange"),
    ratingValueBadge: document.getElementById("ratingValueBadge"),
    vehCondSeg: document.getElementById("vehCondSeg"),
    multiSeg: document.getElementById("multiSeg"),
    vehicleTypeSelect: document.getElementById("vehicleTypeSelect"),
    orderTypeSelect: document.getElementById("orderTypeSelect"),
    summarySec1: document.getElementById("summarySec1"),
    summarySec2: document.getElementById("summarySec2"),
    summarySec3: document.getElementById("summarySec3"),
    formErrorBanner: document.getElementById("formErrorBanner"),
    calcButton: document.getElementById("calcButton"),
    ctaSpinner: document.getElementById("ctaSpinner"),
    ctaIcon: document.getElementById("ctaIcon"),
    ctaText: document.getElementById("ctaText"),
    heroCard: document.getElementById("heroCard"),
    etaHeroMain: document.getElementById("etaHeroMain"),
    deliveryLoader: document.getElementById("deliveryLoader"),
    etaValue: document.getElementById("etaValue"),
    narrativeText: document.getElementById("narrativeText"),
    slaValue: document.getElementById("slaValue"),
    bufferBadge: document.getElementById("bufferBadge"),
    rangeText: document.getElementById("rangeText"),
    distKmValue: document.getElementById("distKmValue"),
    profileTitle: document.getElementById("profileTitle"),
    profileDesc: document.getElementById("profileDesc"),
    guidanceBlock: document.getElementById("guidanceBlock"),
    alertsList: document.getElementById("alertsList"),
  };

  function refreshIcons() {
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }
  }

  function populateSelect(selectEl, options, selectedValue) {
    selectEl.innerHTML = "";
    options.forEach((opt) => {
      const optionEl = document.createElement("option");
      optionEl.value = opt.value;
      optionEl.textContent = opt.label;
      if (String(opt.value) === String(selectedValue)) {
        optionEl.selected = true;
      }
      selectEl.appendChild(optionEl);
    });
  }

  function renderSegmentedRadios(container, groupName, items, activeValue, onSelect) {
    container.innerHTML = "";
    items.forEach((item, index) => {
      const wrapper = document.createElement("div");
      wrapper.className = "seg-option";

      const radioId = `${groupName}_${index}`;
      const input = document.createElement("input");
      input.type = "radio";
      input.name = groupName;
      input.id = radioId;
      input.value = String(item.value);
      input.checked = String(item.value) === String(activeValue);
      input.addEventListener("change", () => {
        if (input.checked) {
          onSelect(item.value);
          updateSectionSummaries();
        }
      });

      const label = document.createElement("label");
      label.setAttribute("for", radioId);
      label.textContent = item.shortLabel;
      label.title = item.fullLabel;

      wrapper.appendChild(input);
      wrapper.appendChild(label);
      container.appendChild(wrapper);
    });
  }

  function styleRangeTrack(inputEl) {
    const min = parseFloat(inputEl.min || "0");
    const max = parseFloat(inputEl.max || "100");
    const val = parseFloat(inputEl.value || "0");
    const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
    inputEl.style.background = `linear-gradient(to right, #5F772E 0%, #5F772E ${pct}%, #E1E6D8 ${pct}%, #E1E6D8 100%)`;
  }

  function computeGeoDestination(cityCode, distKm) {
    const centroids = state.metadata?.city_centroids || {};
    const [lat1Deg, lon1Deg] = centroids[cityCode] || [19.07, 72.87];
    const lat1 = (lat1Deg * Math.PI) / 180;
    const lon1 = (lon1Deg * Math.PI) / 180;
    const delta = Math.max(0.1, distKm) / EARTH_RADIUS_KM;
    const bearing = (45 * Math.PI) / 180;

    const sinLat2 = Math.sin(lat1) * Math.cos(delta) + Math.cos(lat1) * Math.sin(delta) * Math.cos(bearing);
    const lat2 = Math.asin(Math.max(-1, Math.min(1, sinLat2)));
    const lon2 =
      lon1 +
      Math.atan2(
        Math.sin(bearing) * Math.sin(delta) * Math.cos(lat1),
        Math.cos(delta) - Math.sin(lat1) * Math.sin(lat2)
      );

    return {
      rest_lat: Number(lat1Deg.toFixed(5)),
      rest_lon: Number(lon1Deg.toFixed(5)),
      del_lat: Number(((lat2 * 180) / Math.PI).toFixed(5)),
      del_lon: Number(((lon2 * 180) / Math.PI).toFixed(5)),
    };
  }

  function syncCoordInputs() {
    els.restLatInput.value = Number(state.form.rest_lat).toFixed(5);
    els.restLonInput.value = Number(state.form.rest_lon).toFixed(5);
    els.delLatInput.value = Number(state.form.del_lat).toFixed(5);
    els.delLonInput.value = Number(state.form.del_lon).toFixed(5);
  }

  function openSection(sectionId) {
    state.activeSection = Number(sectionId);
    document.querySelectorAll(".acc-item").forEach((item) => {
      const target = Number(item.dataset.section);
      const isOpen = target === state.activeSection;
      item.classList.toggle("is-open", isOpen);
      const trigger = item.querySelector(".acc-trigger");
      if (trigger) {
        trigger.setAttribute("aria-expanded", String(isOpen));
      }
    });
  }

  function updateSectionSummaries() {
    const cityOpt = state.metadata?.cities.find((c) => c.value === state.form.ciudad_codigo);
    const cityText = cityOpt ? cityOpt.label : state.form.ciudad_codigo;
    const routeText = state.form.use_coords
      ? "Coordenadas (lat/lon)"
      : `${Number(state.form.dist_haversine_km).toFixed(1)} km`;
    els.summarySec1.textContent = `${cityText}, ${routeText}`;

    const trafficShort = {
      Low: "Fluido (Low)",
      Medium: "Moderado (Medium)",
      High: "Alto (High)",
      Jam: "Embotellamiento (Jam)",
    }[state.form.road_traffic_density] || state.form.road_traffic_density;
    els.summarySec2.textContent = `${state.form.time_orderd}, ${trafficShort}`;

    const multiTxt = state.form.multiple_deliveries === 0 ? "Directo" : `${state.form.multiple_deliveries} agrup.`;
    els.summarySec3.textContent = `${Number(state.form.delivery_person_ratings).toFixed(1)} / 5.0, ${multiTxt}`;
  }

  function animateValue(element, start, end, suffixHtml, decimals = 1) {
    const from = Number.isFinite(start) ? start : end;
    const duration = 240;
    const t0 = performance.now();
    function step(now) {
      const p = Math.min(1, (now - t0) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      const current = from + (end - from) * eased;
      element.innerHTML = `${current.toFixed(decimals)} ${suffixHtml}`;
      if (p < 1) {
        requestAnimationFrame(step);
      }
    }
    requestAnimationFrame(step);
  }

  function renderPrediction(data) {
    const prevEta = state.lastEta;
    state.lastEta = data.eta_min;

    animateValue(els.etaValue, prevEta, data.eta_min, '<span class="eta-unit">min</span>', 1);
    els.narrativeText.textContent = data.resumen_narrativo;

    els.slaValue.textContent = `${data.sla_min.toFixed(1)} min`;
    els.bufferBadge.textContent = `+${data.buffer_sla_min.toFixed(1)} min Margen de Seguridad (Buffer SLA)`;
    els.rangeText.textContent = `Rango esperado: ${data.rango_min.toFixed(1)} a ${data.rango_max.toFixed(1)} Minutos (min)`;

    els.distKmValue.textContent = `${data.dist_km.toFixed(2)} Kilómetros (km)`;

    els.profileTitle.textContent = data.perfil_titulo;
    els.profileDesc.textContent = data.perfil_descripcion;

    const hasAlerts = data.recomendaciones.some((rec) => rec.es_alerta);
    els.guidanceBlock?.classList.toggle("has-alerts", hasAlerts);

    els.alertsList.innerHTML = "";
    data.recomendaciones.forEach((rec) => {
      const row = document.createElement("div");
      row.className = "guidance-item" + (rec.es_alerta ? " is-alert" : "");

      const icon = document.createElement("i");
      icon.setAttribute("data-lucide", rec.es_alerta ? "alert-triangle" : "check-circle-2");
      icon.className = "icon-sm guidance-icon";

      const span = document.createElement("span");
      span.textContent = rec.texto;

      row.appendChild(icon);
      row.appendChild(span);
      els.alertsList.appendChild(row);
    });

    els.heroCard.classList.remove("flash-update");
    void els.heroCard.offsetWidth;
    els.heroCard.classList.add("flash-update");
    refreshIcons();
  }

  function showError(msg) {
    if (!msg) {
      els.formErrorBanner.classList.add("hidden");
      els.formErrorBanner.textContent = "";
      return;
    }
    els.formErrorBanner.textContent = msg;
    els.formErrorBanner.classList.remove("hidden");
  }

  function waitMs(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function submitPrediction(withDeliveryLoader = true) {
    showError("");
    els.calcButton.disabled = true;
    els.ctaSpinner.classList.remove("hidden");
    els.ctaIcon?.classList.add("hidden");
    els.ctaText.textContent = "Calculando estimación...";

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (withDeliveryLoader) {
      els.etaHeroMain?.classList.add("hidden");
      els.deliveryLoader?.classList.remove("hidden");
      if (!reducedMotion && els.deliveryLoader) {
        els.deliveryLoader.querySelectorAll(".loader-rider, .loader-trail").forEach((el) => {
          el.style.animation = "none";
          void el.offsetWidth;
          el.style.animation = "";
        });
      }
    }

    try {
      const fetchPromise = fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.form),
      });
      const minAnimDelay = withDeliveryLoader ? waitMs(reducedMotion ? 120 : 760) : Promise.resolve();
      const [response] = await Promise.all([fetchPromise, minAnimDelay]);

      const payload = await response.json();
      if (!response.ok) {
        const detail = payload?.detail?.[0]?.msg || payload?.detail || "Error al calcular el tiempo estimado.";
        throw new Error(detail);
      }
      renderPrediction(payload);
    } catch (err) {
      showError(err instanceof Error ? err.message : "No fue posible conectar con el servicio de estimación.");
    } finally {
      if (withDeliveryLoader) {
        els.deliveryLoader?.classList.add("hidden");
        els.etaHeroMain?.classList.remove("hidden");
      }
      els.calcButton.disabled = false;
      els.ctaSpinner.classList.add("hidden");
      els.ctaIcon?.classList.remove("hidden");
      els.ctaText.textContent = "Calcular Tiempo Estimado de Llegada (ETA)";
    }
  }

  function wireEvents() {
    document.querySelectorAll("[data-acc-target]").forEach((btn) => {
      btn.addEventListener("click", () => {
        openSection(btn.dataset.accTarget);
      });
    });

    els.ciudadSelect.addEventListener("change", (e) => {
      state.form.ciudad_codigo = e.target.value;
      const geo = computeGeoDestination(state.form.ciudad_codigo, state.form.dist_haversine_km);
      Object.assign(state.form, geo);
      syncCoordInputs();
      updateSectionSummaries();
    });

    els.cityTypeSelect.addEventListener("change", (e) => {
      state.form.city = e.target.value;
    });

    els.useCoordsToggle.addEventListener("change", (e) => {
      state.form.use_coords = e.target.checked;
      els.distanceBlock.classList.toggle("hidden", state.form.use_coords);
      els.coordsBlock.classList.toggle("hidden", !state.form.use_coords);
      updateSectionSummaries();
    });

    els.distRange.addEventListener("input", (e) => {
      state.form.dist_haversine_km = parseFloat(e.target.value);
      els.distValueBadge.textContent = `${state.form.dist_haversine_km.toFixed(1)} km`;
      styleRangeTrack(els.distRange);
      updateSectionSummaries();
    });

    [
      [els.restLatInput, "rest_lat"],
      [els.restLonInput, "rest_lon"],
      [els.delLatInput, "del_lat"],
      [els.delLonInput, "del_lon"],
    ].forEach(([inputEl, key]) => {
      inputEl.addEventListener("input", (e) => {
        const parsed = parseFloat(e.target.value);
        if (Number.isFinite(parsed)) {
          state.form[key] = parsed;
        }
      });
    });

    els.timeInput.addEventListener("change", (e) => {
      state.form.time_orderd = e.target.value || "19:45";
      updateSectionSummaries();
    });

    els.prepRange.addEventListener("input", (e) => {
      state.form.tiempo_preparacion_min = parseFloat(e.target.value);
      els.prepValueBadge.textContent = `${state.form.tiempo_preparacion_min.toFixed(0)} min`;
      styleRangeTrack(els.prepRange);
    });

    els.weatherSelect.addEventListener("change", (e) => {
      state.form.weather_conditions = e.target.value;
    });

    els.festivalSelect.addEventListener("change", (e) => {
      state.form.festival = e.target.value;
    });

    els.ageRange.addEventListener("input", (e) => {
      state.form.delivery_person_age = parseInt(e.target.value, 10);
      els.ageValueBadge.textContent = `${state.form.delivery_person_age} años`;
      styleRangeTrack(els.ageRange);
    });

    els.ratingRange.addEventListener("input", (e) => {
      state.form.delivery_person_ratings = parseFloat(e.target.value);
      els.ratingValueBadge.textContent = `${state.form.delivery_person_ratings.toFixed(1)} / 5.0`;
      styleRangeTrack(els.ratingRange);
      updateSectionSummaries();
    });

    els.vehicleTypeSelect.addEventListener("change", (e) => {
      state.form.type_of_vehicle = e.target.value;
    });

    els.orderTypeSelect.addEventListener("change", (e) => {
      state.form.type_of_order = e.target.value;
    });

    els.form.addEventListener("submit", (e) => {
      e.preventDefault();
      submitPrediction();
    });
  }

  async function init() {
    refreshIcons();
    try {
      const resp = await fetch("/api/metadata");
      if (!resp.ok) {
        throw new Error("No se pudieron cargar los catálogos iniciales.");
      }
      const meta = await resp.json();
      state.metadata = meta;
      Object.assign(state.form, meta.defaults);

      populateSelect(els.ciudadSelect, meta.cities, state.form.ciudad_codigo);
      populateSelect(els.cityTypeSelect, meta.city_types, state.form.city);
      populateSelect(els.weatherSelect, meta.weather_conditions, state.form.weather_conditions);
      populateSelect(els.festivalSelect, meta.festival_options, state.form.festival);
      populateSelect(els.vehicleTypeSelect, meta.vehicle_types, state.form.type_of_vehicle);
      populateSelect(els.orderTypeSelect, meta.order_types, state.form.type_of_order);

      els.useCoordsToggle.checked = Boolean(state.form.use_coords);
      els.distRange.value = state.form.dist_haversine_km;
      els.distValueBadge.textContent = `${Number(state.form.dist_haversine_km).toFixed(1)} km`;
      syncCoordInputs();

      els.timeInput.value = state.form.time_orderd;
      els.prepRange.value = state.form.tiempo_preparacion_min;
      els.prepValueBadge.textContent = `${Number(state.form.tiempo_preparacion_min).toFixed(0)} min`;

      els.ageRange.value = state.form.delivery_person_age;
      els.ageValueBadge.textContent = `${state.form.delivery_person_age} años`;

      els.ratingRange.value = state.form.delivery_person_ratings;
      els.ratingValueBadge.textContent = `${Number(state.form.delivery_person_ratings).toFixed(1)} / 5.0`;

      [els.distRange, els.prepRange, els.ageRange, els.ratingRange].forEach(styleRangeTrack);

      const trafficShortMap = {
        Low: "Fluido (Low)",
        Medium: "Moderado (Medium)",
        High: "Alto (High)",
        Jam: "Saturado (Jam)",
      };
      renderSegmentedRadios(
        els.trafficSeg,
        "road_traffic_density",
        meta.traffic_levels.map((t) => ({
          value: t.value,
          shortLabel: trafficShortMap[t.value] || t.label,
          fullLabel: t.label,
        })),
        state.form.road_traffic_density,
        (val) => {
          state.form.road_traffic_density = val;
        }
      );

      renderSegmentedRadios(
        els.vehCondSeg,
        "vehicle_condition",
        [
          { value: 0, shortLabel: "0 Básico", fullLabel: "Estado 0 (Básico)" },
          { value: 1, shortLabel: "1 Regular", fullLabel: "Estado 1 (Regular)" },
          { value: 2, shortLabel: "2 Bueno", fullLabel: "Estado 2 (Bueno)" },
          { value: 3, shortLabel: "3 Óptimo", fullLabel: "Estado 3 (Óptimo)" },
        ],
        state.form.vehicle_condition,
        (val) => {
          state.form.vehicle_condition = Number(val);
        }
      );

      renderSegmentedRadios(
        els.multiSeg,
        "multiple_deliveries",
        [
          { value: 0, shortLabel: "0 Directo", fullLabel: "0 entregas adicionales" },
          { value: 1, shortLabel: "1 extra", fullLabel: "1 entrega agrupada" },
          { value: 2, shortLabel: "2 extras", fullLabel: "2 entregas agrupadas" },
          { value: 3, shortLabel: "3 extras", fullLabel: "3 entregas agrupadas" },
        ],
        state.form.multiple_deliveries,
        (val) => {
          state.form.multiple_deliveries = Number(val);
        }
      );

      updateSectionSummaries();
      wireEvents();
      await submitPrediction(false);
    } catch (err) {
      showError(err instanceof Error ? err.message : "Error inicializando la aplicación.");
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
