// Leaderboard: live filter and sort, compare chips, styled tooltips, community refresh.
// The server renders every state already (GET form, chip buttons, ?compare=); this only enhances.
(function () {
  "use strict";
  var root = document.querySelector("[data-lb]");
  if (!root) return;
  var briefs = Number(root.getAttribute("data-briefs")) || 0;  // the frozen suite's brief count

  // ------------------------------------------------------------ tooltips
  var tip = document.createElement("div");
  tip.className = "lb-tip";
  tip.setAttribute("role", "tooltip");
  tip.id = "lb-tip";
  tip.hidden = true;
  document.body.appendChild(tip);

  function adopt(el) {
    if (el.hasAttribute("data-tip")) return;
    var text = "";
    if (el.hasAttribute("title")) {
      text = el.getAttribute("title");
      el.removeAttribute("title");
    } else {
      var t = el.querySelector(":scope > title");
      if (t) { text = t.textContent; if (!el.getAttribute("aria-label")) el.setAttribute("aria-label", text); t.remove(); }
    }
    if (text) el.setAttribute("data-tip", text);
  }
  root.querySelectorAll("[title], svg circle, svg rect").forEach(adopt);

  function place(x, y) {
    var pad = 12, w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(Math.max(pad, x + 14), window.innerWidth - w - pad);
    var top = y - h - 12;
    if (top < pad) top = y + 18;
    tip.style.left = left + "px";
    tip.style.top = Math.min(top, window.innerHeight - h - pad) + "px";
  }
  var anchorEl = null;
  function show(el, x, y) {
    anchorEl = el;
    tip.textContent = el.getAttribute("data-tip");
    tip.hidden = false;
    place(x, y);
  }
  function showAt(el) {
    var r = el.getBoundingClientRect();
    show(el, r.left + r.width / 2 - 14, r.top);
  }
  function hide() { tip.hidden = true; anchorEl = null; }
  root.addEventListener("mouseover", function (e) {
    var el = e.target.closest && e.target.closest("[data-tip]");
    if (el && root.contains(el)) show(el, e.clientX, e.clientY);
  });
  root.addEventListener("mousemove", function (e) { if (!tip.hidden) place(e.clientX, e.clientY); });
  root.addEventListener("mouseout", function (e) {
    var el = e.target.closest && e.target.closest("[data-tip]");
    if (el && !(e.relatedTarget && el.contains(e.relatedTarget))) hide();
  });
  root.addEventListener("focusin", function (e) {
    var el = e.target.closest && e.target.closest("[data-tip]");
    if (!el) return hide();
    showAt(el);
  });
  root.addEventListener("focusout", hide);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") hide(); });
  // Keep the tip on its mark while the page scrolls (scrolling a focused mark into view fires this too).
  window.addEventListener("scroll", function () {
    if (!anchorEl) return;
    var r = anchorEl.getBoundingClientRect();
    if (r.bottom < 0 || r.top > window.innerHeight) hide(); else place(r.left + r.width / 2 - 14, r.top);
  }, { passive: true });

  // ------------------------------------------------------------ filter and sort
  var form = root.querySelector("[data-lb-controls]");
  var countEl = root.querySelector("[data-lb-count]");
  var lists = Array.prototype.slice.call(root.querySelectorAll("[data-lb-rows]"));
  function readJson(id) {
    var el = document.getElementById(id);
    try { return el ? JSON.parse(el.textContent) : {}; } catch (err) { return {}; }
  }
  var data = readJson("lb-radar-data");   // per model: label, spoke values, shown strings
  var facts = readJson("lb-facts");       // per model: label, provider, prov, rank, order, first, n, spend

  // order: leaderboard position (ranked models first). A model with no published spend sorts last by cost.
  function sortKey(sort, k) {
    var f = facts[k] || {};
    if (sort === "first") return [-f.first, f.order];
    if (sort === "evaluated") return [-f.n, f.order];
    if (sort === "cost") return f.spend == null ? [1, 0, f.order] : [0, f.spend, f.order];
    return [f.order];
  }
  function apply(push) {
    if (!form) return;
    var q = form.q.value.trim().toLowerCase(), prov = form.prov.value, sort = form.sort.value, shown = 0;
    var keys = Object.keys(facts);
    keys.forEach(function (k) {
      var f = facts[k];
      var ok = (prov === "all" || prov === f.prov) &&
        (!q || String(f.label || "").toLowerCase().indexOf(q) >= 0 ||
          String(f.provider || "").toLowerCase().indexOf(q) >= 0);
      f.visible = ok;
      if (ok) shown += 1;
    });
    keys.sort(function (a, b) {
      var x = sortKey(sort, a), y = sortKey(sort, b);
      for (var i = 0; i < x.length; i++) if (x[i] !== y[i]) return x[i] - y[i];
      return 0;
    });
    lists.forEach(function (list) {
      var byKey = {};
      Array.prototype.forEach.call(list.children, function (el) { byKey[el.getAttribute("data-model")] = el; });
      keys.forEach(function (k) {
        var el = byKey[k];
        if (!el) return;
        el.hidden = !facts[k].visible;
        list.appendChild(el);
      });
    });
    if (countEl) countEl.textContent = shown + " of " + countEl.getAttribute("data-total") + " model rows shown";
    if (push) {
      var params = new URLSearchParams(location.search);
      [["q", form.q.value.trim(), ""], ["prov", prov, "all"], ["sort", sort, "within3"]].forEach(function (p) {
        if (p[1] && p[1] !== p[2]) params.set(p[0], p[1]); else params.delete(p[0]);
      });
      var s = params.toString();
      history.replaceState(null, "", location.pathname + (s ? "?" + s : "") + location.hash);
    }
  }
  if (form) {
    var applyBtn = form.querySelector("[data-lb-apply]");
    if (applyBtn) applyBtn.hidden = true;
    form.addEventListener("submit", function (e) { e.preventDefault(); apply(true); });
    form.q.addEventListener("input", function () { apply(true); });
    form.prov.addEventListener("change", function () { apply(true); });
    form.sort.addEventListener("change", function () { apply(true); });
    var reset = form.querySelector("[data-lb-reset]");
    if (reset) reset.addEventListener("click", function (e) {
      e.preventDefault();
      form.q.value = ""; form.prov.value = "all"; form.sort.value = "within3";
      apply(true);
    });
  }

  // ------------------------------------------------------------ compare radar
  var SERIES = [["#C08318", "circle"], ["#2FA596", "square"], ["#9A7BE0", "diamond"]];
  var SPOKES = [["Passed within three", "Passed within three"], ["Readable at try 1", "Readable at try 1"],
    ["Repaired after a miss", "Repaired after a miss"], ["Fewer tokens", "Tokens per brief"],
    ["Faster", "Mean latency"], ["Lower spend", "Run spend"]];
  var SVGNS = "http://www.w3.org/2000/svg";
  var CX = 330, CY = 240, R = 158;
  var compareForm = root.querySelector("[data-lb-compare]");
  var series = root.querySelector("[data-lb-series]");
  var radar = root.querySelector("[data-lb-radar]");
  var valuesTable = root.querySelector("[data-lb-values]");

  function pt(r, i, n) {
    var a = -Math.PI / 2 + 2 * Math.PI * i / n;
    return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
  }
  function f1(v) { return (Math.round(v * 10) / 10).toFixed(1); }
  function swatch(slot) {
    var c = SERIES[slot][0], s = SERIES[slot][1];
    var inner = s === "circle" ? '<circle cx="8" cy="8" r="5" fill="' + c + '"></circle>'
      : s === "square" ? '<rect x="3.2" y="3.2" width="9.6" height="9.6" fill="' + c + '"></rect>'
        : '<rect x="3.5" y="3.5" width="9" height="9" transform="rotate(45 8 8)" fill="' + c + '"></rect>';
    return '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" style="flex: none">' + inner + "</svg>";
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  // Pick order (which sets colour and marker) is the order of the series the server drew.
  var selection = [];
  root.querySelectorAll("polygon.lb-series").forEach(function (p) { selection.push(p.getAttribute("data-model")); });

  function draw() {
    if (!series) return;
    var polys = "", marks = "";
    selection.forEach(function (key, slot) {
      var d = data[key];
      if (!d) return;
      var c = SERIES[slot][0], shape = SERIES[slot][1];
      var pts = d.values.map(function (v, i) { return pt(R * v, i, 6); });
      polys += '<polygon class="lb-series" data-model="' + esc(key) + '" data-values="' + d.values.map(function (v) { return Math.round(v * 1000) / 1000; }).join(",") +
        '" points="' + pts.map(function (p) { return f1(p[0]) + "," + f1(p[1]); }).join(" ") + '" fill="' + c +
        '" fill-opacity="0.1" stroke="' + c + '" stroke-width="2.25" stroke-linejoin="round"></polygon>';
      pts.forEach(function (p, i) {
        var t = esc(d.label + " · " + SPOKES[i][0].toLowerCase() + ": " + d.shown[i]);
        var common = 'class="lb-tipmark" tabindex="0" data-tip="' + t + '" aria-label="' + t + '" fill="' + c + '" stroke="#15110F" stroke-width="2"';
        var x = p[0], y = p[1];
        if (shape === "circle") marks += "<circle " + common + ' cx="' + f1(x) + '" cy="' + f1(y) + '" r="5"></circle>';
        else if (shape === "square") marks += "<rect " + common + ' x="' + f1(x - 4.8) + '" y="' + f1(y - 4.8) + '" width="9.6" height="9.6"></rect>';
        else marks += "<rect " + common + ' x="' + f1(x - 4.5) + '" y="' + f1(y - 4.5) + '" width="9" height="9" transform="rotate(45 ' + f1(x) + " " + f1(y) + ')"></rect>';
      });
    });
    var tmp = document.createElementNS(SVGNS, "svg");
    tmp.innerHTML = polys + marks;
    series.replaceChildren.apply(series, Array.prototype.slice.call(tmp.childNodes));
    var names = selection.map(function (k) { return data[k] ? data[k].label : k; });
    var who = names.length > 1 ? names.slice(0, -1).join(", ") + " and " + names[names.length - 1] : (names[0] || "no models");
    if (radar) radar.setAttribute("aria-label", "Radar chart comparing " + who + " on six axes. The values are in the table beside it.");
    if (valuesTable) {
      var TH = valuesTable.querySelector("thead th").getAttribute("style");
      var head = '<tr><th scope="col" style="' + TH + '">VALUES SHOWN</th>' + selection.map(function (k, slot) {
        return '<th scope="col" style="' + TH + '"><span style="display: inline-flex; align-items: center; gap: 6px; font-family: \'Anek Latin\', \'Segoe UI\', system-ui, sans-serif; font-size: 14px; font-weight: 600; letter-spacing: 0; color: #F4EFE6">' +
          swatch(slot) + esc(data[k].label) + "</span></th>";
      }).join("") + "</tr>";
      var td = "padding: 9px 12px 9px 0; border-top: 1px solid #2E2621; vertical-align: top";
      var body = SPOKES.map(function (s, i) {
        return '<tr><th scope="row" style="text-align: left; font-weight: 500; ' + td + '">' + s[1] + "</th>" + selection.map(function (k) {
          return '<td style="' + td + "; font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; font-size: 13px; white-space: nowrap\">" + esc(data[k].shown[i]) + "</td>";
        }).join("") + "</tr>";
      }).join("");
      valuesTable.tHead.innerHTML = head;
      valuesTable.tBodies[0].innerHTML = body;
    }
    root.querySelectorAll(".lb-chip").forEach(function (b) {
      var key = b.getAttribute("data-key"), slot = selection.indexOf(key);
      var label = data[key] ? data[key].label : b.textContent;
      b.setAttribute("aria-pressed", slot >= 0 ? "true" : "false");
      b.innerHTML = (slot >= 0 ? swatch(slot) : "") + esc(label);
      var next = slot >= 0 ? selection.filter(function (k) { return k !== key; }) : selection.concat([key]).slice(-3);
      b.value = next.join(",");
    });
  }

  if (compareForm) {
    compareForm.addEventListener("click", function (e) {
      var b = e.target.closest(".lb-chip");
      if (!b) return;
      e.preventDefault();
      var key = b.getAttribute("data-key"), i = selection.indexOf(key);
      if (i >= 0) selection.splice(i, 1);
      else { selection.push(key); if (selection.length > 3) selection.shift(); }
      draw();
      var params = new URLSearchParams(location.search);
      params.delete("compare");
      var rest = params.toString();
      // Commas stay literal so the shared link reads ?compare=gpt-6-astra,claude-opus-5-5,beam-hf
      var query = "compare=" + selection.map(encodeURIComponent).join(",") + (rest ? "&" + rest : "");
      history.replaceState(null, "", location.pathname + "?" + query + "#profiles");
      b.focus();
    });
  }

  // ------------------------------------------------------------ community runs
  var refresh = root.querySelector("[data-lb-refresh]");
  if (refresh) {
    refresh.addEventListener("click", function (e) {
      e.preventDefault();
      var status = root.querySelector("[data-lb-community-status]");
      var wrap = root.querySelector("[data-lb-community-table]");
      var tbody = root.querySelector("[data-lb-community-rows]");
      refresh.disabled = true;
      status.textContent = "Checking for community runs…";
      fetch(refresh.getAttribute("data-src"), { cache: "no-store" }).then(function (r) {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      }).then(function (snap) {
        var rows = (snap && snap.rows) || [];
        if (!rows.length) {
          status.textContent = "No community runs have been merged yet.";
          wrap.hidden = true;
          return;
        }
        tbody.innerHTML = rows.map(function (r) {
          // One aggregate row per community model; "k of n" uses that run's own brief count.
          var label = r.label || r.key || "Unnamed model";
          var first = r.first_try != null ? r.first_try : "";
          return '<tr style="border-top: 1px solid #2E2621"><th scope="row" style="padding: 10px 16px 10px 0; text-align: left; font-weight: 600">' +
            esc(label) + ' <span style="margin-left: 8px; font-family: \'JetBrains Mono\', ui-monospace, Menlo, monospace; font-size: 11px; letter-spacing: 0.08em; color: #948980">COMMUNITY</span></th>' +
            '<td style="padding: 10px 16px; color: #B8ADA1">' + esc(r.provider || "") + "</td>" +
            "<td style=\"padding: 10px 16px; font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace\">" + (r.within3 != null ? esc(r.within3 + " of " + (r.n != null ? r.n : briefs)) : "") + "</td>" +
            "<td style=\"padding: 10px 0 10px 16px; font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace\">" + esc(first) + "</td></tr>";
        }).join("");
        wrap.hidden = false;
        status.textContent = rows.length + (rows.length === 1 ? " community run has" : " community runs have") + " been merged.";
      }).catch(function () {
        status.textContent = "Community runs could not be loaded. Reload the page to try again.";
      }).then(function () { refresh.disabled = false; });
    });
  }
})();
