// Home: wall inspector and keyboard walk, kolam chips and draw-on, spot the break. Enhancement only.
(function () {
  "use strict";
  var wall = document.querySelector('[data-home="wall"]');
  if (!wall) return;

  var phone = window.matchMedia("(max-width: 719px)");
  var still = window.matchMedia("(prefers-reduced-motion: reduce)");

  // ---- The wall ---------------------------------------------------------------------------------
  (function () {
    var grid = wall.querySelector("[data-home-wall]");
    var main = wall.querySelector(".hw-main");
    var insp = wall.querySelector(".hw-insp");
    var blob = document.getElementById("hw-data");
    if (!grid || !insp || !blob) return;
    var data = JSON.parse(blob.textContent);
    var tiles = grid.querySelectorAll("a.hw-t");
    // Every tile carries its own cell: a model can miss runs, so NodeList position is not a cell.
    var COLS = Number(grid.getAttribute("data-cols")) || 36;
    var rows = Number(grid.getAttribute("data-rows")) || 1;
    var cellOf = [], at = {};
    tiles.forEach(function (t, k) {
      var c = { row: Number(t.getAttribute("data-row")), col: Number(t.getAttribute("data-col")) };
      cellOf[k] = c;
      at[c.row * COLS + c.col] = k;
    });
    function tileAt(row, col) { var k = at[row * COLS + col]; return k === undefined ? -1 : k; }
    var tones = {};
    Object.keys(data.verdicts).forEach(function (k) { tones[data.verdicts[k]] = data.tones[k]; });
    var el = {
      tile: insp.querySelector(".hw-tile"), model: insp.querySelector(".hi-model"),
      place: insp.querySelector(".hi-place"), trad: insp.querySelector(".hi-tn"),
      level: insp.querySelector(".hi-lv"), verdict: insp.querySelector(".hi-verdict"),
      open: insp.querySelector(".hi-open")
    };
    var styles = getComputedStyle(wall);
    var current = tileAt(Number(styles.getPropertyValue("--sr")), Number(styles.getPropertyValue("--sc")));
    if (!(current >= 0 && current < tiles.length)) current = 0;
    wall.classList.add("hw-js");

    function pct(v) { return (Math.round(v * 1000) / 1000) + "%"; }

    function place(i) {
      if (phone.matches) return;
      var row = cellOf[i].row, col = cellOf[i].col;
      var m = main.getBoundingClientRect(), t = tiles[i].getBoundingClientRect();
      var w = insp.offsetWidth, h = insp.offsetHeight, gap = 10, top0 = grid.getBoundingClientRect().top - m.top;
      var right = t.right - m.left + gap, left = t.left - m.left - gap - w;
      var x = col >= 20 ? (left >= 0 ? left : right) : (right + w <= m.width ? right : left);
      var y = row >= rows / 2 ? t.bottom - m.top - h : t.top - m.top;
      x = Math.max(0, Math.min(x, m.width - w));
      y = Math.max(top0, Math.min(y, m.height - h));
      insp.style.left = x + "px";
      insp.style.top = y + "px";
      insp.style.transform = "none";
    }

    function select(i) {
      var a = tiles[i];
      if (!a) return;
      var row = cellOf[i].row, col = cellOf[i].col, level = row % 2 ? "hard" : "easy";
      var verdict = a.getAttribute("aria-label").split(": ").pop().toUpperCase();
      var never = a.classList.contains("hw-n");
      var model = data.models[Math.floor(row / 2)], state = data.states[col][0];
      wall.style.setProperty("--sc", col);
      wall.style.setProperty("--sr", row);
      el.model.textContent = model;
      el.place.textContent = state;
      el.trad.textContent = data.trad[col + level.charAt(0)] || "";
      el.level.textContent = " · " + level;
      el.verdict.textContent = verdict;
      el.verdict.style.color = tones[verdict] || "";
      el.open.href = a.getAttribute("href");
      el.tile.classList.toggle("hw-empty", never);
      el.tile.style.backgroundPosition = never ? "" : pct(col / Math.max(COLS - 1, 1) * 100) + " " + pct(row / Math.max(rows - 1, 1) * 100);
      el.tile.setAttribute("aria-label", never ? "No readable pattern" :
        "Tile drawn by " + model + " for " + state + ", " + level + " brief");
      current = i;
      place(i);
    }

    function rove(i) {
      tiles.forEach(function (t, k) { t.tabIndex = k === i ? 0 : -1; });
    }

    rove(current);
    select(current);

    var touched = false, wasCurrent = false;
    grid.addEventListener("pointerdown", function (e) {
      var a = e.target.closest && e.target.closest("a.hw-t");
      touched = e.pointerType === "touch" || e.pointerType === "pen";
      wasCurrent = !!a && Array.prototype.indexOf.call(tiles, a) === current;
    });
    grid.addEventListener("mouseover", function (e) {
      var a = e.target.closest && e.target.closest("a.hw-t");
      if (a && !touched) select(Array.prototype.indexOf.call(tiles, a));
    });
    grid.addEventListener("focusin", function (e) {
      var i = Array.prototype.indexOf.call(tiles, e.target);
      if (i < 0) return;
      rove(i);
      select(i);
    });
    grid.addEventListener("click", function (e) {
      var a = e.target.closest && e.target.closest("a.hw-t");
      if (!a) return;
      var i = Array.prototype.indexOf.call(tiles, a);
      if (touched && !wasCurrent) {
        e.preventDefault();
        rove(i);
        select(i);
      }
      touched = false;
    });
    grid.addEventListener("keydown", function (e) {
      var i = Array.prototype.indexOf.call(tiles, e.target);
      if (i < 0) return;
      var row = cellOf[i].row, col = cellOf[i].col, next = i;
      // The nearest tile in a direction, stepping over cells with no run; stay put at the edge.
      function walk(dr, dc) {
        for (var r = row + dr, c = col + dc; r >= 0 && r < rows && c >= 0 && c < COLS; r += dr, c += dc) {
          var k = tileAt(r, c);
          if (k >= 0) return k;
        }
        return i;
      }
      function rowEnd(fromCol, dc) {
        for (var c = fromCol; c >= 0 && c < COLS; c += dc) { var k = tileAt(row, c); if (k >= 0) return k; }
        return i;
      }
      switch (e.key) {
        case "ArrowRight": next = walk(0, 1); break;
        case "ArrowLeft": next = walk(0, -1); break;
        case "ArrowDown": next = walk(1, 0); break;
        case "ArrowUp": next = walk(-1, 0); break;
        case "Home": next = e.ctrlKey ? 0 : rowEnd(0, 1); break;
        case "End": next = e.ctrlKey ? tiles.length - 1 : rowEnd(COLS - 1, -1); break;
        default: return;
      }
      e.preventDefault();
      rove(next);
      tiles[next].focus();
    });
    window.addEventListener("resize", function () { place(current); });

    // Phone: the inspector is a bottom sheet while the wall is on screen.
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        wall.classList.toggle("hw-sheet", entries[0].isIntersecting);
      }, { rootMargin: "0px 0px -15% 0px" }).observe(grid);
    }
  })();

  // ---- Watch one get checked: chips and strokes ---------------------------------------------------
  (function () {
    var section = document.querySelector('[data-home="kolam"]');
    if (!section) return;
    var chips = section.querySelectorAll(".hk-chip");
    var groups = section.querySelectorAll(".hk-tries");
    var animate = !still.matches && "IntersectionObserver" in window;
    var seen = false;

    function draw(group) {
      if (!group.classList.contains("hk-pending")) return;
      group.getBoundingClientRect();
      window.requestAnimationFrame(function () {
        window.requestAnimationFrame(function () { group.classList.remove("hk-pending"); });
      });
    }

    if (animate) {
      groups.forEach(function (g) { g.classList.add("hk-pending"); });
      new IntersectionObserver(function (entries, observer) {
        if (!entries[0].isIntersecting) return;
        seen = true;
        groups.forEach(function (g) { if (!g.hidden) draw(g); });
        observer.disconnect();
      }, { threshold: 0.35 }).observe(section.querySelector(".hk-tries").parentNode);
    }

    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        var key = chip.getAttribute("data-model");
        chips.forEach(function (c) {
          var on = c === chip;
          c.setAttribute("aria-pressed", on ? "true" : "false");
          c.style.borderColor = on ? "#F4EFE6" : "#4A3F38";
          c.style.background = on ? "#F4EFE6" : "transparent";
          c.style.color = on ? "#15110F" : "#F4EFE6";
        });
        groups.forEach(function (g) {
          g.hidden = g.getAttribute("data-model") !== key;
          if (!g.hidden && seen) draw(g);
        });
      });
    });
  })();

  // ---- Spot the break ---------------------------------------------------------------------------
  (function () {
    var section = document.querySelector('[data-home="spot"]');
    if (!section) return;
    var question = section.querySelector(".hs-q");
    var answer = section.querySelector(".hs-answer");
    var buttons = section.querySelectorAll(".hs-reveal");
    var cover = section.querySelector(".hs-cover");
    if (!question || !answer) return;

    function show(revealed, focus) {
      section.classList.toggle("hs-covered", !revealed);
      question.hidden = revealed;
      answer.hidden = !revealed;
      if (!focus) return;
      if (revealed) answer.querySelector("h2").focus();
      else buttons[0].focus();
    }

    show(false, false);
    buttons.forEach(function (b) { b.addEventListener("click", function () { show(true, true); }); });
    if (cover) cover.addEventListener("click", function () { show(false, true); });
  })();
})();
