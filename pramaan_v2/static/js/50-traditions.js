// Traditions: filter the cards by pattern type in place. Without this file the chips submit ?grammar=.
(function () {
  "use strict";
  var root = document.querySelector("[data-traditions]");
  if (!root) return;
  var form = root.querySelector("[data-tr-filter]");
  if (!form) return;
  var buttons = [].slice.call(form.querySelectorAll('button[name="grammar"]'));
  var cards = [].slice.call(root.querySelectorAll("article[data-tradition]"));
  var chips = [].slice.call(root.querySelectorAll("a[data-state]"));
  var empty = root.querySelector("[data-tr-empty]");
  var emptyLabel = root.querySelector("[data-tr-empty-label]");
  var ON = { borderColor: "#F4EFE6", background: "#F4EFE6", color: "#15110F" };
  var OFF = { borderColor: "#4A3F38", background: "transparent", color: "#F4EFE6" };

  function apply(grammar) {
    buttons.forEach(function (button) {
      var on = button.value === grammar;
      var look = on ? ON : OFF;
      button.setAttribute("aria-pressed", on ? "true" : "false");
      button.style.borderColor = look.borderColor;
      button.style.background = look.background;
      button.style.color = look.color;
    });
    var shown = 0;
    cards.forEach(function (card) {
      var show = !grammar || card.getAttribute("data-grammar") === grammar;
      card.hidden = !show;
      if (show) shown += 1;
    });
    if (empty) {
      empty.hidden = shown > 0;
      if (emptyLabel && grammar) {
        var pressed = buttons.filter(function (b) { return b.value === grammar; })[0];
        emptyLabel.textContent = pressed ? pressed.textContent.replace(/\s*\d+\s*$/, "").toLowerCase() : "";
      }
    }
    chips.forEach(function (chip) {
      var counts = {};
      try { counts = JSON.parse(chip.getAttribute("data-counts") || "{}"); } catch (e) { counts = {}; }
      var count = grammar ? (counts[grammar] || 0) : chip.getAttribute("data-all");
      var number = chip.querySelector("[data-count]");
      if (number) number.textContent = String(count);
      var href = "/traditions/" + chip.getAttribute("data-state") + (grammar ? "?grammar=" + grammar : "");
      chip.setAttribute("href", href);
    });
    root.setAttribute("data-grammar", grammar);
  }

  form.addEventListener("submit", function (event) {
    var button = event.submitter;
    if (!button || button.name !== "grammar") return;
    event.preventDefault();
    var grammar = button.value;
    apply(grammar);
    try {
      var url = new URL(window.location.href);
      if (grammar) url.searchParams.set("grammar", grammar); else url.searchParams.delete("grammar");
      window.history.replaceState(null, "", url.pathname + url.search + url.hash);
    } catch (e) { /* history is optional */ }
  });

  // A ?grammar= link opened before the server reads the query: apply it here.
  try {
    var asked = new URLSearchParams(window.location.search).get("grammar") || "";
    var known = buttons.some(function (b) { return b.value === asked; });
    if (known && asked !== (root.getAttribute("data-grammar") || "")) apply(asked);
  } catch (e) { /* no URLSearchParams */ }
})();
