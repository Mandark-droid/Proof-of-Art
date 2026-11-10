// Replays and one-run pages: progressive enhancement only. Both pages work without this file.
(function () {
  "use strict";

  // Replays: choosing a state opens that state's brief at the same level.
  var brief = document.querySelector("[data-rv-brief]");
  var select = brief && brief.querySelector("[data-rv-state]");
  if (select) {
    select.addEventListener("change", function () {
      if (select.value) window.location.href = "/replays/" + encodeURIComponent(select.value);
    });
  }

  var run = document.querySelector("[data-rv-run]");
  if (!run) return;

  // "What the eye sees" / "What the checker sees": hide or show the checker's marks.
  var overlay = run.querySelector("[data-rv-overlay]");
  var buttons = run.querySelectorAll("[data-rv-view]");
  var chip = function (button, on) {
    button.setAttribute("aria-pressed", on ? "true" : "false");
    button.style.background = on ? "#F4EFE6" : "transparent";
    button.style.color = on ? "#15110F" : "#F4EFE6";
    button.style.borderColor = on ? "#F4EFE6" : "#4A3F38";
  };
  Array.prototype.forEach.call(buttons, function (button) {
    button.addEventListener("click", function () {
      var checker = button.getAttribute("data-rv-view") === "checker";
      Array.prototype.forEach.call(buttons, function (b) { chip(b, b === button); });
      if (overlay) overlay.classList.toggle("rv-off", !checker);
    });
  });

  // Copy link: copies the canonical URL of this run, including ?try=.
  var copy = run.querySelector("[data-rv-copy]");
  var status = run.querySelector("[data-rv-copied]");
  if (copy && navigator.clipboard) {
    copy.addEventListener("click", function (event) {
      event.preventDefault();
      var label = copy.textContent;
      navigator.clipboard.writeText(copy.href).then(function () {
        copy.textContent = "Link copied";
        if (status) status.textContent = "Link copied";
        window.setTimeout(function () {
          copy.textContent = label;
          if (status) status.textContent = "";
        }, 2000);
      });
    });
  }
})();
