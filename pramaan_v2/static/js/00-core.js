// Pramaan v2: progressive enhancement only. Every page works without this file.
(function () {
  "use strict";
  var header = document.querySelector(".v2-header");
  var menu = header && header.querySelector(".v2-menu");
  if (menu) {
    menu.addEventListener("click", function () {
      var open = header.classList.toggle("open");
      menu.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }
  window.Pramaan = window.Pramaan || {};
})();
