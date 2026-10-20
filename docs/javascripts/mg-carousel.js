// Minimal click-through carousel. Each `.mg-carousel` holds N
// `.mg-carousel__slide` children plus a `.mg-carousel__nav` bar with
// prev/next buttons and an (auto-populated) `.mg-carousel__dots` box.
function mgInitCarousels() {
  document.querySelectorAll(".mg-carousel").forEach(function (car) {
    if (car.dataset.mgInit) return;
    var slides = [].slice.call(car.querySelectorAll(".mg-carousel__slide"));
    if (!slides.length) return;
    car.dataset.mgInit = "1";

    var dotsBox = car.querySelector(".mg-carousel__dots");
    var dots = [];
    var idx = 0;

    function show(i) {
      idx = (i + slides.length) % slides.length;
      slides.forEach(function (s, j) { s.classList.toggle("mg-active", j === idx); });
      dots.forEach(function (d, j) { d.classList.toggle("mg-active", j === idx); });
    }

    if (dotsBox) {
      slides.forEach(function (s, j) {
        var d = document.createElement("button");
        d.type = "button";
        d.className = "mg-carousel__dot";
        var title = s.getAttribute("data-title") || "Slide " + (j + 1);
        d.title = title;
        d.setAttribute("aria-label", title);
        d.addEventListener("click", function () { show(j); });
        dotsBox.appendChild(d);
        dots.push(d);
      });
    }

    var prev = car.querySelector(".mg-carousel__prev");
    var next = car.querySelector(".mg-carousel__next");
    if (prev) prev.addEventListener("click", function () { show(idx - 1); });
    if (next) next.addEventListener("click", function () { show(idx + 1); });

    show(0);
  });
}

// Material exposes the `document$` observable; use it so the carousel
// re-inits on instant navigation. Fall back to DOMContentLoaded.
if (typeof window.document$ !== "undefined" && window.document$.subscribe) {
  window.document$.subscribe(mgInitCarousels);
} else {
  document.addEventListener("DOMContentLoaded", mgInitCarousels);
}
