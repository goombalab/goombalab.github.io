(function () {
  "use strict";
  function mountRippleCharts() {
    const fitTarget = document.getElementById("ripple-blog-fit");
    const efficiencyTarget = document.getElementById("ripple-blog-efficiency");
    if (!fitTarget || !efficiencyTarget || fitTarget.dataset.mounted) return;
    const options = { baseline: "v1.0" };
    window.RippleCharts.mountFit(fitTarget, window.RippleData, options);
    window.RippleCharts.mountEfficiencySwitcher(efficiencyTarget, window.RippleData, window.RippleRelativeData, options);
    fitTarget.dataset.mounted = "true";
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountRippleCharts, { once: true });
  } else {
    mountRippleCharts();
  }
})();
