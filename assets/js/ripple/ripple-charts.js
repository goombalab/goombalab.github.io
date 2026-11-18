/* Ripple's published observations + an independently fitted baseline.
 * No charting-library or network dependency. See README.md for embedding.
 */
(function (global) {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const FINAL = 'Final (RI-Mix-14T)';
  const seriesColor = series => series.key === FINAL ? `var(--rc-final-color, ${series.color})` : series.color;
  let counter = 0;
  const label = key => key === 'Olmo 3' ? 'OLMo 3' : key === FINAL ? 'Final Ripple' : key;
  const exp = n => String(n).replace(/-/g, '−').split('').map(c => ({'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','−':'⁻'}[c] || c)).join('');
  function sci(n) {
    if (!Number.isFinite(n)) return 'No finite match';
    const power = Math.floor(Math.log10(Math.abs(n) || 1));
    return `${(n / 10 ** power).toPrecision(3).replace(/\.?0+$/, '')} × 10${exp(power)}`;
  }
  function gain(n) {
    if (!Number.isFinite(n)) return 'No finite match';
    if (n >= 1e6) return `${(n / 1e6).toPrecision(3)}M×`;
    if (n >= 1e3) return `${(n / 1e3).toPrecision(3)}k×`;
    return `${n < 2 ? n.toFixed(2) : n < 100 ? n.toFixed(1) : n.toFixed(0)}×`;
  }
  function node(tag, attrs = {}, text) {
    const el = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function svgNode(tag, attrs = {}, text) {
    const el = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function predict(c, fit, reference) { return fit.E + fit.K * (c / reference) ** -fit.alpha; }
  function observation(c, loss, fit, reference) {
    const logC = loss > fit.E ? Math.log(reference) + Math.log(fit.K / (loss - fit.E)) / fit.alpha : NaN;
    const equivalent = logC < Math.log(Number.MAX_VALUE) ? Math.exp(logC) : NaN;
    const inside = Number.isFinite(equivalent) && loss >= fit.lossRange[0] && loss <= fit.lossRange[1] && equivalent >= fit.computeRange[0] && equivalent <= fit.computeRange[1];
    return { c, loss, equivalent, ce: equivalent / c, inside, finite: Number.isFinite(equivalent), reason: loss <= fit.E ? 'Loss is at or below the fitted floor.' : 'Inverse exceeds numerical range.' };
  }
  function appendTooltip(tooltip, title, entries, note) {
    tooltip.replaceChildren(node('strong', {}, title));
    const dl = node('dl');
    entries.forEach(([term, value]) => dl.append(node('dt', {}, term), node('dd', {}, value)));
    tooltip.append(dl);
    if (note) tooltip.append(node('p', {}, note));
  }
  function mount(target, data, options, type) {
    const host = typeof target === 'string' ? document.querySelector(target) : target;
    if (!host) throw new Error('RippleCharts: mount target does not exist.');
    const reference = data.referenceCompute || 1e20;
    const relative = Boolean(options.relative);
    const seriesLabel = series => relative ? series.comparisonLabel : label(series.key);
    let baseline = options.baseline || 'v1.0';
    if (!data.fits[baseline]) throw new Error(`RippleCharts: missing fit for ${baseline}.`);
    const id = `ripple-${++counter}`;
    const root = node('figure', { class: 'ripple-chart', 'data-chart': type });
    const title = node('figcaption', { id: `${id}-title`, class: 'rc-caption' });
    const plot = node('div', { class: 'rc-plot' });
    const legend = node('div', { class: 'rc-legend', 'aria-label': relative ? 'Select a recipe comparison' : 'Select a recipe' });
    const rangeKey = node('div', { class: 'rc-range-key' });
    const tooltip = node('div', { class: 'rc-tooltip', role: 'tooltip', id: `${id}-tip`, hidden: '' });
    let destroyed = false, pinned = FINAL, active = null;
    let pointerSeries = null, focusSeries = null, inputMode = 'pointer', selectionFrame = null;
    let groups = [], targets = [], labelGroups = [], legendButtons = [];
    root.setAttribute('aria-labelledby', title.id);
    root.append(plot);
    if (type === 'efficiency') root.append(legend, rangeKey);
    root.append(title);
    host.append(root);

    function setActive(key) {
      legendButtons.forEach(item => {
        item.el.setAttribute('aria-pressed', String(item.key === pinned));
        item.el.dataset.active = String(item.key === key);
      });
      if (active === key) return;
      active = key;
      root.dataset.activeSeries = key;
      groups.forEach(item => item.el.style.opacity = item.key === key ? '1' : '.18');
      labelGroups.forEach(item => item.el.style.display = item.key === key ? '' : 'none');
      targets.forEach(item => item.el.dataset.active = item.key === key ? 'true' : 'false');
      const group = groups.find(item => item.key === key);
      if (group) group.el.parentNode.append(group.el);
      const labels = labelGroups.find(item => item.key === key);
      if (labels) labels.el.parentNode.append(labels.el);
    }
    function resolveSelection() {
      const key = inputMode === 'keyboard' ? focusSeries || pointerSeries || pinned : pointerSeries || focusSeries || pinned;
      setActive(key);
    }
    function scheduleSelection() {
      if (selectionFrame !== null) return;
      selectionFrame = global.requestAnimationFrame(() => {
        selectionFrame = null;
        if (!destroyed) resolveSelection();
      });
    }
    function bindSelection(button, key, isLegend) {
      button.addEventListener('pointerenter', () => {
        inputMode = 'pointer'; pointerSeries = key; scheduleSelection();
      });
      button.addEventListener('pointermove', event => {
        if (event.movementX || event.movementY) {
          inputMode = 'pointer'; pointerSeries = key; scheduleSelection();
        }
      });
      // A legend item keeps its preview through gaps until the pointer exits the legend.
      if (!isLegend) button.addEventListener('pointerleave', () => {
        if (pointerSeries === key) pointerSeries = null;
        scheduleSelection();
      });
      button.addEventListener('focus', () => {
        focusSeries = key;
        if (button.matches(':focus-visible')) inputMode = 'keyboard';
        scheduleSelection();
      });
      button.addEventListener('blur', () => {
        if (focusSeries === key) focusSeries = null;
        scheduleSelection();
      });
      button.addEventListener('click', () => { pinned = key; scheduleSelection(); });
    }
    legend.addEventListener('pointerleave', () => { pointerSeries = null; scheduleSelection(); });
    function hideTooltip() { tooltip.hidden = true; }
    function showTooltip(x, y, series, point, fit) {
      const pred = predict(point.c, fit, reference);
        appendTooltip(tooltip, label(series.key), [
          ['Compute', `${sci(point.c)} FLOPs`], ['Loss', point.loss.toFixed(3)], ['Predicted loss', pred.toFixed(4)]
        ]);
      tooltip.hidden = false;
      const width = plot.clientWidth;
      const box = tooltip.getBoundingClientRect();
      const left = Math.max(6, Math.min(x + 12, width - box.width - 6));
      const top = Math.max(4, y - box.height - 14);
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
    }
    function pointTarget(x, y, series, point, fit) {
      const btn = node('button', { type: 'button', class: 'rc-point-target', 'data-series': series.key,
        'data-point-index': series.points.findIndex(([c,l]) => c === point.c && l === point.loss) + 1,
        'data-status': type === 'fit' ? 'observed' : point.inside ? 'within-fit-range' : 'extrapolated',
        'aria-label': `${seriesLabel(series)}: compute ${point.c.toExponential(3)} FLOPs, loss ${point.loss.toFixed(3)}${type === 'efficiency' ? `, compute efficiency ${point.ce.toFixed(2)} times, ${point.inside ? 'within fit range' : 'extrapolated fit'}` : `, predicted loss ${predict(point.c, fit, reference).toFixed(4)}`}` });
      btn.style.left = `${x}px`; btn.style.top = `${y}px`;
      if (type === 'efficiency') {
        bindSelection(btn, series.key, false);
      } else {
        const reveal = () => showTooltip(x, y, series, point, fit);
        btn.addEventListener('pointerenter', reveal);
        btn.addEventListener('focus', reveal);
        btn.addEventListener('pointerleave', () => { if (document.activeElement !== btn) hideTooltip(); });
        btn.addEventListener('blur', hideTooltip);
        btn.addEventListener('click', reveal);
      }
      plot.append(btn);
      targets.push({ key: series.key, el: btn });
    }
    function makeAxes(svg, width, height, x, y, bounds, kind) {
      const { left, right, top, bottom } = bounds;
      const powers = width < 430 ? [19, 21] : [19, 20, 21, 22];
      powers.forEach(power => {
        const xx = x(10 ** power);
        if (xx >= left && xx <= width - right) {
          svg.append(svgNode('line', { x1: xx, x2: xx, y1: top, y2: height - bottom, class: 'rc-grid-line', opacity: '.35' }));
          svg.append(svgNode('text', { x: xx, y: height - bottom + 20, 'text-anchor': 'middle', class: 'rc-tick' }, `10${exp(power)}`));
        }
      });
      let ticks;
      if (kind === 'fit') ticks = [1, 1.2, 1.4, 1.6, 1.8, 2];
      else {
        ticks = [.25, .5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        if (bounds.yMax > 1024) {
          ticks = [1];
          for (let n = 10; n <= bounds.yMax * 1.01; n *= 10) ticks.push(n);
        }
        const visible = ticks.filter(t => t >= bounds.yMin && t <= bounds.yMax);
        ticks = visible.length > 8 ? visible.filter((_, i) => i % 2 === 0) : visible;
      }
      ticks.forEach(t => {
        const yy = y(t);
        if (yy >= top && yy <= height - bottom) {
          svg.append(svgNode('line', { x1: left, x2: width - right, y1: yy, y2: yy, class: 'rc-grid-line' }));
          svg.append(svgNode('text', { x: left - 8, y: yy + 4, 'text-anchor': 'end', class: 'rc-tick' }, kind === 'fit' ? t.toFixed(1) : t < 1000 ? `${t}×` : gain(t)));
        }
      });
      svg.append(svgNode('line', { x1: left, x2: width - right, y1: height - bottom, y2: height - bottom, class: 'rc-axis' }));
      svg.append(svgNode('line', { x1: left, x2: left, y1: top, y2: height - bottom, class: 'rc-axis' }));
      svg.append(svgNode('text', { x: (left + width - right) / 2, y: height - 7, 'text-anchor': 'middle' }, 'Training compute (FLOPs)'));
      const yy = (top + height - bottom) / 2;
      svg.append(svgNode('text', { x: 14, y: yy, transform: `rotate(-90,14,${yy})`, 'text-anchor': 'middle' }, kind === 'fit' ? 'Cross-entropy Loss' : 'Compute Efficiency'));
    }
    function draw() {
      if (destroyed) return;
      const width = Math.max(280, Math.round(root.clientWidth));
      const height = (width < 450 ? 340 : 390) - (type === 'fit' ? 20 : 0);
      const bounds = { left: width < 450 ? 57 : 65, right: 25, top: type === 'fit' ? 12 : 32, bottom: 55 };
      const fit = data.fits[baseline];
      const base = data.series.find(s => s.key === baseline);
      if (!base) throw new Error('RippleCharts: required baseline series missing.');
      groups = []; targets = []; labelGroups = []; legendButtons = [];
      active = null; pointerSeries = null; focusSeries = null;
      plot.replaceChildren(); legend.replaceChildren(); rangeKey.replaceChildren();
      title.textContent = type === 'fit' ? `${label(baseline)} baseline fit` : relative ? 'Compute efficiency relative to the previous ladder version' : `Compute efficiency relative to ${label(baseline)}`;
      const svg = svgNode('svg', { class: 'rc-svg', width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-labelledby': `${id}-svg-title ${id}-svg-desc` });
      svg.append(svgNode('title', { id: `${id}-svg-title` }, title.textContent));
      svg.append(svgNode('desc', { id: `${id}-svg-desc` }, type === 'fit' ? 'Published baseline observations and power law fitted to all six points. Each point has keyboard and pointer details.' : relative ? 'Equal-loss compute ratio versus training compute, with each comparison using its own fitted baseline. v1.5 is compared with v1.3. Filled circles are within that baseline’s measured range; open circles extrapolate. Select or hover a comparison to show its gains.' : 'Equal-loss compute ratio versus training compute. Filled circles are within baseline support; open circles extrapolate. Select or hover a recipe to show its gains.'));
      plot.append(svg);
      const xMin = type === 'fit' ? Math.min(...base.points.map(p => p[0])) / 1.35 : 3.5e18;
      const xMax = type === 'fit' ? Math.max(...base.points.map(p => p[0])) * 1.25 : 1.3e22;
      const x = c => bounds.left + (Math.log10(c) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin)) * (width - bounds.left - bounds.right);
      if (type === 'fit') {
        const losses = base.points.map(p => p[1]);
        const lossMin = Math.min(...losses, predict(base.points[base.points.length - 1][0], fit, reference));
        const lossMax = Math.max(...losses);
        const lossPadding = (lossMax - lossMin) * .1;
        const yMin = lossMin - lossPadding, yMax = lossMax + lossPadding;
        const y = value => height - bounds.bottom - (value - yMin) / (yMax - yMin) * (height - bounds.bottom - bounds.top);
        makeAxes(svg, width, height, x, y, bounds, 'fit');
        const coords = [];
        for (let i = 0; i <= 180; i++) {
          const c = Math.exp(Math.log(fit.computeRange[0]) + i / 180 * Math.log(fit.computeRange[1] / fit.computeRange[0]));
          coords.push(`${x(c)},${y(predict(c, fit, reference))}`);
        }
        svg.append(svgNode('polyline', { points: coords.join(' '), fill: 'none', stroke: seriesColor(base), 'stroke-width': 2.3, 'data-fit-segment': 'fitted' }));
        base.points.forEach(([c,l], index) => {
          svg.append(svgNode('circle', { cx: x(c), cy: y(l), r: 4.1, fill: seriesColor(base), stroke: seriesColor(base), 'stroke-width': 1, 'data-point-index': index + 1 }));
          pointTarget(x(c), y(l), base, { c, loss: l }, fit);
        });
      } else {
        const series = data.series.slice().sort((a,b) => a.key === FINAL ? -1 : b.key === FINAL ? 1 : a.key.localeCompare(b.key));
        const all = series.map(s => {
          const comparisonFit = relative ? data.fits[s.baselineKey] : fit;
          if (!comparisonFit) throw new Error(`RippleCharts: missing fit for ${s.baselineKey}.`);
          return { ...s, fit: comparisonFit, observations: s.points.map(([c,l]) => observation(c,l,comparisonFit,reference)) };
        });
        const shown = all.map(s => ({ ...s, visible: s.observations.filter(p => p.finite) }));
        const values = shown.flatMap(s => s.visible.map(p => p.ce));
        bounds.yMin = Math.min(.5, Math.min(...values) / 1.25);
        bounds.yMax = Math.max(4, Math.max(...values) * 1.5);
        const y = value => height - bounds.bottom - (Math.log(value) - Math.log(bounds.yMin)) / Math.log(bounds.yMax / bounds.yMin) * (height - bounds.bottom - bounds.top);
        makeAxes(svg, width, height, x, y, bounds, 'efficiency');
        svg.append(svgNode('line', { x1: bounds.left, x2: width-bounds.right, y1: y(1), y2: y(1), class: 'rc-axis', 'stroke-dasharray': '4 4' }));
        shown.forEach(series => {
          const g = svgNode('g', { class: 'rc-curve', 'data-series': series.key });
          const gains = svgNode('g', { 'data-gains': series.key, 'pointer-events': 'none' });
          let last = null;
          series.observations.forEach(p => {
            const visible = p.finite;
            if (!visible) { last = null; return; }
            if (last) g.append(svgNode('line', { x1: x(last.c), x2: x(p.c), y1: y(last.ce), y2: y(p.ce), stroke: seriesColor(series), 'stroke-width': series.key === FINAL ? 2.3 : 1.7, 'stroke-dasharray': last.inside && p.inside ? 'none' : '4 4' }));
            g.append(svgNode('circle', { cx: x(p.c), cy: y(p.ce), r: 4.2, fill: p.inside ? seriesColor(series) : 'var(--rc-bg)', stroke: seriesColor(series), 'stroke-width': 1.8 }));
            pointTarget(x(p.c), y(p.ce), series, p, series.fit);
            last = p;
          });
          let prior = null;
          series.visible.forEach(p => {
            let yy = y(p.ce) - 12;
            const xx = x(p.c), text = gain(p.ce), textWidth = text.length * 6.5;
            if (prior && xx - prior.x < (textWidth + prior.w) / 2 + 3 && Math.abs(yy - prior.y) < 14) yy -= 15;
            yy = Math.max(12, yy);
            gains.append(svgNode('text', { x: xx, y: yy, 'text-anchor': 'middle', class: 'rc-gain' }, text));
            prior = { x: xx, y: yy, w: textWidth };
          });
          svg.append(g, gains); groups.push({ key: series.key, el: g }); labelGroups.push({ key: series.key, el: gains });
          const button = node('button', { type: 'button', 'aria-pressed': 'false', 'data-series': series.key });
          const swatch = node('span', { class: 'rc-swatch', 'aria-hidden': 'true' }); swatch.style.setProperty('--series', seriesColor(series));
          button.append(swatch, document.createTextNode(seriesLabel(series)));
          bindSelection(button, series.key, true);
          legend.append(button); legendButtons.push({ key: series.key, el: button });
        });
        const fillKey = node('span'), openKey = node('span');
        fillKey.append(node('i', { class: 'rc-range-dot', 'aria-hidden': 'true' }), document.createTextNode(relative ? 'Within each baseline’s fit range' : 'Within fit range'));
        openKey.append(node('i', { class: 'rc-range-dot rc-open', 'aria-hidden': 'true' }), document.createTextNode('Extrapolated baseline fit'));
        rangeKey.append(fillKey, openKey);
        setActive(pinned);
      }
      if (type === 'fit') { plot.append(tooltip); hideTooltip(); }
    }
    root.addEventListener('keydown', event => {
      inputMode = 'keyboard';
      if (event.key === 'Escape') {
        hideTooltip(); pinned = FINAL; pointerSeries = null; focusSeries = null;
      }
      if (type === 'efficiency') scheduleSelection();
    });
    let previousWidth = 0;
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(entries => {
      const width = Math.round(entries[0].contentRect.width);
      if (width !== previousWidth) { previousWidth = width; draw(); }
    }) : null;
    if (observer) observer.observe(root); else global.addEventListener('resize', draw);
    draw();
    return {
      setBaseline(key) {
        if (!data.fits[key]) throw new Error(`RippleCharts: missing fit for ${key}.`);
        baseline = key; pinned = FINAL; draw();
      },
      destroy() { destroyed = true; if (selectionFrame !== null) global.cancelAnimationFrame(selectionFrame); if (observer) observer.disconnect(); else global.removeEventListener('resize', draw); root.remove(); }
    };
  }
  function mountEfficiencySwitcher(target, data, relativeData, options = {}) {
    const host = typeof target === 'string' ? document.querySelector(target) : target;
    if (!host) throw new Error('RippleCharts: mount target does not exist.');
    const root = node('div', { class: 'ripple-chart-switcher' });
    const controls = node('fieldset', { class: 'rc-mode-switch' });
    controls.append(node('legend', { class: 'rc-sr' }, 'Compute efficiency comparison'));
    const name = `ripple-mode-${++counter}`;
    const inputs = [];
    for (const [value, text] of [['v1.0', 'Relative to v1.0'], ['previous', 'Relative to previous version']]) {
      const option = node('label', { class: 'rc-mode-option' });
      const input = node('input', { type: 'radio', name, value, class: 'rc-sr' });
      option.append(input, node('span', {}, text));
      controls.append(option);
      inputs.push(input);
    }
    const chartHost = node('div');
    root.append(controls, chartHost);
    host.append(root);
    let chart = null, mode = null, destroyed = false;
    function setMode(nextMode) {
      if (destroyed || nextMode === mode) return;
      if (!['v1.0', 'previous'].includes(nextMode)) throw new Error(`RippleCharts: unknown comparison mode ${nextMode}.`);
      if (chart) chart.destroy();
      mode = nextMode;
      inputs.forEach(input => { input.checked = input.value === mode; });
      root.dataset.mode = mode;
      chart = mount(chartHost, mode === 'previous' ? relativeData : data,
        { ...options, baseline: 'v1.0', relative: mode === 'previous' }, 'efficiency');
    }
    controls.addEventListener('change', event => {
      if (event.target.matches('input[type="radio"]')) setMode(event.target.value);
    });
    setMode(options.mode || 'v1.0');
    return {
      setMode,
      destroy() { destroyed = true; if (chart) chart.destroy(); root.remove(); }
    };
  }
  global.RippleCharts = {
    mountFit: (target, data, options = {}) => mount(target, data, options, 'fit'),
    mountEfficiency: (target, data, options = {}) => mount(target, data, options, 'efficiency'),
    mountEfficiencySwitcher
  };
})(window);
