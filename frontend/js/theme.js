// Applies the saved or system colour theme before first paint to avoid a flash.
// Loaded synchronously in <head>; kept separate so no inline script is needed.
(function () {
  var theme = null;
  try { theme = localStorage.getItem('mp-theme'); } catch (e) { /* storage unavailable */ }
  if (!theme) {
    theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.setAttribute('data-theme', theme);
})();
