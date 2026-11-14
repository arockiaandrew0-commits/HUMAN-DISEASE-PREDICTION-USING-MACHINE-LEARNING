// Frontend configuration.
//
// API_BASE_URL is where the Flask backend (backend/app.py) is running.
// - Opened through the backend (http://localhost:5000): '' = same server, nothing to change.
// - Opened from a separate static server or by double-clicking index.html:
//   the backend is expected at http://localhost:5000.
// Set a fixed URL here if you deploy the backend somewhere else.
(function () {
  var servedByBackend = location.protocol.indexOf('http') === 0 && location.port !== '8000';
  window.MEDPREDICT_CONFIG = {
    API_BASE_URL: servedByBackend ? '' : 'http://localhost:5000'
  };
})();
