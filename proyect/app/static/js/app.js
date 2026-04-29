// app.js — utilidades globales del Corpus Documental
document.addEventListener('DOMContentLoaded', () => {
  // Activar tooltips de Bootstrap si los hay
  const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  tooltips.forEach(el => new bootstrap.Tooltip(el));
});
