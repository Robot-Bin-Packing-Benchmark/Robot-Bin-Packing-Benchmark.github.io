'use strict';

// Preserve the existing leaderboard click handlers while making them keyboard accessible.
const resultsTables = document.querySelectorAll('.page-benchmark table[id]');
function labelSortableHeaders(table) {
  table.querySelectorAll('th[onclick]').forEach((header) => {
    header.tabIndex = 0;
    header.setAttribute('title', 'Sort by ' + header.innerText.replace(/\s+/g, ' ').trim());
  });
}
resultsTables.forEach((table) => {
  labelSortableHeaders(table);
  new MutationObserver(() => labelSortableHeaders(table)).observe(table.querySelector('thead'), {childList: true, subtree: true});
  table.addEventListener('keydown', (event) => {
    if (event.target.matches('th[onclick]') && ['Enter', ' '].includes(event.key)) {
      event.preventDefault();
      event.target.click();
    }
  });
});

// Native fragment links retain deep-linking and respect the shared sticky-header offset.
const contentsLinks = [...document.querySelectorAll('.sidebar a[href^="#"]')];
function updateContentsLink() {
  contentsLinks.forEach((link) => {
    if (link.hash === location.hash) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
window.addEventListener('hashchange', updateContentsLink);
updateContentsLink();
