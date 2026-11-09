'use strict';

const heroVideo = document.querySelector('#hero-video');
const heroToggle = document.querySelector('#hero-toggle');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
function syncPlaybackButton() {
  const label = heroVideo.paused ? 'Play demo' : 'Pause demo';
  heroToggle.textContent = label + (heroVideo.paused ? ' ▶' : ' Ⅱ');
  heroToggle.setAttribute('aria-label', label);
}
heroVideo.addEventListener('play', syncPlaybackButton);
heroVideo.addEventListener('pause', syncPlaybackButton);
heroToggle.addEventListener('click', async () => {
  if (heroVideo.paused) {
    try { await heroVideo.play(); } catch { /* Keep the poster and play button available. */ }
  } else heroVideo.pause();
  syncPlaybackButton();
});
function respectMotionPreference() {
  if (reducedMotion.matches) {
    heroVideo.autoplay = false;
    heroVideo.pause();
  }
  syncPlaybackButton();
}
respectMotionPreference();
reducedMotion.addEventListener('change', respectMotionPreference);

const settings = {
  1: { name: 'Math Pack', title: 'Does the placement fit?', description: 'Purely geometric placement without physics or robot operation, isolating the algorithm’s spatial packing decisions.' },
  2: { name: 'Physics Pack', title: 'Does the stack remain stable?', description: 'Gravity, friction, and collisions test physical feasibility and stacking stability, without robotic execution.' },
  3: { name: 'Execution Pack', title: 'Can the robot execute the placement?', description: 'Physics simulation meets robotic motion planning and control, accounting for reachability, collisions, and stability during execution.' }
};
const tabs = [...document.querySelectorAll('[role="tab"]')];
const panel = document.querySelector('#demo-panel');
const demoVideos = [...panel.querySelectorAll('video')];
function selectSetting(tab) {
  const setting = tab.dataset.setting;
  if (tab.getAttribute('aria-selected') === 'true') return;
  tabs.forEach((item) => {
    const active = item === tab;
    item.setAttribute('aria-selected', String(active));
    item.tabIndex = active ? 0 : -1;
  });
  panel.setAttribute('aria-labelledby', tab.id);
  document.querySelector('#setting-title').textContent = settings[setting].title;
  document.querySelector('#setting-description').textContent = settings[setting].description;
  demoVideos.forEach((video) => {
    video.pause();
    video.src = `display/setting${setting}/${video.dataset.dataset}.mp4`;
    video.poster = `assets/setting${setting}-${video.dataset.dataset}.jpg`;
    const datasetName = video.closest('figure').querySelector('strong').textContent;
    video.setAttribute('aria-label', `${datasetName} dataset under ${settings[setting].name}`);
    video.load();
  });
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectSetting(tab));
  tab.addEventListener('keydown', (event) => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault();
    tabs[next].focus();
    selectSetting(tabs[next]);
  });
});
demoVideos.forEach((video) => video.addEventListener('play', () => {
  demoVideos.forEach((other) => { if (other !== video) other.pause(); });
}));

const copyButton = document.querySelector('#copy-citation');
copyButton.addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    status.textContent = 'Citation copied to clipboard.';
    copyButton.textContent = 'Copied!';
  } catch {
    status.textContent = 'Select the BibTeX text above to copy it manually.';
  }
});
