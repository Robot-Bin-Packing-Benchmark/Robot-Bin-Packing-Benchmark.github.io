# Homepage assets

The homepage is a buildless static page: `index.html`, `home.css`, and `home.js`. It is published by the existing GitHub Pages configuration from the repository root on `main`.

- `execution-demo.mp4`: derived from `display/setting3/time_series.mp4`, source interval 00:08–00:56, shown at 3× speed (16 seconds), 960 × 720, H.264, 24 fps, no audio. This is simulation footage.
- `execution-poster.jpg`: a still from the same original video at 00:20.
- `setting{1,2,3}-{time_series,occupancy,flat_long}.jpg`: video posters captured at 00:05 from each corresponding original video in `display/`.
- The framework overview uses the existing, unchanged `display/overview.png`.

The home page uses relative media paths. Demo selection supports mouse, touch, and arrow-key navigation. The hero respects reduced-motion preferences and has a pause/play control. Dataset videos load on demand and only one gallery video plays at a time. Citation text matches the repository README's arXiv citation until journal publication metadata is available.

To preview locally, run `python3 -m http.server 8766 --bind 127.0.0.1` from the repository root.

## Shared page styling

The leaderboard, documentation, download, and submission pages use the homepage's base styles (`home.css`) with `pages.css` for article layouts, dataset selectors, scrollable tables, and the documentation contents navigation. `site.js` owns the common navigation on all five pages; `home.js` retains the homepage demo and citation controls. `pages.js` adds keyboard activation to the existing sortable headers and highlights the selected documentation fragment.

The leaderboard's original CSV loading, sorting code, metric values, ranking highlights, and method-detail links are retained. The original dataset request form, email submission links, documentation examples, and submission policies remain in place.
