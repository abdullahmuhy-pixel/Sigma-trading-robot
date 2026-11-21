// Sigma service worker: lets the app install on phones/desktops and open instantly.
// Live data (/api) always comes from the server so every device sees the same state.
const CACHE = 'sigma-shell-v2';
const SHELL = ['/', '/manifest.webmanifest', '/icon-192.png', '/icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).catch(() => {}));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin || url.pathname.startsWith('/api/')) return;
  // network first, fall back to the cached shell when offline
  e.respondWith(
    fetch(e.request).then(r => {
      if (r.ok && SHELL.includes(url.pathname)) caches.open(CACHE).then(c => c.put(e.request, r.clone()));
      return r;
    }).catch(() => caches.match(e.request).then(m => m || caches.match('/')))
  );
});

// tapping an opportunity notification opens (or focuses) the app on the Markets page
self.addEventListener('notificationclick', e => {
  e.notification.close();
  e.waitUntil(self.clients.matchAll({type: 'window', includeUncontrolled: true}).then(list => {
    for (const c of list) { if ('focus' in c) { c.navigate('/#markets'); return c.focus(); } }
    return self.clients.openWindow('/#markets');
  }));
});
