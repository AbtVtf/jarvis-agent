'use strict';

const {
  app,
  BrowserWindow,
  screen,
  session,
  globalShortcut,
  Tray,
  Menu,
  nativeImage,
  ipcMain,
  Notification,
} = require('electron');
const path = require('path');

const SERVER_URL = 'http://127.0.0.1:8710/?widget=1';
const SERVER_ORIGIN = 'http://127.0.0.1:8710';
const MARGIN = 24;
const WIDTH = 420;
const HEIGHT = 640;
const RETRY_MS = 3000;

// 22x22 solid blue rounded dot (valid PNG, RGBA, anti-aliased circle).
const TRAY_ICON_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAABYAAAAWCAYAAADEtGw7AAAAhklEQVR42s3VwQ2AIAwFUHb6' +
  'e3SjTsMiTMIiPdRLTTw0ishXD/9CyEsDtBSoF0bKV7BAvUK9Q90iPdZkBgbUG9T9Ii32DsES' +
  'lflgLKs+q/QOesRxBrcJ9HgsKSwP0D2SwXUBXDO4L4B7BtsC2F6FaUdBuzzac6M1CLWlaUOI' +
  'Ojapg/6/f94Gl2EtOVBYk8QAAAAASUVORK5CYII=';

// The page plays TTS audio without user gestures and needs the mic.
// Must be set before the 'ready' event.
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');

let win = null;
let tray = null;
let retryTimer = null;
let isQuitting = false;

// Single-instance lock: focus/show the existing instance instead of running twice.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) win.showInactive();
  });

  app.whenReady().then(onReady);
}

function positionWindow() {
  if (!win) return;
  const { workArea } = screen.getPrimaryDisplay();
  win.setBounds({
    x: workArea.x + workArea.width - WIDTH - MARGIN,
    y: workArea.y + workArea.height - HEIGHT - MARGIN,
    width: WIDTH,
    height: HEIGHT,
  });
}

function loadServer() {
  if (!win) return;
  win.loadURL(SERVER_URL).catch(() => {
    /* handled by did-fail-load */
  });
}

function scheduleRetry() {
  if (retryTimer || !win) return;
  retryTimer = setTimeout(() => {
    retryTimer = null;
    loadServer();
  }, RETRY_MS);
}

function toggleWindow() {
  if (!win) return;
  if (win.isVisible()) {
    win.hide();
  } else {
    win.showInactive();
  }
}

function quitApp() {
  isQuitting = true;
  app.quit();
}

function createTray() {
  // GNOME may not support tray icons (no StatusNotifier host) — never fatal.
  try {
    const icon = nativeImage.createFromBuffer(
      Buffer.from(TRAY_ICON_BASE64, 'base64')
    );
    tray = new Tray(icon);
    tray.setToolTip('Jarvis');
    tray.setContextMenu(
      Menu.buildFromTemplate([
        { label: 'Show/Hide Jarvis', click: toggleWindow },
        { type: 'separator' },
        { label: 'Quit', click: quitApp },
      ])
    );
    tray.on('click', toggleWindow);
  } catch (err) {
    console.warn('Tray unavailable:', err.message);
  }
}

function onReady() {
  // Auto-grant mic (media) permission for the local Jarvis server only.
  session.defaultSession.setPermissionRequestHandler(
    (webContents, permission, callback, details) => {
      const url = details.requestingUrl || webContents.getURL() || '';
      callback(permission === 'media' && url.startsWith(SERVER_ORIGIN));
    }
  );

  win = new BrowserWindow({
    width: WIDTH,
    height: HEIGHT,
    frame: false,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: false,
    backgroundColor: '#0b0d12',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.setAlwaysOnTop(true, 'screen-saver');

  positionWindow();
  screen.on('display-metrics-changed', positionWindow);

  // Retry until the Jarvis server is up.
  win.webContents.on(
    'did-fail-load',
    (event, errorCode, errorDescription, validatedURL, isMainFrame) => {
      if (isMainFrame) scheduleRetry();
    }
  );
  loadServer();

  // Closing hides the widget; the app only quits via tray menu or shell:quit.
  win.on('close', (event) => {
    if (!isQuitting) {
      event.preventDefault();
      win.hide();
    }
  });

  globalShortcut.register('Ctrl+Shift+J', toggleWindow);

  createTray();

  // IPC surface used by preload.js.
  ipcMain.on('shell:show', () => win && win.showInactive());
  ipcMain.on('shell:hide', () => win && win.hide());
  ipcMain.on('shell:toggle', toggleWindow);
  ipcMain.on('shell:minimize', () => win && win.minimize());
  ipcMain.on('shell:notify', (event, { title, body } = {}) => {
    new Notification({ title: String(title || 'Jarvis'), body: String(body || '') }).show();
    if (win) win.showInactive();
  });
  ipcMain.on('shell:quit', quitApp);
}

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});
