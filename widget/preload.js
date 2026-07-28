'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('jarvisShell', {
  isWidget: true,
  show() {
    ipcRenderer.send('shell:show');
  },
  hide() {
    ipcRenderer.send('shell:hide');
  },
  toggle() {
    ipcRenderer.send('shell:toggle');
  },
  minimize() {
    ipcRenderer.send('shell:minimize');
  },
  notify(title, body) {
    ipcRenderer.send('shell:notify', { title, body });
  },
  quit() {
    ipcRenderer.send('shell:quit');
  },
});
