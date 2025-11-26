const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  generateContent: (model, prompt) => ipcRenderer.invoke('generate-content', { model, prompt }),
  generateImages: (model, prompt, config) => ipcRenderer.invoke('generate-images', { model, prompt, config }),
});
