const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { GoogleGenAI } = require('@google/genai');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.cjs'),
    },
    icon: path.join(__dirname, 'icon.png') // Optional: add an icon file
  });

  // In development, load from Vite dev server
  // In production, load from built files
  const startUrl = process.env.ELECTRON_START_URL || `file://${path.join(__dirname, 'dist/index.html')}`;
  
  mainWindow.loadURL(startUrl);

  // Open DevTools in development
  if (process.env.ELECTRON_START_URL) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.on('ready', () => {
  createWindow();

  ipcMain.handle('generate-content', async (event, { model, prompt }) => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: model,
        contents: prompt,
      });
      return response;
    } catch (error) {
      console.error('Error in generate-content:', error);
      throw error;
    }
  });

  ipcMain.handle('generate-images', async (event, { model, prompt, config }) => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.API_KEY });
      const response = await ai.models.generateImages({
        model: model,
        prompt: prompt,
        config: config,
      });
      return response;
    } catch (error) {
      console.error('Error in generate-images:', error);
      throw error;
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
