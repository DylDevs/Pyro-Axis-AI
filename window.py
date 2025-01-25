from multiprocessing import JoinableQueue
import multiprocessing
import screeninfo
import requests
import logging
import webview
import ctypes
import json
import time
import os

CACHE = os.path.join(os.path.dirname(__file__), "cache")
DEBUG_MODE = False
WIDTH = 1300
HEIGHT = 820
IS_TRANSPARENT = False

if not os.path.exists(CACHE):
    os.mkdir(CACHE)
if not os.path.exists(os.path.join(CACHE, "cache.json")):
    with open(os.path.join(CACHE, "cache.json"), "w") as f:
        json.dump({}, f)
        f.close()

def GetCacheItem(key, default=None):
    cache_content = json.load(open(os.path.join(CACHE, "cache.json"), "r"))
    if key in cache_content:
        return cache_content[key]
    
    with open(os.path.join(CACHE, "cache.json"), "w") as f:
        cache_content[key] = default
        json.dump(cache_content, f)
    return default

screen = screeninfo.get_monitors()[GetCacheItem("screen", 0)]
screen_width = screen.width
screen_height = screen.height

webview.settings = {
    'ALLOW_DOWNLOADS': False,
    'ALLOW_FILE_URLS': True,
    'OPEN_EXTERNAL_LINKS_IN_BROWSER': True,
    'OPEN_DEVTOOLS_IN_DEBUG': True
}

def start_webpage(frontend_started):
    global window, html
    
    def load_website(window:webview.Window):
        # Wait until the server is ready
        RETRY_INTERVAL = 0.5
        HAS_STARTED = False
        while not HAS_STARTED:
            try:
                response = requests.get(
                    f'http://localhost:3000',
                    timeout=2
                )
                if response.ok:
                    HAS_STARTED = True
                    break
            except:
                pass  # Handle timeout
            
            time.sleep(RETRY_INTERVAL)
        window.load_url('http://localhost:3000')

    # Load window x any y position from cache, make sure it's in screen bounds
    window_x = GetCacheItem("window_x", screen_width // 2 - WIDTH // 2)
    window_y = GetCacheItem("window_y", screen_height // 2 - HEIGHT // 2)
    if window_x < 0: window_x = 0
    if window_y < 0: window_y = 0
    if window_x > screen_width - WIDTH: window_x = screen_width - WIDTH
    if window_y > screen_height - HEIGHT: window_y = screen_height - HEIGHT
    if window_x + WIDTH > screen_width: window_x = screen_width - WIDTH
    if window_y + HEIGHT > screen_height: window_y = screen_height - HEIGHT

    if not frontend_started: # If we're not starting the frontend, assume it's already started locally
        html = html.replace("Please wait while Pyro Axis AI Training Dashboard loads", "Connecting to Pyro Axis AI Training Dashboard")
   
    window = webview.create_window(
        f'Torch Training UI', 
        html=html, 
        x = window_x,
        y = window_y,
        width=WIDTH+20, 
        height=HEIGHT+40,
        background_color="#000000",
        resizable=False, 
        zoomable=True,
        confirm_close=True, 
        text_select=True,
        frameless=True, 
        easy_drag=False
    )
    
    webview.start(
        load_website, 
        window,
        private_mode=False, # Save cookies, local storage and cache
        debug=DEBUG_MODE, # Show developer tools
        storage_path=CACHE
    )

html = """
<html>
    <style>
        body {
            background-color: get_theme();
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        p {
            color: #333;
            font-size: 16px;
            font-family: sans-serif;
        }
    
    @keyframes spinner {
        to {transform: rotate(360deg);}
    }
    
    .spinner:before {
        content: '';
        box-sizing: border-box;
        position: absolute;
        top: 50%;
        left: 50%;
        width: 20px;
        height: 20px;
        margin-top: 20px;
        margin-left: -10px;
        border-radius: 50%;
        border-top: 2px solid #333;
        border-right: 2px solid transparent;
        animation: spinner .6s linear infinite;
    }

    </style>
    <body class="pywebview-drag-region">
        <div style="flex; justify-content: center; align-items: center;">
            <p>Please wait while Pyro Axis AI Training Dashboard loads</p>
            <div class="spinner"></div>
        </div>
    </body>
</html>"""