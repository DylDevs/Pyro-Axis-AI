import type { AppProps } from 'next/app';
import { ThemeProvider } from '@/components/theme_provider';
import { Loading } from '@/components/loading';
import { useEffect, useState, useCallback } from 'react'; 
import { toast, Toaster } from 'sonner';
import { Metadata } from 'next';
import { AttemptServerConnection } from '@/components/connect_to_server';
import '@/styles/globals.css';

// @ts-ignore | Prevents module not found error from js-cookie, even though it is installed
import Cookies from 'js-cookie';

export const metadata: Metadata = {
    title: "Torch AI Training",
    description: "Frontend UI for Torch AI Training",
    icons: ["favicon.ico"],
};

function MyApp({ Component, pageProps }: AppProps) {
  const clear_cache = false;
  if (clear_cache) {
    Cookies.remove('ip');
    Cookies.remove('webserver_url');
    Cookies.remove('frontend_url');
    Cookies.remove('connected');
  }
  const [showLoading, setShowLoading] = useState(true);

  // Initialize the cookie with a default value if it does not exist
  if (Cookies.get("connected") === undefined) {
    Cookies.set("connected", "false");
  }

  let ip = Cookies.get("ip") ?? "localhost";
  if (ip !== "localhost") {
    console.log("IP extracted from cookie: " + ip);
  } else {
    console.log("Using default IP: " + ip);
  }
  let frontend_url = `http://${ip}:3000`;
  let webserver_url = `http://${ip}:8000`;

  const setupConnection = async () => {
    try {
      let { connected, newUrl, error, ip } = await AttemptServerConnection(webserver_url);
      if (connected) {
        webserver_url = newUrl;
        frontend_url = `http://${ip}:3000`;
        Cookies.set("ip", ip);
        Cookies.set("connected", "true");
      } else {
        Cookies.set("ip", "localhost");
        Cookies.set("connected", "false");
      }

      Cookies.set("webserver_url", webserver_url);
      Cookies.set("frontend_url", frontend_url);

      if (!connected) {
        throw new Error("Failed to connect to server");
      }
    } catch (error) {
      throw error;
    }
  };

  useEffect(() => {
    toast.promise(
      new Promise<void>(async (resolve, reject) => {
        try {
          // await setupConnection();
          setTimeout(() => {
            setShowLoading(false);
            resolve();
          }, 5000);
          console.log("Connected to training server at " + webserver_url);
        } catch (error) {
          reject(error);
          console.log(error);
          setShowLoading(false);
        }
      }),
      {
        loading: "Connecting to server...",
        success: "Connected to server!",
        error: "Failed to connect to server",
      }
    );
  }, []);

  return (
    <ThemeProvider defaultTheme="dark" attribute="class">
      <div className="flex flex-row gap-3 m-3">
        {showLoading && <Loading loading_text="Connecting to server..." />}
        <div style={{ display: showLoading ? "none" : "block" }} className='overflow-hidden'>
          <Component {...pageProps} />
        </div>
      </div>
    </ThemeProvider>
  );
}

export default MyApp;
