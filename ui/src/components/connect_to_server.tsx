import { connect } from 'http2';
import { useEffect, useState } from 'react';
import { toast } from 'sonner';

export async function AttemptServerConnection(webserver_url: string) {
  let newUrl = "";
  let connected = true;
  let ip = "";
  let error = null;

  try {;
    let reponse = await fetch(webserver_url);
    if (!reponse.ok) {
      throw new Error("Response was not ok to URL: " + webserver_url + " (Response Status: " + reponse.status + ")");
    }
    let data = await reponse.json();
    if (data.status !== "ok") {
      throw new Error("Failed to get data from server: " + data.traceback);
    }

    connected = true;
    newUrl = data.url;
    ip = data.ip;

  } catch (error : any) {
    connected = false;
    newUrl = webserver_url;
    ip = "localhost";
    error = error.message;
  }

  return { connected, newUrl, ip, error };
}
