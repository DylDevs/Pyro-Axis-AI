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

export async function GetModelsFromServer(webserver_url: string) {
  let reponse = await fetch(webserver_url + "/get_models");
  if (!reponse.ok) {
    throw new Error("Response was not ok to URL: " + webserver_url + " (Response Status: " + reponse.status + ")");
  }
  let data = await reponse.json();
  if (data.status !== "ok") {
    throw new Error("Failed to get data from server: " + data.traceback);
  }
  return data.models;
}

export async function SendTrainingRequest(webserver_url: string, hyperparameters : any, model_index : number) {
  let reponse = await fetch(webserver_url + "/train", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      "model_index": model_index,
      "hyperparameters": hyperparameters
    }),
  });
  if (!reponse.ok) {
    throw new Error("Response was not ok to URL: " + webserver_url + " (Response Status: " + reponse.status + ")");
  }
  let data = await reponse.json();
  if (data.status !== "ok") {
    throw new Error("Failed to get data from server: " + data.traceback);
  }
  return data;
}

export async function GetModelStatuses(webserver_url: string) {
  let reponse = await fetch(webserver_url + "/status");
  if (!reponse.ok) {
    throw new Error("Response was not ok to URL: " + webserver_url + " (Response Status: " + reponse.status + ")");
  }
  let data = await reponse.json();
  if (data.status !== "ok") {
    throw new Error("Failed to get data from server: " + data.traceback);
  }
  return data.data;
}