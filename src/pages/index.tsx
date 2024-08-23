import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useRouter } from "next/router";
import { useState, useEffect } from "react";

// @ts-ignore | Prevents module not found error from js-cookie, even though it is installed
import Cookies from 'js-cookie'; 

function getTime() {
  const now = new Date();
  const hours = now.getHours();
  let timeOfDay;
  if (hours >= 0 && hours < 12) {
    timeOfDay = "Good morning!";
  } else if (hours >= 12 && hours < 17) {
    timeOfDay = "Good afternoon!";
  } else {
    timeOfDay = "Good evening!";
  }
  const localTime = now.toLocaleTimeString([], {timeStyle: 'medium'});
  const localTimeZone = now.toLocaleTimeString([], {timeZoneName: 'short'}).split(' ')[2];
  return [
    `${localTime} ${localTimeZone}`,
    `${timeOfDay}`
  ];
}

export default function Home() {
  const [connected, setConnected] = useState(false);
  const [ip, setIp] = useState("");
  const [newServerIp, setNewServerIp] = useState("");

  useEffect(() => {
    const connectedCookie = Cookies.get("connected");
    setConnected(connectedCookie === "true");
    const ipCookie = Cookies.get("ip");
    setIp(ipCookie || "localhost");
    setNewServerIp(ipCookie || "localhost");
  }, []);

  const router = useRouter();
  const { push } = useRouter();

  function AttemptServerReconnection() {
    Cookies.set("ip", newServerIp);
    Cookies.set("webserver_url", `http://${newServerIp}:8000`);
    Cookies.set("frontend_url", `http://${newServerIp}:3000`);
    router.reload();
  }

  return (
    <Card className="flex flex-col w-full h-[calc(100vh-120px)] space-y-5 pb-0 overflow-auto rounded-t-md">
      <div className="flex flex-col items-center justify-center h-screen space-y-2">
        <h1 className="text-xl font-bold">{getTime()[1]}</h1>
        <div className="flex flex-col items-center justify-center space-y-3">
          <h1 className="text-2xl font-bold">PyTorch AI Training Dashboard</h1>
          <p className="text-zinc-600">Press F1 to open the command menu</p>
          <div className="flex flex-row gap-3">
            <Button variant={"secondary"} onClick={() => push("/train_model")}>Train a Model</Button>
            <Button variant={"secondary"} onClick={() => push("/model_dashboard")}>Model Training Dashboard</Button>
            <Button variant={"secondary"} onClick={() => push("/saved_model_dashboard")}>Manage Trained Models</Button>
          </div>
          {connected ? <p className="text-green-500">Connected to Training Server</p> : <p className="text-red-500">Not Connected to Training Server</p>}
          {!connected && 
          <div className="flex flex-row gap-3">
            <Input placeholder="Enter Training Server IP" value={newServerIp} onChange={(e) => setNewServerIp(e.target.value)}/>
            <Button variant={"secondary"} onClick={AttemptServerReconnection}>Connect</Button>
          </div>
          }
        </div>
      </div>
    </Card>
  );
}
