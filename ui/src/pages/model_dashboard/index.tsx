import React, { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Loading } from "@/components/loading";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts';
const data = [{name: '1', t_loss: 3.2, v_loss: 4.2}, 
            {name: "2", t_loss: 4.2, v_loss: 5.2},
            {name: "2", t_loss: 5.2, v_loss: 6.2},
            {name: "3", t_loss: 3.6, v_loss: 4.6},
            {name: "4", t_loss: 2.3, v_loss: 1.5},
            {name: "5", t_loss: 3.2, v_loss: 4.2},
            {name: "6", t_loss: 4.2, v_loss: 5.2},
            {name: "7", t_loss: 5.2, v_loss: 6.2},
            {name: "8", t_loss: 0.2, v_loss: 1.1},
            {name: "9", t_loss: 3.2, v_loss: 4.2},
            {name: "10", t_loss: 0.5, v_loss: 0.1}];

// @ts-ignore | Prevents module not found error from js-cookie, even though it is installed
import Cookies from 'js-cookie';

// @ts-ignore | Hide errors from variables
function CustomTooltip({ payload, label, active }) {
    if (active && payload[0] !== undefined && payload[1] !== undefined) {
        return (
            <div className="custom-tooltip">
                <Card className="p-4">
                    <p>{`Training Loss: ${payload[0].value}`}</p>
                    <p>{`Validation Loss: ${payload[1].value}`}</p>
                </Card>
            </div>
        );
    }
    return null;
}

export default function ModelDashboard() {
    const { push } = useRouter();
    const [retrievingInitialData, setRetrievingInitialData] = useState(false);
    const [training_data, setTrainingData] = useState<any>();

    const [selected_model, setSelectedModel] = useState(-1);
    const [last_selected_model, setLastSelectedModel] = useState(-1);
    const [update_cooldown, setUpdateCooldown] = useState(5);
    const [timeUntilUpdate, setTimeUntilUpdate] = useState(update_cooldown);
    const [timeUntilUpdateText, setTimeUntilUpdateText] = useState("5 seconds");
    const [update_data, setUpdateData] = useState(false);
    const [getting_data, setGettingData] = useState(false);
    const [chart_data, setChartData] = useState<any>([]);

    const [data_card_size, setDataCardSize] = useState<number>(0);

    const webserver_url = Cookies.get("webserver_url") || "http://localhost:8000";
    const connected = Cookies.get("connected") === "true";

    useEffect(() => {
        const updateSize = () => {
            setDataCardSize(window.innerWidth * 0.8);
        }
        window.addEventListener('resize', updateSize);
        updateSize();
        return () => window.removeEventListener('resize', updateSize);
    }, []);

    useEffect(() => {
        const intervalId = setInterval(() => {
          setTimeUntilUpdate(prev => {
            // When countdown is finished, execute the data fetch and reset the countdown
            if (prev <= 100) {
              if (update_data) {
               getData(true);
              }
              return 5000;  // Reset countdown to 5 seconds
            }
      
            // Update the countdown display
            if (!getting_data && update_data) {
              setTimeUntilUpdateText(`Updating in ${(prev - 100) / 1000} seconds`);
            } else {
              setTimeUntilUpdateText(`Updating data is disabled`);
            }
      
            // Decrement the countdown timer
            return prev - 100;
          });
        }, 100);
      
        // Cleanup function to clear interval on component unmount
        return () => clearInterval(intervalId);
      }, [update_data, getting_data]);  // Dependencies for useEffect

    const getData = async (change_update_text : boolean = false) => {
        try {
            setGettingData(true);
            if (change_update_text) {
                setTimeUntilUpdateText("Updating data...");
            }

            const response = await fetch(`${webserver_url}/models`, {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                }
            });

            if (!response.ok) {
                throw new Error("Network response was not ok to URL: " + webserver_url + " (Response Status: " + response.status + ")");
            }

            const data = await response.json();
            if (data.status !== "ok") {
                throw new Error("Failed to get initial model training data! Full traceback: " + data.traceback);
            }

            if (selected_model !== -1) {
                // Add the new data point
                const newEntry = {
                    name: null,
                    t_loss: data.training_data[selected_model].training_loss,
                    v_loss: data.training_data[selected_model].val_loss
                };
            
                // Append the new entry and then clip the array to the last 10 entries
                const updatedChartData = [...chart_data, newEntry].slice(-10);

                updatedChartData.forEach((entry, index) => {
                    entry.name = index + 1;
                });
            
                // Update the state with the clipped array
                setChartData(updatedChartData);
            }

            setTrainingData(data.training_data);
            setGettingData(false);
            if (change_update_text) {
                setTimeUntilUpdateText("Updating in 5 seconds");
            }
        } catch (error) {
            console.error("Error:", error);
            throw error;
        }
    };

    useEffect(() => {
        setRetrievingInitialData(true);

        toast.promise(
            new Promise(async (resolve, reject) => {
                try {
                    await getData()
                    setTimeout(() => {
                        setRetrievingInitialData(false);
                    }, 3000);
                    resolve(0);
                } catch (error) {
                    console.log(error);
                    reject();
                    setRetrievingInitialData(false);
                }
            }),
            {
                loading: "Retrieving initial model training data...",
                success: "Retrieved initial model training data!",
                error: "Failed to connect to training server!",
            }
        );
    }, []); // Empty dependency array to run only once on mount

    function HandleSelectedModelChange(index : number) {
        setChartData([]);
        setSelectedModel(index);
    }

    if (retrievingInitialData) {
        return (
            <Loading loading_text="Retrieving initial model training data..." />
        );
    }

    return (
        <div className="flex flex-row w-full h-[calc(100vh-120px)] space-x-3">
            <div className="flex flex-col w-[20%] space-y-2">
                <Button variant={"secondary"} onClick={() => push("/train_model")}>Train Model</Button>
                <Button variant={"secondary"} onClick={() => push("/saved_model_dashboard")}>Saved Model Dashboard</Button>
                <Card className="flex flex-col w-full h-full overflow-y-auto rounded-t-md">
                    {training_data?.map((element: any, index: number) => (
                        <React.Fragment key={index}>
                            <Button
                                variant={index === selected_model ? "default" : "outline"}
                                className="flex items-center justify-center m-2"
                                onClick={() => HandleSelectedModelChange(index)}
                            >{index} ({element.type})</Button>
                        </React.Fragment>
                    ))}
                </Card>
                <div className="grid grid-cols-2">
                    <p className="m-1">{timeUntilUpdateText}</p>
                    <div className="flex items-center space-x-2 justify-end m-1">
                        <Switch checked={update_data} onCheckedChange={(checked: boolean) => setUpdateData(checked)} />
                        <p>Update Data</p>
                    </div>
                </div>
            </div>
            <Card className="flex flex-col w-[80%] h-[calc(100vh-120px)] space-y-5 pb-0 overflow-auto rounded-t-md">
                {!connected && (
                    <div className="flex flex-col items-center justify-center h-full space-y-2">
                        <h1 className="text-3xl font-bold">Not connected to training server!</h1>
                        <p className="text-zinc-500">Please connect to the server and try again.</p>
                    </div>
                )}
                {training_data === undefined && connected && (
                    <div className="flex flex-col items-center justify-center h-full space-y-2">
                        <h1 className="text-3xl font-bold">No model training data available!</h1>
                        <p className="text-zinc-500">Failed to retrieve model training data.</p>
                    </div>
                )}
                {Array.isArray(training_data) && training_data.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full space-y-2">
                        <h1 className="text-3xl font-bold">No model training data available!</h1>
                        <p className="text-zinc-500">Train a model to get started.</p>
                    </div>
                )}
                {selected_model === -1 && training_data !== undefined && connected && (
                    <div className="flex flex-col items-center justify-center h-full space-y-2">
                        <h1 className="text-3xl font-bold">No model selected</h1>
                        <p className="text-zinc-500">Select a model on the left bar to get started.</p>
                    </div>
                )}
                {training_data !== undefined && selected_model !== -1 && connected && (
                    <div>
                        <h1 className="text-3xl font-bold m-4">{training_data[selected_model].status}</h1>
                        {chart_data.length > 0 ? (
                            <LineChart width={data_card_size - 50} height={500} data={chart_data}>
                                <Line type="monotone" dataKey="t_loss" stroke="#8884d8" />
                                <Line type="monotone" dataKey="v_loss" stroke="#82ca9d" />
                                <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
                                <XAxis dataKey="name" />
                                <YAxis />
                                {/* @ts-ignore Hide tooltip error*/}
                                <Tooltip content={<CustomTooltip />} />
                                <Legend />
                            </LineChart>
                        ) : (
                            update_data === true ? (
                                <div className={`min-w-${data_card_size - 50} min-h-500 m-10 text-center justify-center`}>
                                    <h1 className="text-3xl font-bold">Please Wait...</h1>
                                    <p className="text-zinc-500">The data for the graph is being fetched</p>
                                </div>
                            ) : (
                                <div className={`min-w-${data_card_size - 50} min-h-500 m-10 text-center justify-center`}>
                                    <h1 className="text-3xl font-bold">No Data Available</h1>
                                    <p className="text-zinc-500">The data for the graph is not available. Enable the update data option.</p>
                                </div>
                            )
                        )}
                        <Separator orientation="horizontal" className="w-[calc(100vw-475px)] " />
                        <div className="grid grid-cols-3 m-4">
                            {Object.keys(training_data[selected_model]).map((key) => {
                                if (key == "exception") { return null }
                                const value = training_data[selected_model][key];
                                const newKey = key.replace(/_./g, (x) => ' ' + x[1].toUpperCase());
                                const finalKey = newKey.charAt(0).toUpperCase() + newKey.slice(1);
                                return (
                                    <div key={key} className="flex flex-col m-2">
                                        <h1 className="text-2xl font-bold">{finalKey}</h1>
                                        <p className="text-zinc-500">
                                            {typeof value === "number" && value % 1 !== 0
                                                ? value.toFixed(4)
                                                : value}
                                        </p>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </Card>
        </div>
    );
}