// Utilities
import { GetModelsFromServer, SendTrainingRequest, GetModelStatuses } from "@/components/webserver";
import { useEffect, useRef, useState, useCallback, memo } from "react";
import { useRouter } from "next/router";
import { toast } from "sonner";

// UI
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Dialog, DialogContent, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis, Legend } from "recharts";
import { Plus, ArrowDownToLine, X, Image, Text, Flame, ScrollText  } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Loading } from "@/components/loading";
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card";
import ErrorPopup from "@/components/error_popup";

class ProgressBar {
  title: string;
  tooltip: string | null;
  current: number;
  total: number;
  progress_text: string;

  constructor(data: any) {
    this.title = data["title"];
    this.tooltip = data["tooltip"];
    this.current = data["current"];
    this.total = data["total"];
    this.progress_text = data["progress_text"];
  }
}

class GraphData {
  title: string;
  color: string;
  values: number[];

  constructor(data: any) {
    this.title = data["title"];
    this.color = data["color"];
    this.values = data["values"];
  }
}
class Graph {
  title: string;
  tooltip: string | null;
  lines: GraphData[];

  constructor(data: any) {
    this.title = data["title"];
    this.tooltip = data["tooltip"];
    this.lines = [];
    for (let i = 0; i < data["lines"].length; i++) {
      this.lines.push(new GraphData(data["lines"][i]))
    }
  }
}

class DropdownData {
  title: string;
  value: string;
  tooltip: string | null;

  constructor(title: string, value: string, tooltip: string | null) {
    this.title = title;
    this.value = value;
    this.tooltip = tooltip;
  }
}
class Dropdown {
  title: string;
  tooltip: string | null;
  data: DropdownData[];

  constructor(data: any) {
    this.title = data["title"];
    this.tooltip = data["tooltip"];
    this.data = []
    for (let i = 0; i < data["data"].length; i++) {
      this.data.push(new DropdownData(data["data"][i]["title"], data["data"][i]["value"], data["data"][i]["tooltip"]))
    }
  }
}

class TraingModel {
  type: string;
  data_type: string;
  status: string;
  epoch: number;
  epochs: number;
  estimated_time: number | string;
  progress_bars: ProgressBar[];
  graphs: Graph[];
  dropdowns: Dropdown[];

  constructor(data: any) {
    this.type = data["type"];
    this.data_type = data["data_type"];
    this.status = data["status"];
    this.epoch = data["epoch"];
    this.epochs = data["epochs"];
    this.estimated_time = data["estimated_time"];
    this.progress_bars = [];
    this.graphs = [];
    this.dropdowns = [];
    for (let i = 0; i < data["progress_bars"].length; i++) {
      this.progress_bars.push(new ProgressBar(data["progress_bars"][i]))
    }
    for (let i = 0; i < data["graphs"].length; i++) {
      this.graphs.push(new Graph(data["graphs"][i]))
    }
    for (let i = 0; i < data["dropdowns"].length; i++) {
      this.dropdowns.push(new Dropdown(data["dropdowns"][i]))
    }
  }
}

class ModelTypeHyperparameter {
  name: string;
  default: string | number | boolean;
  min_value: number | null;
  max_value: number | null;
  incriment: number | null;
  special_type: string | null;
  options: string[] | null;
  description: string | null;

  constructor(data: any) {
    this.name = data["name"];
    this.default = data["default"];
    this.min_value = data["min_value"] ?? null;
    this.max_value = data["max_value"] ?? null;
    this.incriment = data["incriment"] ?? null;
    this.special_type = data["special_type"] ?? null;
    this.options = data["options"] ?? null;
    this.description = data["description"] ?? null;
  }
}

class CompletedHyperparameter {
  name: string;
  value: string | number | boolean;

  constructor(data: any) {
    this.name = data["name"];
    this.value = data["value"];
  }

  ToDict() {
    return {
      "name": this.name,
      "value": this.value
    }
  }
}

class ModelType {
  name: string;
  description: string;
  data_type: string;
  hyperparameters: ModelTypeHyperparameter[];

  constructor(data: any) {
    this.name = data["name"];
    this.description = data["description"];
    this.data_type = data["data_type"];
    this.hyperparameters = [];
    for (let i = 0; i < data["hyperparameters"].length; i++) {
      this.hyperparameters.push(new ModelTypeHyperparameter(data["hyperparameters"][i]))
    }
  }
}

class ErrorPopupData {
  error: string;
  traceback: string;

  constructor(error: string, traceback: string) {
    this.error = error;
    this.traceback = traceback;
  }
}

const getGreeting = () => {
  const time = new Date();
  const currentHourInt = time.getHours();
  const currentHour = String(time.getHours()).padStart(2, '0');
  const currentMinute = String(time.getMinutes()).padStart(2, '0');

  if (currentHourInt < 12) {
    return `Good Morning! ${currentHour}:${currentMinute}`;
  } else if (currentHourInt < 18) {
    return `Good Afternoon! ${currentHour}:${currentMinute}`;
  } else {
    return `Good Evening! ${currentHour}:${currentMinute}`;
  }
};

function ConvertToProgressValue(value: number, max: number): number {
  return (value / max) * 100
}

export default function Index() {
  const { push } = useRouter();
  const [greeting, setGreeting] = useState("Loading...");
  const [showLoading, setShowLoading] = useState(false);
  const [models, setModels] = useState<TraingModel[]>([])
  const [current_model_index, set_current_model_index] = useState(-1);
  const [error_popup_data, setErrorPopupData] = useState<ErrorPopupData | null>(null);
  const sidebarScrollRef = useRef(null);
  const [creating_model, setCreatingModel] = useState(false);
  const [lastMousePosition, setLastMousePosition] = useState({ x: 0, y: 0 });
  const [windowPosition, setWindowPosition] = useState({ x: 0, y: 0 });
  const [isMouseInDragArea, setIsMouseInDragArea] = useState(false);
  const offsetRef = useRef({ x: 0, y: 0 });
  const draggingRef = useRef(false);

  useEffect(() => {
    const initialWindowPosition = { x: window.screenX || 0, y: window.screenY || 0 };
    setWindowPosition(initialWindowPosition);
  }, []);

  const handleMouseDown = useCallback((e: MouseEvent) => {
    if (e.clientY <= 40) {
      setIsMouseInDragArea(true);
      draggingRef.current = true;
      // Calculate offset between the mouse and the top-left corner of the window
      offsetRef.current = {
        x: e.screenX - windowPosition.x,
        y: e.screenY - windowPosition.y,
      };
      setLastMousePosition({ x: e.screenX, y: e.screenY });
    }
  }, [windowPosition]);

  const handleMouseUp = useCallback(() => {
    draggingRef.current = false;
    setIsMouseInDragArea(false);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (draggingRef.current) {
        const newX = e.screenX - offsetRef.current.x;
        const newY = e.screenY - offsetRef.current.y;
        setWindowPosition({ x: newX, y: newY });

        // @ts-ignore
        window.pywebview._bridge.call('pywebviewMoveWindow', [newX, newY], 'move');
      }
  }, []);

  useEffect(() => {
    window.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      window.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, [handleMouseDown, handleMouseUp, handleMouseMove]);
  
  useEffect(() => {
    setGreeting(getGreeting());
  
    const interval = setInterval(() => {
      if (current_model_index === -1) setGreeting(getGreeting());
    }, 30000);
  
    return () => clearInterval(interval);
  }, [current_model_index]);

  useEffect(() => {
    const interval = setInterval(() => {
      async function UpdateModels() {
        let data = await GetModelStatuses();

        if (data.error ?? false) {
          setErrorPopupData(new ErrorPopupData(data.error, data.traceback))
          return
        }
        
        data = data.data
        const updated_data : TraingModel[] = []

        if (Array.isArray(data) && data.forEach) {
          try { 
            data.forEach(function UpdateData(entry : any) {
              updated_data.push(new TraingModel(entry))
            })
            if (JSON.stringify(updated_data) !== JSON.stringify(models)) {
              setModels(updated_data);
            }
          } catch (error) {
            console.log("Failed to update models:", error);
            setModels([])
          }
        } else {
          console.log("Expected an array, but got:", data);
          setModels([])
        }
      }

      UpdateModels();
    }, 7500);
    return () => {
      clearInterval(interval);
    };
  }, []);

  async function GetModels() {
    try {
      toast.loading("Retrieving models...");

      const model_data_dict = await GetModelsFromServer();
      const model_data = model_data_dict.map((model_data: any) => new ModelType(model_data));
      
      toast.dismiss();

      if (!model_data) {
        toast.error("Failed to retrieve models. Check the console for more info.");
        return;
      } else {
        toast.success("Successfully retrieved models.");
        return model_data;
      }
    } catch (error) {
      toast.error("Failed to retrieve models. Check the console for more info.");
      console.error(error);
    }
  }

  function CreateNewModel() {
    const [modelsData, setModelsData] = useState<ModelType[] | null>(null);
    const [completed_hyps, setCompletedHyps] = useState<CompletedHyperparameter[]>([]);
    const [model_selector_open, setModelSelectorOpen] = useState<boolean>(true);
    const [set_hyp_selector_open, setHypSelectorOpen] = useState<boolean>(false);
    const [model_hyp_index, setModelHypIndex] = useState<string>("-1"); // Stringified number index (for selector key)
    const [model_hyp_index_int, setModelHypIndexInt] = useState<number>(-1); // Unstringified number index

    useEffect(() => {
      const fetchData = async () => {
        const data = await GetModels();
        if (data) {
          data.forEach((model: any) => {
            model.hyperparameters.forEach((hyperparameter: any) => {
              setCompletedHyps((prevCompletedHyps) => [...prevCompletedHyps, new CompletedHyperparameter({
                "name": hyperparameter.name,
                "value": hyperparameter.default
              })]);
            });
          });
          setModelsData(data);
        } else {
          toast.error("Failed to retrieve models. Check the console for more info.");
        }
      };
    
      fetchData();
    }, []);

    function ChangeHypValue(index: number, value: any) {
      if (!modelsData) return;
      setCompletedHyps((prevCompletedHyps) => {
        const newHyps = [...prevCompletedHyps];
        newHyps[index].value = value;
        return newHyps;
      });
    }

    function HandleModelSelection() {
      if (model_hyp_index === "-1") {
        toast.error("Please select a model type.")
        return;
      }

      setModelSelectorOpen(false);
      setModelHypIndexInt(Number(model_hyp_index));
      setHypSelectorOpen(true);
    }

    function HandleHypSelection() {
      if (!modelsData) return;
      setHypSelectorOpen(false);
      let hyp_dict: any = []
      completed_hyps.forEach((hyp) => {
        hyp_dict.push({
          "name": hyp.name,
          "value": hyp.value
        })
      })
      SendTrainingRequest(hyp_dict, model_hyp_index_int);
      setCreatingModel(false);
    }

    if (!modelsData) {
      toast.loading("Retrieving models...");
    }
    else if (model_selector_open) {
      return (
        <div>
          <Dialog open={model_selector_open} onOpenChange={setModelSelectorOpen}>
            <DialogContent aria-describedby={undefined}>
              <DialogTitle className="font-bold text-2xl">Model Selector</DialogTitle>
              <DialogDescription className="text-md">Select the type of model you would like to train.</DialogDescription>
              {modelsData ? (
                <div>
                  <Select value={model_hyp_index} onValueChange={setModelHypIndex}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select a model type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {modelsData.map((model, index) => (
                          <SelectItem key={index} value={String(index)}>
                            <h1 className="text-md font-bold">{model.name}</h1>
                            <p className="text-zinc-500 text-sm">{model.description}</p>
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                  <Button onClick={HandleModelSelection} className="mt-4 w-full">Conitnue</Button>
                </div>
              ) : (
                <p>Loading...</p>
              )}
            </DialogContent>
          </Dialog>
        </div>
      );
    } else if (set_hyp_selector_open) {
      return (
        <div>
          <Dialog open={set_hyp_selector_open}>
            <DialogContent aria-describedby={undefined}>
              <DialogTitle className="font-bold text-2xl">Hyperparameter Selector</DialogTitle>
              <DialogDescription className="text-md">Select the hyperparameters you would like to train your {modelsData[model_hyp_index_int].name} model with.</DialogDescription>
              <div className="flex flex-col h-[400px] overflow-y-scroll">
                {modelsData[model_hyp_index_int].hyperparameters.map((param, index) => (
                    <div key={index} className="parameter-form">
                      <h1 className="text-md font-bold">{param.name}</h1>
                      <p className="text-zinc-500 text-sm mb-2">{param.description}</p>
                      {param.special_type === null && (
                        <>
                          {typeof completed_hyps[index].value === 'number' && param.special_type === null && (
                            <Input type="number" 
                              value={completed_hyps[index].value} 
                              min={param.min_value ? param.min_value : -Infinity}
                              max={param.max_value ? param.max_value : Infinity} 
                              step={param.incriment ? param.incriment : 1}
                              onChange={(e) => ChangeHypValue(index, Number(e.target.value))}
                            />
                          )}
                          {typeof completed_hyps[index].value === 'string' && param.special_type === null && (
                            <Input type="text" value={completed_hyps[index].value} onChange={(e) => ChangeHypValue(index, e.target.value)} />
                          )}
                          {typeof completed_hyps[index].value === 'boolean' && param.special_type === null && (
                            <Select defaultValue={completed_hyps[index].value === true ? "Yes" : "No"} onValueChange={(value) => ChangeHypValue(index, value === "Yes" ? true : false)}>
                              <SelectTrigger className="w-full">
                                <SelectValue placeholder="Select an option" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="Yes">Yes</SelectItem>
                                <SelectItem value="No">No</SelectItem>
                              </SelectContent>
                            </Select>
                          )}
                        </>
                      )}
            
                      {param.special_type === 'path' && typeof completed_hyps[index].value === 'string' && (
                        <Input type="text" value={completed_hyps[index].value} onChange={(e) => ChangeHypValue(index, e.target.value)} />
                      )}
            
                      {param.special_type === 'dropdown' && typeof completed_hyps[index].value === 'string' && param.options && (
                        <Select value={completed_hyps[index].value} onValueChange={(value) => ChangeHypValue(index, value)}>
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select an option" />
                          </SelectTrigger>
                          <SelectContent>
                            {param.options.map((option, index) => (
                              <SelectItem key={index} value={option}>
                                {option}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      )}
                      <Separator orientation="horizontal" className="bg-zinc-600 w-full mt-4 mb-3" />
                    </div>            
                ))}
              </div>
              <Button className="mt-4 w-full" onClick={HandleHypSelection}>Start Training</Button>
            </DialogContent>
          </Dialog>
        </div>
      );
    }
  }

  const Sidebar = () => {
    return (
      <Card className="bg-black w-72 h-[calc(100vh-25px)] flex flex-col">
        <div className="bg-black pt-4 pb-2 flex flex-col">
          <div className="flex rounded-lg hover:bg-zinc-800 mb-3 py-1 w-[calc(100%-30px)] ml-[15px]" onClick={() => set_current_model_index(-1)}>
            <Flame className="h-14 w-14" color="#eb4b2b"/>
            <div className="flex flex-col pl-4">
              <h1 className="text-white text-2xl font-bold">Pyro Axis AI</h1>
              <h2 className="text-white text-sm font-small">Training Dashboard</h2>
            </div>
          </div>
          <Separator orientation="horizontal" className="bg-zinc-600 w-[calc(100%-30px)] ml-[15px]" />
        </div>
        <div className="bg-black flex flex-col flex-grow overflow-y-auto" ref={sidebarScrollRef}>
          {models.length === 0 && 
            <div>
              <p className="text-center mt-4 w-[calc(100%-30px)] ml-[15px] font-bold text-lg">No models found</p>
              <p className="text-center mt-1 w-[calc(100%-30px)] ml-[15px] text-sm text-zinc-500">Train a new model to begin</p>
            </div>
          }
          {models.map((model, index) => (
            <div key={index} className="relative w-[calc(100%-30px)] ml-[15px] h-auto p-4 mt-2 bg-zinc-900 rounded-lg hover:bg-zinc-800"
              onClick={() => set_current_model_index(index)}>
              <div className="items-center gap-3 flex">
                {model.data_type === "image" ? (
                  <Image width={24} height={24} />
                ) : (
                  <Text width={24} height={24} />
                )}
                <p className="text-sm font-semibold">{model.type}</p>
              </div>
              <div className="mt-2 overflow-hidden text-xs text-gray-400">
                <div>Status: {model.status}</div>
                <div>Progress: {(model.epoch / model.epochs * 100).toFixed(1)}%</div>
              </div>
            </div>
          ))}
        </div>
        <div className="bg-black flex flex-col mb-4">
          <Separator orientation="horizontal" className="bg-zinc-600 w-[calc(100%-30px)] ml-[15px] my-4" />
          <div className="flex flex-col gap-2 w-[calc(100%-30px)] ml-[15px]">
            <Button variant="outline" className="w-full bg-zinc-900 hover:bg-zinc-800" onClick={() => setCreatingModel(true)}><Plus />Train Model</Button>
            <Button variant="outline" className="w-full bg-zinc-900 hover:bg-zinc-800"><X />Stop Training</Button>
            <Button variant="outline" className="w-full bg-zinc-900 hover:bg-zinc-800"><ArrowDownToLine />Stop Training and Save</Button>
            <Button variant="outline" className="w-full bg-zinc-900 hover:bg-zinc-800" onClick={() => push("/docs")}><ScrollText />Documentation</Button>
          </div>
        </div>
      </Card>
    );
  };

  const Home = () => {
    return (
      <Card className="flex flex-col gap-2 items-center justify-center ml-3 h-[calc(100vh-25px)] w-[calc(100vw-324px)] bg-black">
        <h1 className="text-2xl font-bold">{greeting}</h1>
        <p className="text-zinc-500 text-md">Welcome to the training dashboard!</p>
      </Card>
    );
  };
  
  const ModelVisualizer = () => {
    const model = models[current_model_index];

    const chart_configs : ChartConfig[] = []
    for (let i = 0; i < model.graphs.length; i++) {
      const chart_config_i = {} satisfies ChartConfig
      for (let j = 0; j < model.graphs[i].lines.length; j++) {
        // @ts-ignore
        chart_config_i[j] = { label: model.graphs[i].lines[j].title, color: model.graphs[i].lines[j].color }
      }
      chart_configs.push(chart_config_i)
    }

    const graphs_data : any = []
    for (let i = 0; i < model.graphs.length; i++) {
      const graph_data = []
      for (let j = 0; j < model.graphs[i].lines[0].values.length; j++) {
        const values = {"Epoch": j}
        for (let k = 0; k < model.graphs[i].lines.length; k++) {
          // @ts-ignore
          values[model.graphs[i].lines[k].title] = model.graphs[i].lines[k].values[j]
        }
        graph_data.push(values)
      }
      graphs_data.push(graph_data)
    }

    return (
      <Card className="flex flex-col ml-3 h-[calc(100vh-25px)] w-[calc(100vw-324px)] bg-black overflow-y-auto">
        <div className="m-4 overflow-y-auto overflow-x-hidden">
          <h1 className="text-3xl font-bold mb-4">{model.type}</h1>
          <p className="text-sm">
            {model.status} • {model.estimated_time} remaining
          </p>
    
          {error_popup_data ? (
            <div className="flex flex-col">
              <p className="text-sm text-zinc-500 mt-3">{error_popup_data.error}</p>
              <div className="p-3 rounded-md w-full overflow-auto bg-zinc-800 mt-2">
                <code>
                  {error_popup_data.traceback.split("\n").map((line, index) => <p key={index}>{line}</p>)}
                </code>
              </div>
            </div>
          ) : model.status === "Initializing" ? (
            <div className="flex items-center justify-center w-full h-full">
              <p className="text-sm text-zinc-500">Model is initializing, please wait...</p>
            </div>
          ) : model.status === "Training" ? (
            <div>
              {model.progress_bars.map((progress_bar, index) => (
                <div className="flex flex-col gap-3 mt-8" key={index}>
                  <h2 className="text-xl font-bold mb-1">{progress_bar.title}</h2>
                  <Progress
                    value={ConvertToProgressValue(progress_bar.current, progress_bar.total)}
                    className="h-2 w-[calc(100vw-380px)]"
                  />
                  <p>{progress_bar.progress_text}</p>
                </div>
              ))}
    
              <Separator orientation="horizontal" className="bg-zinc-600 w-[calc(100%-30px)] ml-[15px] mt-8" />
              {model.graphs.map((graph, index) => (
                <Card className="w-[calc(100vw-370px)] h-[640px] mt-8">
                  <h1 className="text-center mt-1 text-lg text-zinc-400">{graph.title}</h1>
                  <ChartContainer config={chart_configs[index]} className="h-[600px] w-[calc(100vw-380px)] mt-2">
                    <AreaChart accessibilityLayer data={graphs_data[index]}>
                      <CartesianGrid vertical={true} />
                      <XAxis  
                        dataKey="Epoch" 
                        tickLine={true} 
                        axisLine={false} 
                        tickCount={5}
                        label={{ value: "Epoch", position: "bottom", offset: -5 }}
                      />
                      <YAxis 
                        tickLine={true} 
                        axisLine={false} 
                        tickCount={5}
                        label={{ value: "Loss", angle: -90, position: "insideLeft", offset: 10 }}
                      />
                      <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                      <Legend layout="horizontal" verticalAlign="bottom" align="left" wrapperStyle={{ marginLeft: '60px', marginBottom: '5px' }}/>
                      <defs>
                        {graph.lines.map((line, index) => (
                          <linearGradient key={`fill${index}`} id={`fill${index}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={line.color} stopOpacity={0.8} />
                            <stop offset="60%" stopColor={line.color} stopOpacity={0.3} />
                            <stop offset="100%" stopColor={line.color} stopOpacity={0.05} />
                          </linearGradient>
                        ))}
                      </defs>
                      {graph.lines.map((line, index) => (
                        <Area 
                          dataKey={line.title} 
                          type="natural" 
                          fill={`url(#fill${index})`}
                          stroke={line.color} 
                          stackId="a" 
                        />
                      ))}
                    </AreaChart>
                  </ChartContainer>
                </Card>
              ))}
              <Separator orientation="horizontal" className="bg-zinc-600 w-[calc(100%-30px)] ml-[15px] my-8" />
    
              <h1 className="text-2xl font-bold mb-4 ml-1">Additional information:</h1>
              <div className="w-[calc(100%-20px)] ml-[5px]">
                {model.dropdowns.map((dropdown, index) => (
                  <Accordion type="single" collapsible key={index}>
                    <AccordionItem value={`item-${index}`}>
                      <AccordionTrigger className="text-lg font-bold">{dropdown.title}</AccordionTrigger>
                      <AccordionContent>
                        {Object.entries(dropdown.data).map(([key, data], dataIndex) => (
                          <div key={dataIndex} className="flex flex-row w-full justify-between mb-6">
                            <p className="text-md font-bold">{data.title}</p>
                            <p className="text-md mr-4 text-zinc-500">
                              {typeof data.value === "boolean" ? (data.value ? "Yes" : "No") : data.value}
                            </p>
                          </div>
                        ))}
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </Card>
    );    
  };

  return (
    <div className="flex flex-row">
      {error_popup_data && <ErrorPopup error={error_popup_data.error} traceback={error_popup_data.traceback} />}
      {showLoading ? (
        <Loading loading_text="Retrieving models..." fullscreen />
      ) : (
        <>
          <Sidebar />
          {current_model_index === -1 ? <Home /> : <ModelVisualizer/>}
          {creating_model && <CreateNewModel />}
        </>
      )}
    </div>
  );
}