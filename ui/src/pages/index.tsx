// Utilities
import { GetModelsFromServer, SendTrainingRequest, GetModelStatuses } from "@/components/webserver";
import { useEffect, useRef, useState, useCallback, memo } from "react";
import { toast } from "sonner";

// UI
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Dialog, DialogContent, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis, Legend } from "recharts";
import { Plus, ArrowDownToLine, X, Power, Image, Text, Flame  } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Loading } from "@/components/loading";
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card";

// @ts-ignore | Prevents module not found error from js-cookie, even though it is installed
import Cookies from 'js-cookie';

class Model {
  type: string;
  data_type: string;
  status: string;
  epoch: number;
  epochs: number;
  training_losses : number[];
  val_losses : number[];
  best_epoch: number;
  best_training_loss: number;
  best_val_loss: number;
  elapsed: number;
  estimated_time: number | string;
  time_per_epoch: number;
  model_data : any;
  additional_training_data : any;
  hyperparameters: any;

  constructor(data: any) {
    this.type = data.type;
    this.data_type = data.data_type;
    this.status = data.status;
    this.epoch = data.epoch;
    this.epochs = data.epochs;
    this.training_losses = data.training_losses;
    this.val_losses = data.val_losses;
    this.best_epoch = data.best_epoch;
    this.best_training_loss = data.best_training_loss;
    this.best_val_loss = data.best_val_loss;
    this.elapsed = data.elapsed;
    this.estimated_time = data.estimated_time;
    this.time_per_epoch = data.time_per_epoch;
    this.model_data = data.model_data;
    this.additional_training_data = data.additional_training_data;
    this.hyperparameters = data.hyperparameters;
  }
}

class Hyperparameter {
  name: string;
  value: any;
  min_value: number;
  max_value: number;
  incriment: number;
  special_type: "path" | "dropdown";
  options: string[];
  description: string;

  constructor(name: string, value: any, min_value: number, max_value: number, incriment: number, special_type: "path" | "dropdown", options: string[], description: string) {
    this.name = name;
    this.value = value;
    this.min_value = min_value;
    this.max_value = max_value;
    this.incriment = incriment;
    this.special_type = special_type;
    this.options = options;
    this.description = description;
  }
}

function HypToDict(hyperparameters: Hyperparameter[]) {
  let dict : any = {};
  for (let i = 0; i < hyperparameters.length; i++) {
    dict[hyperparameters[i].name] = hyperparameters[i].value;
  }
  console.log(dict);
  return dict;
}

class ModelData {
  name: string;
  description: string;
  data_type: string;
  hyperparameters: Hyperparameter[];

  constructor(name: string, description: string, data_type: string, hyperparameters: Hyperparameter[]) {
    this.name = name;
    this.description = description;
    this.data_type = data_type;
    this.hyperparameters = hyperparameters;
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

function formatDuration(seconds: number): string {
  if (seconds < 0) return "0 seconds";

  const days = Math.floor(seconds / (24 * 3600));
  seconds %= 24 * 3600;
  const hours = Math.floor(seconds / 3600);
  seconds %= 3600;
  const minutes = Math.floor(seconds / 60);
  seconds %= 60;

  const parts = [];
  if (days > 0) parts.push(`${days} days`);
  if (hours > 0) parts.push(`${hours} hours`);
  if (minutes > 0) parts.push(`${minutes} minutes`);
  if (seconds > 0 || parts.length === 0) parts.push(`${seconds.toFixed(2)} seconds`);

  return parts.join(", ");
}

function ConvertToProgressValue(value: number, max: number): number {
  return (value / max) * 100
}

export default function Index() {
  const [greeting, setGreeting] = useState(getGreeting());
  const [showLoading, setShowLoading] = useState(false);

  const [models, setModels] = useState<Model[]>([])
  const [hovered_model_index, setHovered_model_index] = useState(-1);
  const [current_model_index, set_current_model_index] = useState(-1);

  const sidebarScrollRef = useRef(null);
  const sidebarScrollPosition = useRef(0);

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
    },
    []
  );

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
    const interval = setInterval(() => {
      if (current_model_index === -1) setGreeting(getGreeting());
    }, 30000);
    return () => clearInterval(interval);
  }, [current_model_index]);

  useEffect(() => {
    if (sidebarScrollRef.current) {
      // @ts-ignore
      sidebarScrollRef.current.scrollTop = sidebarScrollPosition.current; // Restore scroll position
    }
  }, [hovered_model_index]); // Runs when hover state changes

  const handleMouseEnter = useCallback((index : number) => {
    // @ts-ignore
    sidebarScrollPosition.current = sidebarScrollRef.current.scrollTop; // Save current scroll position
    setHovered_model_index(index);
  }, []);

  const handleMouseLeave = useCallback(() => {
    // @ts-ignore
    sidebarScrollPosition.current = sidebarScrollRef.current.scrollTop; // Save current scroll position
    setHovered_model_index(-1);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      async function UpdateModels() {
        const data = await GetModelStatuses(Cookies.get("webserver_url") ?? "http://localhost:8000");
        const updated_data : Model[] = []

        if (Array.isArray(data) && data.forEach) {
          try { 
            data.forEach(function UpdateData(entry : any) {
              updated_data.push(new Model(entry))
            })
            if (JSON.stringify(updated_data) !== JSON.stringify(models)) {
              setModels(updated_data);
            }
          } catch (error) {
            console.log("Failed to update models:", error);
          }
        } else {
          console.log("Expected an array, but got:", data);
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

      const webserver_url = Cookies.get("webserver_url") ?? "http://localhost:8000";
      const model_data = await GetModelsFromServer(webserver_url);
      HypToDict(model_data[0].hyperparameters);

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
    const [modelsData, setModelsData] = useState<ModelData[] | null>(null);
    const [model_selector_open, setModelSelectorOpen] = useState<boolean>(true);
    const [set_hyp_selector_open, setHypSelectorOpen] = useState<boolean>(false);
    const [model_hyp_index, setModelHypIndex] = useState<string>("-1"); // Stringified number index (for selector key)
    const [model_hyp_index_int, setModelHypIndexInt] = useState<number>(-1); // Unstringified number index

    useEffect(() => {
      const fetchData = async () => {
        const data = await GetModels();
        if (data) {
          setModelsData(data);
        } else {
          toast.error("Failed to retrieve models. Check the console for more info.");
        }
      };
    
      fetchData();
    }, []);

    function ChangeHypValue(index: number, value: any) {
      if (!modelsData) return;
      modelsData[model_hyp_index_int].hyperparameters[index].value = value;
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
      SendTrainingRequest(Cookies.get("webserver_url") ?? "http://localhost:8000", HypToDict(modelsData[model_hyp_index_int].hyperparameters), model_hyp_index_int);
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
                          {typeof param.value === 'number' && (
                            <Input type="number" 
                              value={param.value ? param.value : 0} 
                              min={param.min_value ? param.min_value : -Infinity}
                              max={param.max_value ? param.max_value : Infinity} 
                              step={param.incriment ? param.incriment : 1}
                              onChange={(e) => ChangeHypValue(index, Number(e.target.value))}
                            />
                          )}
                          {typeof param.value === 'string' && (
                            <Input type="text" value={param.value} onChange={(e) => ChangeHypValue(index, e.target.value)} />
                          )}
                          {typeof param.value === 'boolean' && (
                            <Select defaultValue={param.value === true ? "Yes" : "No"} onValueChange={(value) => ChangeHypValue(index, value === "Yes" ? true : false)}>
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
            
                      {param.special_type === 'path' && (
                        <Input type="text" value={param.value} onChange={(e) => ChangeHypValue(index, e.target.value)} />
                      )}
            
                      {param.special_type === 'dropdown' && (
                        <Select value={param.value} onValueChange={(value) => ChangeHypValue(index, value)}>
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
                      <Separator orientation="horizontal" className="bg-zinc-600 w-full my-3" />
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

  const Sidebar = memo(() => {
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
            <div key={index} className="relative w-[calc(100%-30px)] ml-[15px] h-auto p-4 mt-2 bg-zinc-800 rounded-lg transition-all duration-300 hover:shadow-lg hover:bg-zinc-700"
              onMouseEnter={() => handleMouseEnter(index)} onMouseLeave={handleMouseLeave} onClick={() => set_current_model_index(index)}>
              <div className="items-center gap-3 flex">
                {model.data_type === "image" ? (
                  <Image width={24} height={24} />
                ) : (
                  <Text width={24} height={24} />
                )}
                <p className="text-sm font-semibold">{model.type}</p>
              </div>
              <div
                className={`mt-2 overflow-hidden text-xs text-gray-400 transition-all duration-300 ${hovered_model_index === index ? "max-h-32 scale-up" : "max-h-4"}`}>
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
          </div>
        </div>
      </Card>
    );
  });


  const Home = memo(() => {
    return (
      <Card className="flex flex-col gap-2 items-center justify-center ml-3 h-[calc(100vh-25px)] w-[calc(100vw-324px)] bg-black">
        <h1 className="text-2xl font-bold">{greeting}</h1>
        <p className="text-zinc-500 text-md">Welcome to the training dashboard!</p>
      </Card>
    );
  });

  const LossGraph = memo(({ training_losses, validation_losses }: { training_losses: number[]; validation_losses: number[] }) => {
    const t_loss_color = "#ff0000";
    const v_loss_color = "#00ff00";
  
    const chartConfig = {
      desktop: {
        label: "Training Loss",
        color: t_loss_color,
      },
      mobile: {
        label: "Validation Loss",
        color: v_loss_color,
      },
    } satisfies ChartConfig;
  
    const data = training_losses.map((training_loss, index) => ({
      "Epoch": index + 1,
      "Training Loss": training_loss,
      "Validation Loss": validation_losses[index],
    }));
  
    return (
      <Card className="w-[calc(100vw-370px)] h-[560px]">
        <ChartContainer config={chartConfig} className="h-[542px] w-[calc(100vw-380px)] mt-2">
          <AreaChart accessibilityLayer data={data}>
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
              tickFormatter={(value) => value.toFixed(3)}
              label={{ value: "Loss", angle: -90, position: "insideLeft", offset: 10 }}
            />
            <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
            <Legend layout="horizontal" verticalAlign="bottom" align="left" wrapperStyle={{ marginLeft: '60px', marginBottom: '5px' }}/>
            <defs>
              <linearGradient id="fillTrainingLoss" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={t_loss_color} stopOpacity={0.8} />
                <stop offset="60%" stopColor={t_loss_color} stopOpacity={0.3} />
                <stop offset="100%" stopColor={t_loss_color} stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="fillValidationLoss" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={v_loss_color} stopOpacity={0.8} />
                <stop offset="60%" stopColor={v_loss_color} stopOpacity={0.3} />
                <stop offset="100%" stopColor={v_loss_color} stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <Area 
              dataKey="Training Loss" 
              type="natural" 
              fill="url(#fillTrainingLoss)" 
              fillOpacity={0.4} 
              stroke={t_loss_color} 
              stackId="a" 
            />
            <Area 
              dataKey="Validation Loss" 
              type="natural" 
              fill="url(#fillValidationLoss)" 
              fillOpacity={0.4} 
              stroke={v_loss_color} 
              stackId="a" 
            />
          </AreaChart>
        </ChartContainer>
      </Card>
    );
  });
  
  const ModelVisualizer = memo(({ model }: { model: Model }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const scrollPositionRef = useRef<number>(0);

    // Store the scroll position before the component updates
    useEffect(() => {
      const container = containerRef.current;
      if (container) {
        scrollPositionRef.current = container.scrollTop;
      }
    }, [model]);

    // Restore the scroll position after the component updates
    useEffect(() => {
      const container = containerRef.current;
      if (container) {
        container.scrollTop = scrollPositionRef.current;
      }
    }, [model]);

    return (
      <Card className="flex flex-col ml-3 h-[calc(100vh-25px)] w-[calc(100vw-324px)] bg-black overflow-y-auto" ref={containerRef}>
        <div className="m-4 overflow-y-auto overflow-x-hidden">
          <h1 className="text-3xl font-bold mb-4">{model.type}</h1>
          <p className="text-sm">{model.status} • {typeof model.estimated_time === "number" ? formatDuration(model.estimated_time) : model.estimated_time} remaining</p>
          {model.status === "Initializing" && (
            <div className="flex items-center justify-center w-full h-full">
              <p className="text-sm text-zinc-500">Model is initializing, please wait...</p>
            </div>
          )}
          {model.status === "Training" && (
            <div>
              <div className="flex flex-col gap-3 mt-8">
                <h2 className="text-xl font-bold mb-1">Epochs</h2>
                <Progress value={ConvertToProgressValue(model.epoch, model.epochs)} className="h-2 w-[calc(100vw-380px)]" />
                <p>{model.epoch} out of {model.epochs} epochs ({((model.epoch / model.epochs) * 100).toFixed(1)}%)</p>
              </div>
              {typeof model.estimated_time === "number" && model.estimated_time > 0 && (
                <div className="flex flex-col gap-3 mt-8">
                  <h2 className="text-xl font-bold mb-1">Time Elapsed</h2>
                  <Progress value={ConvertToProgressValue(model.elapsed, model.estimated_time + model.elapsed)} className="h-2 w-[calc(100vw-380px)]" />
                  <p>{formatDuration(model.elapsed)} out of {formatDuration(model.estimated_time + model.elapsed)} ({((model.elapsed / model.estimated_time) * 100).toFixed(1)}%)</p>
                </div>
              )}
              {model.training_losses.length > 0 && model.val_losses.length > 0 && (
                <div className="mt-8">
                  <LossGraph training_losses={model.training_losses} validation_losses={model.val_losses} />
                </div>
              )}
              <Separator orientation="horizontal" className="bg-zinc-600 w-[calc(100%-30px)] ml-[15px] my-8" />
              <h1 className="text-2xl font-bold mb-4 ml-1">Additional information:</h1>
              <div className="w-[calc(100%-20px)] ml-[5px]">
                {model.additional_training_data.length > 0 && (
                  <Accordion type="single" collapsible>
                    <AccordionItem value="training_stats">
                      <AccordionTrigger className="text-lg font-bold">Additional Training Data</AccordionTrigger>
                      <AccordionContent>
                        {/* @ts-ignore */}
                        {Object.entries(model.model_data).map(([_, data]: [string, { name: string; value: boolean | string | number }], index) => (
                          <div key={index} className="flex flex-row w-full justify-between mb-6">
                            <p className="text-md font-bold">{data.name}</p>
                            <p className="text-md mr-4">
                              {typeof data.value === "boolean" ? data.value ? "Yes" : "No" : data.value}
                            </p>
                          </div>
                        ))}
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                )}
                {model.model_data.length > 0 && (
                  <Accordion type="single" collapsible>
                    <AccordionItem value="training_stats">
                      <AccordionTrigger className="text-lg font-bold">Model Data</AccordionTrigger>
                      <AccordionContent>
                        {/* @ts-ignore */}
                        {Object.entries(model.model_data).map(([_, data]: [string, { name: string; value: boolean | string | number }], index) => (
                          <div key={index} className="flex flex-row w-full justify-between mb-6">
                            <p className="text-md font-bold">{data.name}</p>
                            <p className="text-md mr-4">
                              {typeof data.value === "boolean" ? data.value ? "Yes" : "No" : data.value}
                            </p>
                          </div>
                        ))}
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                )}
                {model.hyperparameters.length > 0 && (
                  <Accordion type="single" collapsible>
                    <AccordionItem value="training_stats">
                      <AccordionTrigger className="text-lg font-bold">Hyperparameters</AccordionTrigger>
                      <AccordionContent>
                        {/* @ts-ignore */}
                        {Object.entries(model.hyperparameters).map(([_, data]: [string, { name: string; value: boolean | string | number }], index) => (
                          <div key={index} className="flex flex-row w-full justify-between mb-6">
                            <p className="text-md font-bold">{data.name}</p>
                            <p className="text-md mr-4">
                              {typeof data.value === "boolean" ? data.value ? "Yes" : "No" : data.value}
                            </p>
                          </div>
                        ))}
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                )}
              </div>
            </div>
          )}
        </div>
      </Card>
    )
  });

  return (
    <div className="flex flex-row">
      {showLoading ? (
        <Loading loading_text="Retrieving models..." />
      ) : (
        <>
          <Sidebar />
          {current_model_index === -1 ? <Home /> : <ModelVisualizer model={models[current_model_index]} />}
          {creating_model && <CreateNewModel />}
        </>
      )}
    </div>
  );
}