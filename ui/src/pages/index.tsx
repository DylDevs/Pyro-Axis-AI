import { useEffect, useRef, useState, useCallback} from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { 
  Plus,
  ArrowDownToLine,
  X,
  Power,
  Image,
  Text
} from "lucide-react";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"

import { Area, AreaChart, CartesianGrid, XAxis, YAxis, Legend } from "recharts";

const training_model_example = 
{
  "type": "Language Classification", // Model type, determined by the model config in the file
  "data_type": "text",  // Data format, determined by the extension of the model dataset file(s) (image, text, audio, unknown)
  "status": "Training", // Status of the model (Training, Paused, Finished, Early Stopped, Manually Stopped, Error)
  "epoch": 50, // Current epoch
  "epochs": 100, // Total number of epochs
  "training_losses": [
    1.862, 1.71, 1.6, 1.55, 1.523, 1.45, 1.4, 1.38, 1.35, 1.379,
    1.25, 1.222, 1.198, 1.16, 1.13, 1.115, 1.08, 1.05, 1.02, 1.0,
    0.98, 0.966, 0.94, 0.92, 0.919, 0.88, 0.86, 0.85, 0.84, 0.83,
    0.82, 0.81, 0.872, 0.79, 0.78, 0.77, 0.767, 0.75, 0.745, 0.73,
    0.72, 0.71, 0.792, 0.69, 0.68, 0.67, 0.667, 0.65, 0.64, 0.639
  ],
  "val_losses": [
    1.9, 1.75, 1.7, 1.6, 1.58, 1.5, 1.48, 1.46, 1.45, 1.42,
    1.4, 1.38, 1.35, 1.34, 1.3, 1.28, 1.27, 1.25, 1.23, 1.22,
    1.2, 1.18, 1.16, 1.15, 1.13, 1.12, 1.1, 1.08, 1.07, 1.06,
    1.05, 1.04, 1.03, 1.02, 1.01, 1.0, 0.99, 0.98, 0.97, 0.96,
    0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86
  ],
  "best_epoch": 50, // Best epoch
  "best_training_loss": 1.8, // Best training loss so far
  "best_val_loss": 1.9, // Best validation loss so far
  "elapsed" : 2250, // Elapsed time in seconds since the model training was started
  "estimated_time" : 6000, // Estimated time left in seconds
  "total_time" : null, // Total time in seconds that the model trained for
  "time_per_epoch" : 10.2, // Time per epoch in seconds
  "model_data" : {
    "Train Size (Instances)": 900,
    "Val Size (Instances)": 100,
    "Max Sequence Length": 72,
    "Vocab Size": 254,
    "Total Parameters": 100000,
    "Trainable Parameters": 9876,
    "Non-Trainable Parameters": 124,
    "Model Size (MB)": 132.75,
    "Scaler": "GradScaler",
    "Criterion": "CrossEntropyLoss",
    "Optimizer": "Adam",
    "Loss Scheduler": "OneCycleLR",
  },
  "hyperparameters": {
    "Data Path": "\\data\\language_classification",
    "Model Path": "\\models\\language_classification",
    "Device": "GPU",
    "Epochs": 100,
    "Batch Size": 64,
    "Classes": 2,
    "Learning Rate": 0.001,
    "Max Learning Rate": 0.01,
    "Train Val Split": 0.9,
    "Workers": 0,
    "Dropout": 0.2,
    "Patience": 25,
    "Shuffle Train": true,
    "Shuffle Val": true,
    "Shuffle Each Epoch": true,
    "Pin Memory": false,
    "Drop Last": false,
    "Embedding Dim": 128,
    "Hidden Dim": 512
  }
}

const models = [
  training_model_example
]

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
  if (seconds < 0) throw new Error("Seconds cannot be negative");

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
  if (seconds > 0 || parts.length === 0) parts.push(`${seconds} seconds`);

  return parts.join(", ");
}

function ConvertToProgressValue(value: number, max: number): number {
  return (value / max) * 100
}

export default function Index() {
  const [greeting, setGreeting] = useState(getGreeting());
  const [hovered_model_index, setHovered_model_index] = useState(-1);
  const [current_model_index, set_current_model_index] = useState(-1);
  const scrollRef = useRef(null); // Reference for scrollable container
  const scrollPosition = useRef(0); // Store scroll position

  useEffect(() => {
    const interval = setInterval(() => {
      if (current_model_index === -1) setGreeting(getGreeting());
    }, 30000);
    return () => clearInterval(interval);
  }, [current_model_index]);

  useEffect(() => {
    if (scrollRef.current) {
      // @ts-ignore
      scrollRef.current.scrollTop = scrollPosition.current; // Restore scroll position
    }
  }, [hovered_model_index]); // Runs when hover state changes

  const handleMouseEnter = useCallback((index : number) => {
    // @ts-ignore
    scrollPosition.current = scrollRef.current.scrollTop; // Save current scroll position
    setHovered_model_index(index);
  }, []);

  const handleMouseLeave = useCallback(() => {
    // @ts-ignore
    scrollPosition.current = scrollRef.current.scrollTop; // Save current scroll position
    setHovered_model_index(-1);
  }, []);

  function Sidebar() {
    return (
      <Card className="bg-black w-72 h-[calc(100vh-25px)] flex flex-col">
        <div className="bg-black pt-4 pb-2 flex flex-col">
          <div className="flex rounded-lg hover:bg-zinc-800 mb-3 w-[calc(100%-30px)] ml-[15px]" onClick={() => set_current_model_index(-1)}>
            <img src="/favicon.ico" alt="logo" className="h-14 w-14" />
            <div className="flex flex-col pl-4">
              <h1 className="text-white text-2xl font-bold">Torch AI</h1>
              <h2 className="text-white text-sm font-small">Training Dashboard</h2>
            </div>
          </div>
          <Separator orientation="horizontal" className="bg-zinc-600 w-[calc(100%-30px)] ml-[15px]" />
        </div>
        <div className="bg-black flex flex-col flex-grow overflow-y-auto" ref={scrollRef}>
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
            <Button variant="outline" className="w-full bg-zinc-900 hover:bg-zinc-800"><Plus />Train Model</Button>
            <Button variant="outline" className="w-full bg-zinc-900 hover:bg-zinc-800"><X />Stop Training</Button>
            <Button variant="outline" className="w-full bg-zinc-900 hover:bg-zinc-800"><ArrowDownToLine />Stop Training and Save</Button>
            <Button variant="outline" className="w-full bg-zinc-900 hover:bg-red-600"><Power />Shut Down</Button>
          </div>
        </div>
      </Card>
    );
  }


  function Home() {
    return (
      <Card className="flex flex-col gap-2 items-center justify-center ml-3 h-[calc(100vh-25px)] w-[calc(100vw-324px)] bg-black">
        <h1 className="text-2xl font-bold">{greeting}</h1>
        <p className="text-zinc-600 text-md">Welcome to the training dashboard!</p>
      </Card>
    );
  }

  function LossGraph() {
    const training_losses = models[current_model_index].training_losses;
    const validation_losses = models[current_model_index].val_losses;
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
              label={{ value: "Epoch", position: "bottom", offset: -5 }} // Added label
            />
            <YAxis 
              tickLine={true} 
              axisLine={false} 
              tickCount={5}
              tickFormatter={(value) => value.toFixed(3)} // Adjust formatting as needed
              label={{ value: "Loss", angle: -90, position: "insideLeft", offset: 10 }} // Added label
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
  }
  

  function ModelVisualizer() {
    return (
      <Card className="flex flex-col ml-3 h-[calc(100vh-25px)] w-[calc(100vw-324px)] bg-black overflow-y-auto">
        <div className="m-4 overflow-y-auto overflow-x-hidden">
          <h1 className="text-3xl font-bold mb-4">{models[current_model_index].type}</h1>
          <p className="text-sm">{models[current_model_index].status} • {formatDuration(models[current_model_index].estimated_time)} remaining</p>
          <div className="flex flex-col gap-3 mt-8">
            <h2 className="text-xl font-bold mb-1">Epochs</h2>
            <Progress value={ConvertToProgressValue(models[current_model_index].epoch, models[current_model_index].epochs)} className="h-2 w-[calc(100vw-380px)]" />
            <p>{models[current_model_index].epoch} out of {models[current_model_index].epochs} epochs ({((models[current_model_index].epoch / models[current_model_index].epochs) * 100).toFixed(1)}%)</p>
          </div>
          <div className="flex flex-col gap-3 mt-8">
            <h2 className="text-xl font-bold mb-1">Time Remaining</h2>
            <Progress value={ConvertToProgressValue(models[current_model_index].elapsed, models[current_model_index].estimated_time)} className="h-2 w-[calc(100vw-380px)]" />
            <p>{formatDuration(models[current_model_index].elapsed)} out of {formatDuration(models[current_model_index].estimated_time)} ({((models[current_model_index].elapsed / models[current_model_index].estimated_time) * 100).toFixed(1)}%)</p>
          </div>
          <div className="mt-8">
            <LossGraph />
          </div>
          <Separator orientation="horizontal" className="bg-zinc-600 w-[calc(100%-30px)] ml-[15px] my-8" />
          <h1 className="text-2xl font-bold mb-4 ml-1">Additional information:</h1>
          <div className="w-[calc(100%-20px)] ml-[5px]">
            <Accordion type="single" collapsible>
              <AccordionItem value="training_stats">
                <AccordionTrigger className="text-lg font-bold">Training Stats</AccordionTrigger>
                <AccordionContent>
                  <div className="flex flex-row w-full justify-between mb-6">
                    <p className="text-md font-bold">Under Construction</p>
                    <p className="text-md mr-4">Coming Soon</p>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
            <Accordion type="single" collapsible>
              <AccordionItem value="training_stats">
                <AccordionTrigger className="text-lg font-bold">Model Data</AccordionTrigger>
                <AccordionContent>
                  {Object.entries(models[current_model_index].model_data).map(([name, value], index) => (
                      <div key={index} className="flex flex-row w-full justify-between mb-6">
                        <p className="text-md font-bold">{name}</p>
                        <p className="text-md mr-4">
                          {typeof value === "boolean" ? (value ? "Yes" : "No") : value}
                        </p>
                      </div>
                    ))}
                </AccordionContent>
              </AccordionItem>
            </Accordion>
            <Accordion type="single" collapsible>
              <AccordionItem value="training_stats">
                <AccordionTrigger className="text-lg font-bold">Hyperparameters</AccordionTrigger>
                <AccordionContent>
                  {Object.entries(models[current_model_index].hyperparameters).map(([name, value], index) => (
                    <div key={index} className="flex flex-row w-full justify-between mb-6">
                      <p className="text-md font-bold">{name}</p>
                      <p className="text-md mr-4">
                        {typeof value === "boolean" ? (value ? "Yes" : "No") : value}
                      </p>
                    </div>
                  ))}
                </AccordionContent>
              </AccordionItem>
            </Accordion>
           </div>
        </div>
      </Card>
    )
  }

  return (
    <div className="flex flex-row">
      <Sidebar />
      {current_model_index === -1 ? <Home /> : <ModelVisualizer/>}
    </div>
  );
}