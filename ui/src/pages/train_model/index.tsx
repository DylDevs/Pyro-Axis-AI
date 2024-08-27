import React, { useState, useEffect } from "react";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Card } from "../../components/ui/card";
import { RadioGroup, RadioGroupItem } from "../../components/ui/radio-group";
import { Label } from "../../components/ui/label";
import { useRouter } from "next/router";
import { toast } from "sonner";
import { Loading } from "../../components/loading";

// @ts-ignore | Prevents module not found error from js-cookie, even though it is installed
import Cookies from 'js-cookie';

interface Hyperparameters {
    [key: string]: number | string | boolean;
}

export default function TrainModel() {
    const router = useRouter();
    const { push } = router;

    const [modelType, setModelType] = useState<string>("language_classification");
    const [sendingRequest, setSendingRequest] = useState<boolean>(false);
    const [hyperparameters, setHyperparameters] = useState<Hyperparameters>({});

    const webserver_url = Cookies.get("webserver_url") || "http://localhost:8000";
    const connected = Cookies.get("connected") === "true";

    useEffect(() => {
        const storedHyps = Cookies.get(`${modelType}Hyps`);
        if (storedHyps) {
            setHyperparameters(JSON.parse(storedHyps));
        } else {
            const defaultHyps = getDefaultHyperparameters(modelType);
            setHyperparameters(defaultHyps);
            Cookies.set(`${modelType}Hyps`, JSON.stringify(defaultHyps));
        }
    }, [modelType]);

    const getDefaultHyperparameters = (type: string): Hyperparameters => {
        switch (type) {
            case "language_classification":
                return {
                    num_epochs: 100,
                    batch_size: 32,
                    classes: 3,
                    learning_rate: 0.001,
                    max_learning_rate: 0.001,
                    train_val_ratio: 0.9,
                    num_workers: 0,
                    dropout: 0.3,
                    patience: 5,
                    shuffle_train: true,
                    shuffle_val: true,
                    shuffle_each_epoch: true,
                    pin_memory: false,
                    drop_last: true,
                    cache: true,
                    embedding_dim: 300,
                    hidden_dim: 512
                };
            case "image_classification":
                return {
                    num_epochs: 100,
                    batch_size: 64,
                    classes: 3,
                    img_width: 480,
                    img_height: 480,
                    random_crop: 0.85,
                    random_flip: true,
                    random_rotation: 10,
                    learning_rate: 0.001,
                    max_learning_rate: 0.001,
                    train_val_ratio: 0.9,
                    num_workers: 0,
                    dropout: 0.2,
                    patience: 5,
                    shuffle_train: true,
                    shuffle_val: true,
                    shuffle_each_epoch: true,
                    pin_memory: false,
                    drop_last: true,
                    cache: true,
                };
            case "object_detection":
                return {};
            case "multilabel_image_classification":
                return {};
            default:
                return {};
        }
    };

    const handleModelTypeChange = (type: string) => {
        setModelType(type);
    };

    const changeHyperParam = (param: string, value: string) => {
        let newValue: number | string | boolean = value;
        if (value === "" || value === null) {
            newValue = "";
        } else if (!isNaN(Number(value))) {
            newValue = Number(value);
        } else if (value === "true" || value === "false") {
            newValue = value === "true";
        }

        setHyperparameters(prev => {
            const updated = { ...prev, [param]: newValue };
            Cookies.set(`${modelType}Hyps`, JSON.stringify(updated));
            return updated;
        });
    };

    const sendTrainingRequest = async () => {
        try {
            const response = await fetch(`${webserver_url}/train/${modelType}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ hyperparameters })
            });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const data = await response.json();
            if (data.status !== "ok") {
                throw new Error("Failed to start training! Full traceback: " + data.traceback);
            }
            return data;
        } catch (error) {
            console.error("Error:", error);
            throw error;
        }
    };

    const submit = () => {
        setSendingRequest(true);

        if (Object.values(hyperparameters).some(value => value === null || value === undefined || value === "")) {
            toast.error("Please fill all hyperparameters, train request not sent.");
            setSendingRequest(false);
            return;
        }

        toast.promise(
            new Promise(async (resolve, reject) => {
                try {
                    await sendTrainingRequest();
                    setTimeout(() => {
                        push("/model_dashboard");
                    }, 3000);
                    resolve(0);
                } catch (error) {
                    console.log(error);
                    reject();
                    setSendingRequest(false);
                }
            }),
            {
                loading: "Sending request...",
                success: "Training request sent!",
                error: "Failed to connect to training server!",
            }
        );
    };

    const hyperparam_names: { [key: string]: string } = {
        num_epochs: "Epochs",
        batch_size: "Batch Size",
        classes: "Classes",
        img_width: "Image Width",
        img_height: "Image Height",
        random_crop: "Random Crop",
        random_flip: "Random Flip",
        random_rotation: "Random Rotation",
        learning_rate: "Learning Rate",
        max_learning_rate: "Max Learning Rate",
        train_val_ratio: "Train Val Ratio",
        num_workers: "Num Workers",
        dropout: "Dropout",
        patience: "Patience",
        shuffle_train: "Shuffle Train",
        shuffle_val: "Shuffle Val",
        shuffle_each_epoch: "Shuffle Each Epoch",
        pin_memory: "Pin Memory",
        drop_last: "Drop Last",
        cache: "Cache",
        embedding_dim: "Embedding Dim",
        hidden_dim: "Hidden Dim"
    };

    if (sendingRequest) {
        return <Loading loading_text="Sending training request..." />;
    }
    
    return (
        <Card className="flex flex-col w-full h-[calc(100vh-120px)] space-y-5 pb-0 overflow-auto rounded-t-md">
            <div className="flex flex-col items-center justify-center h-screen space-y-5">
                <div className="flex flex-col items-start space-y-3 p-5">
                    <h1 className="text-3xl font-bold">Train Model</h1>
                    <p className="text-zinc-500">Select the type of model to train and select your hyperparameters. Click train to start the training process.</p>
                    <div className="flex flex-col space-y-2 pt-2">
                        <h2>Model Type:</h2>
                        <RadioGroup value={modelType} onValueChange={handleModelTypeChange}>
                            <div className="flex items-center space-x-2">
                                <RadioGroupItem value="language_classification" id="modelType1" />
                                <Label htmlFor="modelType1">Language Classification</Label>
                            </div>
                            <div className="flex items-center space-x-2">
                                <RadioGroupItem value="image_classification" id="modelType2" />
                                <Label htmlFor="modelType2">Image Classification</Label>
                            </div>
                            <div className="flex items-center space-x-2">
                                <RadioGroupItem value="object_detection" id="ModelType3" />
                                <Label htmlFor="ModelType3">Object Detection</Label>
                            </div>
                            <div className="flex items-center space-x-2">
                                <RadioGroupItem value="multilabel_image_classification" id="ModelType4" />
                                <Label htmlFor="ModelType4">Multi-label Image Classification</Label>
                            </div>
                        </RadioGroup>
                    </div>
                    <div className="grid grid-cols-4 gap-4 py-2 w-full">
                        {Object.keys(hyperparameters).length === 0 && (
                            <div className="col-span-3 flex flex-col justify-center">
                                <h2 className="text-2xl font-bold">Under Construction</h2>
                                <p className="text-zinc-500">This model type is currently under construction. A model of this type cannot be trained.</p>
                            </div>
                        )}
                        {Object.keys(hyperparameters).map((param) => (
                            <div key={param} className="flex flex-col space-y-1">
                                <h2 className="pl-1">{hyperparam_names[param] || param}</h2>
                                <Input
                                    // @ts-ignore Silences type error
                                    value={hyperparameters[param]}
                                    onChange={(e) => changeHyperParam(param, e.target.value)}
                                    placeholder={param}
                                />
                            </div>
                        ))}
                    </div>
                    {connected ? <p className="text-green-500">Connected to Training Server</p> : <p className="text-red-500">Not Connected to Training Server, please check your internet connection</p>}
                    {connected ? <Button onClick={submit} className="w-full">Train</Button> : 
                        <div className="flex flex-row w-full gap-2">
                            <Button disabled className="w-1/2">Train</Button> 
                            <Button onClick={() => router.reload()} className="w-1/2">Attempt to Reconnect</Button>
                        </div>
                    }
                </div>
            </div>
        </Card>
    );
}
