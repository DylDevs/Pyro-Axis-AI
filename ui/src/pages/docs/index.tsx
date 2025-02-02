import { Flame, Home, TriangleAlert } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useRouter } from "next/router";
import { useState, useEffect } from "react";
import { StartDocs, StopDocs } from "@/components/webserver";
import { Loading } from "@/components/loading";
 
const Sidebar = () => {
    const { push } = useRouter();
    return (
        <Card className="bg-black w-72 h-[calc(100vh-25px)] flex flex-col">
            <div className="bg-black pt-4 pb-2 flex flex-col">
                <div className="flex rounded-lg hover:bg-zinc-800 mb-3 py-1 w-[calc(100%-30px)] ml-[15px]">
                    <Flame className="h-14 w-14" color="#eb4b2b"/>
                    <div className="flex flex-col pl-4">
                        <h1 className="text-white text-2xl font-bold">Pyro Axis AI</h1>
                        <h2 className="text-white text-sm font-small">Training Dashboard</h2>
                    </div>
                </div>
                <Separator orientation="horizontal" className="bg-zinc-600 w-[calc(100%-30px)] ml-[15px] mb-2" />
            </div>
            <div className="bg-black flex flex-col mb-4">
                <div className="flex flex-col gap-2 w-[calc(100%-30px)] ml-[15px]">
                    <Button variant="outline" className="w-full bg-zinc-900 hover:bg-zinc-800" onClick={() => { StopDocs(); push("/") }}><Home />Home</Button>
                </div>
            </div>
        </Card>
    );
}

export default function Docs() {
    const [loaded_state, set_loaded_state] = useState<null | boolean>(null);

    useEffect(() => {
        const runDocs = async () => {
          const status = await StartDocs(); // Wait for the function to complete
          set_loaded_state(status); // Set the variable to true when function completes
        };
    
        runDocs();
    }, []);

    if (loaded_state === null) {
        // Show loading screen when loaded_state is null
        return (
            <div className="flex flex-row">
                <Sidebar />
                <Card className="flex flex-col gap-2 items-center justify-center ml-3 h-[calc(100vh-25px)] w-[calc(100vw-324px)] bg-black">
                    <Loading loading_text="Building documentation..." fullscreen={false} />
                </Card>
            </div>
        );
    }

    if (loaded_state === false) {
        // Show error screen when loaded_state is false
        return (
            <div className="flex flex-row">
                <Sidebar />
                <Card className="flex flex-col gap-2 items-center justify-center ml-3 h-[calc(100vh-25px)] w-[calc(100vw-324px)] bg-black">
                    <TriangleAlert className="h-14 w-14"/>
                    <p className="text-muted-foreground text-lg">Error loading documentation, check the console for more info</p>
                </Card>
            </div>
        );
    }

    // Show the embed when loaded_state is true
    return (
        <div className="flex flex-row">
            <Sidebar />
            <Card className="flex flex-col gap-2 items-center justify-center ml-3 h-[calc(100vh-25px)] w-[calc(100vw-324px)] bg-black">
                 <embed src="http://localhost:5000" className="w-full h-full bg-black" />
            </Card>
        </div>
    );
}

