import { Card } from "@/components/ui/card"
import { BarLoader } from "react-spinners"

function Loading({ loading_text, fullscreen }: { loading_text: string, fullscreen: boolean }) {
    if (!fullscreen) {
        return (
            <div className="flex flex-col text-center items-center space-y-5">
                <h2 className="text-xl font-bold">Pyro Axis AI Training Dashboard</h2>
                <BarLoader color="#ffffff" height={5} loading={true} speedMultiplier={1.2} width={250} />
                <p>{loading_text}</p>
            </div>
        );
    }

    return (
        <Card className="flex flex-col w-full text-center items-center justify-center h-[calc(100vh-25px)] space-y-5 pb-0 overflow-auto rounded-t-md bg-black">
            <h2 className="text-xl font-bold">Pyro Axis AI Training Dashboard</h2>
            <BarLoader color="#ffffff" height={5} loading={true} speedMultiplier={1.2} width={250} />
            <p>{loading_text}</p>
        </Card>
    );
}


export { Loading }