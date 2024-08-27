import Image from "next/image"
import { Card } from "@/components/ui/card"
import { BarLoader } from "react-spinners"

function Loading({ loading_text } : { loading_text: string }) {
    return (
        <Card className="flex flex-col w-full text-center items-center justify-center h-[calc(100vh-120px)] space-y-5 pb-0 overflow-auto rounded-t-md">
            <h2 className="text-xl font-bold">PyTorch AI Training Dashboard</h2>
            <BarLoader color="#ffffff" cssOverride={{}} height={5} loading speedMultiplier={1.2} width={250}/>
            <p>{loading_text}</p>
        </Card>
    )
}

export { Loading }