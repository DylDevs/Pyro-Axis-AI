import React from "react";
import { Card } from "../../components/ui/card";

export default function SavedModelDashboard() {
    return (
        <Card className="flex flex-col w-full h-[calc(100vh-120px)] space-y-5 pb-0 overflow-auto rounded-t-md">
            <div className="flex flex-col items-center justify-center space-y-5">
                <h2 className="text-xl font-bold">Saved Model Dashboard</h2>
            </div>
        </Card>
    );
}