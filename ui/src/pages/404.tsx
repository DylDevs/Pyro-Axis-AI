import React, { useState, useEffect } from 'react';
import { Card } from "@/components/ui/card";
import { Loading } from "@/components/loading";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/router";

export default function NotFound() {
    const { push } = useRouter();
    return (
        <Card className="flex flex-col w-full h-[calc(100vh-120px)] space-y-5 pb-0 overflow-auto rounded-t-md">
            <div className="flex flex-col h-screen items-center justify-center space-y-5">
                <h2 className="text-xl font-bold">Torch AI Training Dashboard</h2>
                <h1 className="text-5xl font-bold">404</h1>
                <h3 className="text-xl font-bold">The page you are looking for does not exist</h3>
                <Button variant={"secondary"} onClick={() => push("/") }>Home</Button>
            </div>
        </Card>
    );
}