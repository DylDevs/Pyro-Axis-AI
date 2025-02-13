import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";

export default function ErrorPopup({ error, traceback }: { error: string, traceback: string }) {
    const [isOpen, setIsOpen] = useState(true);
  
    return (
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="flex flex-col w-full max-w-5xl">
          <DialogHeader>
            <DialogTitle className="text-red-600">Error</DialogTitle>
            <DialogDescription className="text-zinc-400">
              {error}
            </DialogDescription>
          </DialogHeader>
            <div className="p-3 rounded-md w-full overflow-auto bg-zinc-800">
              <code>
                {traceback.split("\n").map((line, index) => <p key={index}>{line}</p>)}
              </code>
            </div>
        </DialogContent>
      </Dialog>
    );
}
  