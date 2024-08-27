import {
    Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList, 
    CommandSeparator, CommandShortcut, CommandDialog
} from "@/components/ui/command"
import { useState, useEffect } from "react"
import { useTheme } from "next-themes"
import { useRouter } from "next/navigation"
import { toast } from "sonner"
import { Button } from "./ui/button"

export default function CommandMenu() {
    const [open, setOpen] = useState(false)
    const { push } = useRouter()
    const { theme, setTheme } = useTheme()

    const SetThemeColor = (theme: string) => {
      toast.promise(
        new Promise((resolve) => {
          setTimeout(resolve, 1000);
          setTheme(theme)
        }),
        {
          loading: "Switching theme to " + theme + "...",
          success: "Theme set to " + theme + "!",
          error: "Failed to switch theme!",
        }
      );
      
    }

    useEffect(() => {
        const down = (e: KeyboardEvent) => {
            if (((e.key === "k" || e.key === "Enter") && (e.metaKey || e.ctrlKey)) || e.key === "F1") {
                e.preventDefault()
                setOpen((open) => !open)
            }
        }
        document.addEventListener("keydown", down)
        return () => document.removeEventListener("keydown", down)
    }, [])

    return (
        <CommandDialog open={open} onOpenChange={setOpen}>
            <CommandInput placeholder="Type a command or search..." />
            <CommandList>
                <CommandEmpty>No results found.</CommandEmpty>
                <CommandGroup heading="General">
                    <CommandItem onSelect={() => {push("/train_model"); setOpen(false)}}>
                        Train a Model
                    </CommandItem>
                    <CommandItem onSelect={() => {push("/model_dashboard"); setOpen(false)}}>
                        Model Training Dashboard
                    </CommandItem>
                    <CommandItem disabled>
                        Manage Trained Models
                    </CommandItem>
                </CommandGroup>
                <CommandSeparator />
                <CommandGroup heading="Theme">
                    <CommandItem onClick={() => {SetThemeColor("light"); setOpen(false)}}>
                        <p>Light</p>
                    </CommandItem>
                    <CommandItem onClick={() => {SetThemeColor("dark"); setOpen(false)}}>
                        <p>Dark</p>
                    </CommandItem>
                    <CommandItem onClick={() => {SetThemeColor("system"); setOpen(false)}}>
                        <p>System</p>
                    </CommandItem>
                </CommandGroup>
            </CommandList>
        </CommandDialog>
    )
}
