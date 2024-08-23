"use client";

import { Menubar, MenubarContent, MenubarItem,
  MenubarMenu, MenubarTrigger} from "@/components/ui/menubar";
import { FaSun } from "react-icons/fa6";
import { IoMoonSharp, IoFolderOpenOutline, IoHome } from "react-icons/io5";
import { LuGitBranchPlus } from "react-icons/lu";
import { Button } from "@/components/ui/button";
import { useTheme } from "next-themes"
import { useRouter } from "next/navigation"
import { toast } from "sonner"
import { Badge } from "./ui/badge";

export function PyTorchMenubar() {  
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
      }
    );
    
  }

  return (
    <div className="flex flex-row">
      <div className="mr-3 hidden sm:block">
        <Menubar className="h-12 min-w-58 w-sm:flex">
          <Button className="font-bold" variant={"ghost"} onClick={() => push("/")}>
            PyTorch Training Controller V1.0.0
          </Button>
        </Menubar>
      </div>
      <div className="mr-3 block sm:hidden">
        <Menubar className="h-12 min-w-58 w-sm:flex">
          <Button className="font-bold" variant={"ghost"} onClick={() => push("/")}>
            <IoHome className="w-4 h-4" />
          </Button>
        </Menubar>
      </div>
      <Menubar className="flex flex-row h-12 w-full justify-between">
        <div className="flex flex-row justify-start">
          <MenubarMenu>
              <MenubarTrigger>
                  <div className="flex flex-row gap-1 items-center">
                      <LuGitBranchPlus className="w-4 h-4" />Models
                  </div>
              </MenubarTrigger>
              <MenubarContent>
                  <MenubarItem>
                      <div onClick={() => push("/model_dashboard")}>Model Overview</div>
                  </MenubarItem>
              </MenubarContent>
          </MenubarMenu>
          <MenubarMenu>
              <MenubarTrigger>
                  <div className="flex flex-row gap-1 items-center">
                      <IoFolderOpenOutline className="w-4 h-4" />Saved Models
                  </div>
              </MenubarTrigger>
              <MenubarContent>
                  <MenubarItem>
                      <div onClick={() => push("/saved_model_dashboard")}>Saved Models Overview</div>
                  </MenubarItem>
              </MenubarContent>
          </MenubarMenu>
          <MenubarMenu>
            <MenubarTrigger>
              <div className="flex flex-row gap-1 items-center">
                  {theme === "light" ? <FaSun className="w-4 h-4" /> : <IoMoonSharp className="w-4 h-4" />}Theme    
              </div>
            </MenubarTrigger>
            <MenubarContent>
              <MenubarItem onClick={() => SetThemeColor("light")}>
                <div className="flex flex-row gap-2 items-center">
                  <FaSun className="w-4 h-4"/>Light    
                </div>
              </MenubarItem>
              <MenubarItem onClick={() => SetThemeColor("dark")}>
                <div className="flex flex-row gap-2 items-center">
                  <IoMoonSharp className="w-4 h-4"/>Dark    
                </div>
              </MenubarItem>
            </MenubarContent>
          </MenubarMenu>
          </div>
      </Menubar>
    </div>
  );
}