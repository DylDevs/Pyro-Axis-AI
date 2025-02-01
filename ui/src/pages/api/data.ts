import { NextApiRequest, NextApiResponse } from "next";
import fs from "fs";
import path from "path";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const file_path = path.join(process.cwd(), "..", "cache", "cache.json");

  try {
    const fileContent = fs.readFileSync(file_path, "utf-8");
    const jsonData = JSON.parse(fileContent);
    res.status(200).json(jsonData);
  } catch (error) {
    res.status(500).json({ error: "File not found" });
  }
}
