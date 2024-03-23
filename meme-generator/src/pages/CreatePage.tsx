import { useState, useEffect } from "react";
import MemeCard from "../components/MemeCard";

export default function CreatePage() {
  const [memeTemplates, setMemeTemplates] = useState([]);

  useEffect(() => {
    fetch("https://api.imgflip.com/get_memes")
      .then((res) => res.json())
      .then((data) => setMemeTemplates(data.data.memes));
  }, [memeTemplates]);

  return (
    <main>
      <div className="mt-3 mb-3 overflow-y-scroll h-60">
        <div className="grid grid-cols-10 gap-1">
          {memeTemplates.map((item) => (
            <MemeCard img={item} />
          ))}
        </div>
      </div>
    </main>
  );
}
