import { useState, useEffect } from "react";
import MemeCard from "../components/MemeCard";

export default function CreatePage() {
  const [memeTemplates, setMemeTemplates] = useState([]);
  const [meme, setMeme] = useState({
    imageUrl: "",
    topText: "",
    bottomText: "",
  });

  useEffect(() => {
    fetch("https://api.imgflip.com/get_memes")
      .then((res) => res.json())
      .then((data) => setMemeTemplates(data.data.memes));
  }, [memeTemplates]);

  function handleMeme(event) {
    const { name, value } = event.target;
    setMeme((prevMeme) => ({
      ...prevMeme,
      [name]: value,
    }));
  }

  function selectMeme(index) {
    setMeme((prevMeme) => ({
      ...prevMeme,
      imageUrl: memeTemplates[index].url,
    }));
  }

  return (
    <main>
      <div className="mt-3 mb-3 overflow-y-scroll h-60 flex justify-center">
        <div className="grid grid-cols-10 gap-1">
          {memeTemplates.map((item, index) => (
            <MemeCard
              key={item.id}
              img={item}
              selectMeme={() => selectMeme(index)}
            />
          ))}
        </div>
      </div>
      <div className="">
        <div className="grid grid-cols-2 gap-2 my-3">
          <input
            type="text"
            placeholder="Top Text"
            name="topText"
            value={meme.topText}
            onChange={handleMeme}
            className="meme--textbox"
          />
          <input
            type="text"
            placeholder="Bottom Text"
            name="bottomText"
            value={meme.bottomText}
            onChange={handleMeme}
            className="meme--textbox"
          />
        </div>
        <div className="mx-auto p-1 border-black border-2 flex items-center justify-center relative h-[450px] w-[450px]">
          <img src={meme.imageUrl} className="inset-0 h-full w-full" />
          <h2 className="meme--text absolute top-0">{meme.topText}</h2>
          <h2 className="meme--text absolute bottom-0">{meme.bottomText}</h2>
        </div>
        <div className="">
          <button className="meme--button">Download Meme</button>
          <button className="meme--button">Save Meme</button>
        </div>
      </div>
    </main>
  );
}
