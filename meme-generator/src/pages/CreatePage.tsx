import html2canvas from "html2canvas";
import { saveAs } from "file-saver";
import { useState, useEffect } from "react";
import MemeCard from "../components/MemeCard";
import { nanoid } from "nanoid";

export default function CreatePage() {
  const [memeTemplates, setMemeTemplates] = useState([]);

  const [meme, setMeme] = useState({
    imageUrl: "",
    topText: "",
    bottomText: "",
    id: nanoid(),
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

  const downloadMeme = () => {
    const divToCapture = document.getElementById("meme-screenshot");
    if (!divToCapture) return;

    const fileName = "meme_" + meme.id + ".png";

    html2canvas(divToCapture).then((canvas) => {
      canvas.toBlob((blob) => {
        saveAs(blob, fileName);
      });
    });
  };

  useEffect(() => {
    const storedMeme = JSON.parse(localStorage.getItem("memeDraft"));
    if (storedMeme.imageUrl) {
      setMeme(storedMeme);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("memeDraft", JSON.stringify(meme));
  }, [meme]);

  function saveMemeToLibrary() {
    // --- save meme to Library
    const storedLibrary = JSON.parse(localStorage.getItem("memeLibrary"));
    const updatedMemeLibrary = [...storedLibrary, meme];
    localStorage.setItem("memeLibrary", JSON.stringify(updatedMemeLibrary));

    // --- TODO: show that image was saved

    // --- reset meme
    setTimeout(() => {
      setMeme({
        imageUrl: "",
        topText: "",
        bottomText: "",
        id: nanoid(),
      });
    }, 100);
  }

  return (
    <main>
      <div className="mt-3 mb-3 overflow-y-scroll h-[250px] flex justify-center">
        <div className="grid grid-cols-10 gap-1">
          {memeTemplates &&
            memeTemplates.map((item, index) => (
              <MemeCard
                key={item.id}
                img={item}
                selectMeme={() => selectMeme(index)}
              />
            ))}
        </div>
      </div>
      {meme.imageUrl && (
        <div>
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
            <div
              className="mx-auto p-1 border-black border-2 flex items-center justify-center relative h-[450px] w-[450px]"
              id="meme-screenshot"
            >
              <img src={meme.imageUrl} className="inset-0 h-full w-full" />
              <h2 className="meme--text absolute top-0">{meme.topText}</h2>
              <h2 className="meme--text absolute bottom-0">
                {meme.bottomText}
              </h2>
            </div>
            <div className="">
              <button className="meme--button" onClick={downloadMeme}>
                Download Meme
              </button>
              <button className="meme--button" onClick={saveMemeToLibrary}>
                Save Meme
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
