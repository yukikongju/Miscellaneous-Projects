import { useState, useEffect } from "react";
import LibraryCard from "../components/LibraryCard";

export default function LibraryPage() {
  const [memeLibrary, setMemeLibrary] = useState([]);
  const [meme, setMeme] = useState({});
  const [memeIndex, setMemeIndex] = useState(-1);

  // useEffect(() => {
  //   // localStorage.setItem("memeLibrary", JSON.stringify(memeLibrary));
  // }, [memeLibrary]);

  useEffect(() => {
    const savedLibrary = localStorage.getItem("memeLibrary");
    if (savedLibrary) {
      setMemeLibrary(JSON.parse(savedLibrary));
    }
  }, []);

  function selectMeme(index) {
    const meme = memeLibrary[index];
    setMeme(meme);
    setMemeIndex(index);
    // alert(JSON.stringify(meme));
  }

  function handleMeme(event) {
    const { name, value } = event.target;
    setMeme((prevMeme) => ({
      ...prevMeme,
      [name]: value,
    }));
  }

  function updateLibraryMeme() {
    // TODO
  }

  function deleteLibraryMeme() {
    setMemeLibrary((prevLibrary) => {
      const updatedLibrary = prevLibrary.filter(
        (item, index) => index !== memeIndex
      );
      localStorage.setItem("memeLibrary", JSON.stringify(updatedLibrary));
      return updatedLibrary;
    });
    setMemeIndex(-1);
    setMeme({});

    // update localstorage => cannot update here because delay
    // localStorage.setItem("memeLibrary", JSON.stringify(memeLibrary));
  }

  return (
    <main>
      <div className="mt-3 mb-3 overflow-y-scroll h-[250px] flex justify-center">
        <div className="grid grid-cols-10 gap-1">
          {memeLibrary.length > 0 &&
            memeLibrary.map((item, index) => (
              <LibraryCard
                key={item.id}
                meme={item}
                selectMeme={() => selectMeme(index)}
              />
            ))}
          {memeLibrary.length == 0 && <p>Empty Library!</p>}
        </div>
      </div>
      {meme.id && (
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
            <h2 className="meme--text absolute bottom-0">{meme.bottomText}</h2>
          </div>
          <div>
            <button className="meme--button" onClick={updateLibraryMeme}>
              Update Meme
            </button>
            <button className="meme--button" onClick={deleteLibraryMeme}>
              Delete Meme
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
