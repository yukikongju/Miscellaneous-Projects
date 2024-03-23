import { useState, useEffect } from "react";

export default function LibraryPage() {
  const [memeLibrary, setMemeLibrary] = useState([]);

  useEffect(() => {
    const savedLibrary = localStorage.getItem("memeLibrary");
    if (savedLibrary) {
      setMemeLibrary(JSON.parse(savedLibrary));
    }
  }, []);

  return (
    <main>
      <p>library page</p>
      {memeLibrary.length > 0 &&
        memeLibrary.map((meme) => <p>{meme.imageUrl}</p>)}
    </main>
  );
}
