export default function LibraryCard({ meme, selectMeme }) {
  return (
    <div className="mx-auto border-4 border-black h-[120px] 2-[120px] hover:border-indigo-400 flex flex-col justify-between">
      <h2 className="meme--text text-sm">{meme.topText}</h2>
      <img src={meme.imageUrl} alt={meme.id} onClick={selectMeme} />
      <h2 className="meme--text text-sm">{meme.bottomText}</h2>
    </div>
  );
}
