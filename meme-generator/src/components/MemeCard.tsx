export default function MemeCard({ img, selectMeme }) {
  return (
    <div className="m-0 border-4 border-black hover:border-indigo-400 ">
      <img
        src={img.url}
        alt={img.name}
        className="object-cover h-20 w-15"
        onClick={selectMeme}
      ></img>
    </div>
  );
}
