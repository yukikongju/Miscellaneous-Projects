export default function MemeCard({ img, selectMeme }) {
  // className="object-cover h-20 w-15"

  return (
    <div className="m-0 border-4 border-black hover:border-indigo-400 ">
      <img src={img.url} alt={img.name} className="" onClick={selectMeme}></img>
    </div>
  );
}
