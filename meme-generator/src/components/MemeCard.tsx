export default function MemeCard({ img }) {
  return (
    <div className="m-0 border-4 border-black hover:border-indigo-400 ">
      <img
        src={img.url}
        alt={img.name}
        className="object-cover h-20 w-15"
      ></img>
    </div>
  );
}
