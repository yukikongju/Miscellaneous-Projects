import { Outlet, NavLink } from "react-router-dom";

export default function NavBar() {
  const navbar = [
    { name: "Home", to: "/" },
    { name: "Create", to: "create" },
    { name: "Library", to: "library" },
  ];

  return (
    <>
      <header>
        <div className="w-full bg-gray-600">
          <nav className="flex text-center text-white px-5 py-3">
            <ul className="flex gap-x-2">
              {navbar.map((item) => (
                <li className="">
                  <NavLink
                    to={item.to}
                    className={({ isActive, isPending }) =>
                      isActive
                        ? "navbar--active"
                        : isPending
                        ? "navbar--pending"
                        : "navbar--inactive"
                    }
                  >
                    {item.name}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>
      <Outlet />
    </>
  );
}
