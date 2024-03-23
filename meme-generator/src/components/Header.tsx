// import { Fragment } from "react";
// import { Disclosure, Menu, Transition } from "@headlessui/react";

import { NavLink } from "react-router-dom";

export default function Header() {
  const navigation = [
    { name: "Browse", href: "#", current: true },
    { name: "Create", href: "#", current: false },
    { name: "Library", href: "#", current: false },
    { name: "Profile", href: "#", current: false },
  ];

  return (
    <header>
      <div className="w-full bg-gray-800">
        <nav className="ml-10 flex text-center text-white">
          <ul>
            {navigation.map((item) => (
              <li>
                <NavLink to="/">{item.name}</NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}
