// import { Fragment } from "react";
// import { Disclosure, Menu, Transition } from "@headlessui/react";

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
        <div className="ml-10 flex text-center text-white">
          {navigation.map((item) => (
            <a key={item.name} href={item.href} className="p-4 mr-4">
              {item.name}
            </a>
          ))}
        </div>
      </div>
    </header>
  );
}
