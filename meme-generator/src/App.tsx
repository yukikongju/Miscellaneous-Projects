import { useState } from "react";
import "./App.css";
import Header from "./components/Header";
import {
  RouterProvider,
  createBrowserRouter,
  createRoutesFromElements,
  Route,
} from "react-router-dom";

import NavBar from "./components/NavBar";
import HomePage from "./pages/HomePage";
import CreatePage from "./pages/CreatePage";
import LibraryPage from "./pages/LibraryPage";

const router = createBrowserRouter(
  createRoutesFromElements(
    <Route path="/" element={<NavBar />}>
      <Route index element={<HomePage />} />
      <Route path="create" element={<CreatePage />} />
      <Route path="library" element={<LibraryPage />} />
    </Route>
  )
);

function App() {
  return <RouterProvider router={router} />;
}

export default App;
