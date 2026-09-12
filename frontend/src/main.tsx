import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { applyDocumentLang } from "./i18n";
import "./styles.css";

applyDocumentLang();

const el = document.getElementById("root");
if (!el) throw new Error("找不到 #root");
createRoot(el).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
