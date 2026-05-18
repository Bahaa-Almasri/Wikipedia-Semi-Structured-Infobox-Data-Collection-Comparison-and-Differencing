import { Navigate, Route, Routes } from "react-router-dom";

import AppShell from "./components/layout/AppShell.jsx";
import ExplorePage from "./pages/ExplorePage.jsx";
import GuessGamePage from "./pages/GuessGamePage.jsx";
import HomePage from "./pages/HomePage.jsx";
import MatchmakerPage from "./pages/MatchmakerPage.jsx";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/matchmaker" element={<MatchmakerPage />} />
        <Route path="/explore" element={<ExplorePage />} />
        <Route path="/guess" element={<GuessGamePage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
