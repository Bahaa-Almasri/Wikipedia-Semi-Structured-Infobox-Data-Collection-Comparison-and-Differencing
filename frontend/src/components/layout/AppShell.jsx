import Footer from "./Footer.jsx";
import Navbar from "./Navbar.jsx";

export default function AppShell({ children }) {
  return (
    <div className="app-shell">
      <Navbar />
      <main className="app-main">{children}</main>
      <Footer />
    </div>
  );
}
