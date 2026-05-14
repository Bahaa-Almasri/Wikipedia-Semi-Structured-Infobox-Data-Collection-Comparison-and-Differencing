import { NavLink } from "react-router-dom";

const links = [
  { to: "/matchmaker", label: "Find Your Match" },
  { to: "/explore", label: "Explore" },
  { to: "/guess", label: "Guess" },
];

export default function Navbar() {
  return (
    <header className="navbar">
      <NavLink to="/" className="brand">
        <span className="brand-mark">CS</span>
        <span>CountryScope</span>
      </NavLink>
      <nav className="nav-links">
        {links.map((link) => (
          <NavLink key={link.to} to={link.to} className="nav-link">
            {link.label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
