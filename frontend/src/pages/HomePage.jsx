import { Link } from "react-router-dom";

const experiences = [
  {
    to: "/matchmaker",
    title: "Find Your Country Match",
    text: "Answer a few questions and discover countries that fit your profile.",
  },
  {
    to: "/explore",
    title: "Explore Similar Countries",
    text: "Pick a country and browse the world through unexpected connections.",
  },
  {
    to: "/guess",
    title: "Guess the Country",
    text: "Use similarity clues to solve the mystery country.",
  },
];

export default function HomePage() {
  return (
    <div className="page-stack">
      <section className="hero-panel home-hero">
        <p className="eyebrow">CountryScope</p>
        <h1>Discover the world through hidden country connections.</h1>
        <p>
          Explore public country profiles, find places that match your interests,
          and play a similarity-powered guessing game.
        </p>
        <div className="button-row">
          <Link className="primary-button" to="/matchmaker">Start matching</Link>
          <Link className="ghost-button" to="/explore">Explore countries</Link>
        </div>
      </section>

      <section className="card-grid">
        {experiences.map((item) => (
          <Link key={item.to} to={item.to} className="feature-card">
            <p className="eyebrow">Experience</p>
            <h2>{item.title}</h2>
            <p>{item.text}</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
