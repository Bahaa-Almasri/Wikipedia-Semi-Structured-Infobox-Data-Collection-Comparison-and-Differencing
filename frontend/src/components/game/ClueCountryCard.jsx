export default function ClueCountryCard({ clue }) {
  return (
    <article className="clue-card">
      <p className="eyebrow">Neighbor clue</p>
      <h3>{clue.display_name}</h3>
    </article>
  );
}
