export default function EmptyState({ title = "Nothing to show yet", message }) {
  return (
    <div className="state-card">
      <h3>{title}</h3>
      {message && <p>{message}</p>}
    </div>
  );
}
