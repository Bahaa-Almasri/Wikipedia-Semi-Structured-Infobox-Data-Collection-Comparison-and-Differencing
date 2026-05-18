export default function LoadingState({ message = "Loading..." }) {
  return (
    <div className="state-card">
      <div className="spinner" aria-hidden="true" />
      <p>{message}</p>
    </div>
  );
}
