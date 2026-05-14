export default function ErrorState({ message = "Something went wrong.", onRetry }) {
  return (
    <div className="state-card error-state">
      <h3>Could not load this view</h3>
      <p>{message}</p>
      {onRetry && (
        <button className="primary-button" type="button" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
