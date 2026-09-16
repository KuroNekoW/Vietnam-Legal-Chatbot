import type { ChatResponse } from "@/lib/api";

interface DebugPanelProps {
  result: ChatResponse | null;
  open: boolean;
  onToggle: () => void;
}

export default function DebugPanel({
  result,
  open,
  onToggle,
}: DebugPanelProps) {
  if (!result) {
    return null;
  }

  return (
    <section className="debug-panel">
      <button
        type="button"
        className="debug-header"
        onClick={onToggle}
      >
        <span>RAG Debug</span>

        <span>
          {open ? "Ẩn" : "Hiện"}
        </span>
      </button>

      {open && (
        <div className="debug-content">
          <div className="debug-grid">
            <div className="debug-item">
              <span>Retrieved</span>
              <strong>
                {result.retrieved_count}
              </strong>
            </div>

            <div className="debug-item">
              <span>Reranked</span>
              <strong>
                {result.reranked_count}
              </strong>
            </div>

            <div className="debug-item">
              <span>Documents</span>
              <strong>
                {result.selected_document_count}
              </strong>
            </div>

            <div className="debug-item">
              <span>Chunks</span>
              <strong>
                {result.selected_chunk_count}
              </strong>
            </div>
          </div>

          {result.normalized_query && (
            <div className="normalized-query">
              <div className="debug-label">
                Normalized query
              </div>

              <div className="normalized-query-value">
                {result.normalized_query}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}