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
        <span>Thông tin truy xuất</span>

        <span>
          {open ? "Ẩn" : "Hiện"}
        </span>
      </button>

      {open && (
        <div className="debug-content">
          <div className="debug-grid">
            <div className="debug-item">
              <span>Đã truy xuất</span>
              <strong>
                {result.retrieved_count}
              </strong>
            </div>

            <div className="debug-item">
              <span>Đã xếp hạng lại</span>
              <strong>
                {result.reranked_count}
              </strong>
            </div>

            <div className="debug-item">
              <span>Văn bản</span>
              <strong>
                {result.selected_document_count}
              </strong>
            </div>

            <div className="debug-item">
              <span>Đoạn trích</span>
              <strong>
                {result.selected_chunk_count}
              </strong>
            </div>
          </div>

          {result.normalized_query && (
            <div className="normalized-query">
              <div className="debug-label">
                Câu truy vấn đã chuẩn hóa
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
