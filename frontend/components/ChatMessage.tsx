import type { LegalAnswer } from "@/lib/api";
import CitationCard from "./CitationCard";

interface ChatMessageProps {
  role: "user" | "assistant";
  content?: string;
  answer?: LegalAnswer;
}

export default function ChatMessage({
  role,
  content,
  answer,
}: ChatMessageProps) {
  const isUser = role === "user";

  return (
    <div
      className={`message-row ${
        isUser
          ? "message-row-user"
          : "message-row-assistant"
      }`}
    >
      <div
        className={`message-bubble ${
          isUser
            ? "message-user"
            : "message-assistant"
        }`}
      >
        <div className="message-role">
          {isUser ? "Bạn" : "Vietnam Legal Assistant"}
        </div>

        {content && (
          <div className="message-content">
            {content}
          </div>
        )}

        {answer && (
          <>
            <div className="message-content answer-text">
              {answer.answer}
            </div>

            {answer.insufficient_evidence && (
              <div className="warning-box">
                Context hiện tại chưa cung cấp đủ
                căn cứ để trả lời chắc chắn.
              </div>
            )}

            {answer.note && (
              <div className="note-box">
                <strong>Lưu ý:</strong>{" "}
                {answer.note}
              </div>
            )}

            {answer.citations.length > 0 && (
              <div className="citations-section">
                <div className="section-title">
                  Nguồn pháp lý
                </div>

                <div className="citations-list">
                  {answer.citations.map(
                    (citation, index) => (
                      <CitationCard
                        key={
                          citation.chunk_id ??
                          `${citation.document_id}-${index}`
                        }
                        citation={citation}
                        index={index + 1}
                      />
                    ),
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}