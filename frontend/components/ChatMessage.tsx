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

  if (isUser) {
    return (
      <div className="message-row message-row-user">
        <div className="message-user">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className="message-row message-row-assistant">
      <div className="message-assistant">
        <div className="message-role">
          Trợ lý
        </div>

        {answer && (
          <div className="answer-text">
            {answer.answer}
          </div>
        )}

        {answer?.insufficient_evidence && (
          <div className="warning-box">
            Cơ sở dữ liệu hiện chưa cung cấp đủ căn cứ để trả
            lời chắc chắn cho câu hỏi này.
          </div>
        )}

        {answer?.note && (
          <div className="note-box">
            <strong>Lưu ý.</strong>{" "}
            {answer.note}
          </div>
        )}

        {answer && answer.citations.length > 0 && (
          <div className="citations-section">
            <div className="section-title">
              Căn cứ pháp lý
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
      </div>
    </div>
  );
}
