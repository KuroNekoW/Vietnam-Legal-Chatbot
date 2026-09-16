import type { AnswerCitation } from "@/lib/api";

interface CitationCardProps {
  citation: AnswerCitation;
  index: number;
}

export default function CitationCard({
  citation,
  index,
}: CitationCardProps) {
  return (
    <details className="citation-card">
      <summary className="citation-summary">
        <span className="citation-index">
          {index}
        </span>

        <span className="citation-title">
          {citation.title}
        </span>
      </summary>

      <div className="citation-body">
        <div className="citation-location">
          {citation.article && (
            <span>{citation.article}</span>
          )}

          {citation.clause && (
            <span>{citation.clause}</span>
          )}

          {citation.point && (
            <span>{citation.point}</span>
          )}
        </div>

        {citation.document_id !== null && (
          <div className="citation-meta">
            Document ID: {citation.document_id}
          </div>
        )}

        {citation.chunk_id && (
          <div className="citation-meta">
            Chunk ID: {citation.chunk_id}
          </div>
        )}

        {citation.issuance_date && (
          <div className="citation-meta">
            Ngày ban hành: {citation.issuance_date}
          </div>
        )}
      </div>
    </details>
  );
}