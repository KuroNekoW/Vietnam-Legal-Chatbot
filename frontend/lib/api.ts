export interface AnswerCitation {
  chunk_id: string | null;
  document_id: number | null;
  title: string;
  article: string | null;
  clause: string | null;
  point: string | null;
  issuance_date: string | null;
}

export interface LegalAnswer {
  answer: string;
  citations: AnswerCitation[];
  insufficient_evidence: boolean;
  note: string | null;
}

export interface ChatRequest {
  query: string;
}

export interface ChatResponse {
  query: string;
  normalized_query: string | null;

  retrieved_count: number;
  reranked_count: number;

  selected_document_count: number;
  selected_chunk_count: number;

  answer: LegalAnswer;
}

declare const process: {
  env: {
    NEXT_PUBLIC_API_URL?: string;
  };
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";

export async function sendChat(
  query: string,
): Promise<ChatResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
      } satisfies ChatRequest),
    },
  );

  if (!response.ok) {
    let message = `API error: ${response.status}`;

    try {
      const errorData = await response.json();

      if (typeof errorData?.detail === "string") {
        message = errorData.detail;
      }
    } catch {
      // Keep fallback error message.
    }

    throw new Error(message);
  }

  return response.json() as Promise<ChatResponse>;
}