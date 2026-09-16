"use client";

import {
  FormEvent,
  KeyboardEvent,
  useRef,
  useState,
} from "react";

import {
  sendChat,
  type ChatResponse,
} from "@/lib/api";

import ChatMessage from "./ChatMessage";
import DebugPanel from "./DebugPanel";
import SealMark from "./SealMark";
import ThemeToggle from "./ThemeToggle";

interface ChatHistoryItem {
  id: number;
  role: "user" | "assistant";
  content?: string;
  answer?: ChatResponse["answer"];
}

const EXAMPLE_QUESTIONS = [
  "Goi y gi gio :((",
];

export default function Chat() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<
    ChatHistoryItem[]
  >([]);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const [lastResult, setLastResult] =
    useState<ChatResponse | null>(null);

  const [debugOpen, setDebugOpen] =
    useState(false);

  const nextMessageId =
    useRef(0);

  async function handleSubmit(
    event?: FormEvent,
  ) {
    event?.preventDefault();

    const query = input.trim();

    if (!query || loading) {
      return;
    }

    setError(null);
    setInput("");

    const userMessage: ChatHistoryItem = {
      id: ++nextMessageId.current,
      role: "user",
      content: query,
    };

    setHistory((previous) => [
      ...previous,
      userMessage,
    ]);

    setLoading(true);

    try {
      const result = await sendChat(query);

      setLastResult(result);

      const assistantMessage: ChatHistoryItem = {
        id: ++nextMessageId.current,
        role: "assistant",
        answer: result.answer,
      };

      setHistory((previous) => [
        ...previous,
        assistantMessage,
      ]);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Đã xảy ra lỗi không xác định.";

      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();
      void handleSubmit();
    }
  }

  function clearChat() {
    setHistory([]);
    setLastResult(null);
    setError(null);
    setInput("");
  }

  return (
    <main className="chat-page">
      <div className="chat-shell">
        <header className="chat-header">
          <div className="brand">
            <SealMark className="brand-seal" />

            <div className="brand-text">
              <div className="brand-title">
                Trợ lý Pháp luật
              </div>

              <div className="brand-subtitle">
                Tra cứu văn bản quy phạm pháp luật Việt Nam
              </div>
            </div>
          </div>

          <div className="header-actions">
            {history.length > 0 && (
              <button
                type="button"
                className="clear-button"
                onClick={clearChat}
              >
                Xóa hội thoại
              </button>
            )}

            <ThemeToggle />
          </div>
        </header>

        <section className="chat-body">
          {history.length === 0 ? (
            <div className="welcome">
              <SealMark className="welcome-seal" />

              <h1>
                Tra cứu quy định pháp luật
              </h1>

              <p>
                Đặt câu hỏi bằng tiếng Việt. Hệ thống tìm kiếm
                và xếp hạng các quy định liên quan trước khi
                soạn câu trả lời kèm căn cứ pháp lý cụ thể.
              </p>

              <div className="example-list">
                <span className="example-label">
                  Câu hỏi gợi ý
                </span>

                {EXAMPLE_QUESTIONS.map((question) => (
                  <button
                    key={question}
                    type="button"
                    onClick={() => setInput(question)}
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {history.map((message) => (
                <ChatMessage
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  answer={message.answer}
                />
              ))}

              {loading && (
                <div className="message-row message-row-assistant">
                  <div className="message-assistant">
                    <div className="message-role">
                      Trợ lý
                    </div>

                    <div className="loading-state">
                      <span className="loading-dot" />
                      <span className="loading-dot" />
                      <span className="loading-dot" />

                      <span className="loading-text">
                        Đang xử lý câu hỏi...
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="error-box">
                  <strong>Không thể xử lý:</strong>{" "}
                  {error}
                </div>
              )}
            </div>
          )}

          {lastResult && (
            <DebugPanel
              result={lastResult}
              open={debugOpen}
              onToggle={() =>
                setDebugOpen(
                  (current) => !current,
                )
              }
            />
          )}
        </section>

        <form
          className="chat-input-area"
          onSubmit={handleSubmit}
        >
          <div className="input-wrapper">
            <textarea
              value={input}
              onChange={(event) =>
                setInput(event.target.value)
              }
              onKeyDown={handleKeyDown}
              placeholder="Nhập câu hỏi pháp luật..."
              rows={1}
              disabled={loading}
              aria-label="Câu hỏi pháp luật"
            />

            <button
              type="submit"
              className="send-button"
              disabled={
                loading ||
                input.trim().length === 0
              }
              aria-label="Gửi câu hỏi"
            >
              {loading ? "..." : "➤"}
            </button>
          </div>

          <div className="input-hint">
            Enter để gửi · Shift + Enter để xuống dòng
          </div>
        </form>
      </div>
    </main>
  );
}
