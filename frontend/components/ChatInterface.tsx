"use client";

import { useState, useRef, useEffect } from "react";
import { sendMessage } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  memoriesStored?: number;
}

export default function ChatInterface() {
  const [userId, setUserId] = useState("demo_user");
  const [model, setModel] = useState<"openai" | "anthropic">("openai");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setError("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    try {
      const res = await sendMessage({ user_id: userId, model, message: text });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.response, memoriesStored: res.memories_stored },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Controls */}
      <div className="flex gap-3 items-center flex-wrap">
        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-400">User ID</label>
          <input
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm w-36 focus:outline-none focus:border-indigo-500"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />
        </div>
        <div className="flex gap-2">
          {(["openai", "anthropic"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setModel(m)}
              className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
                model === m
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-white"
              }`}
            >
              {m === "openai" ? "GPT-4o" : "Anthropic"}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 h-[500px] overflow-y-auto p-4 flex flex-col gap-3">
        {messages.length === 0 && (
          <p className="text-gray-600 text-sm text-center mt-8">
            Start a conversation. Memories are extracted and stored automatically.
          </p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[75%] rounded-xl px-4 py-2.5 text-sm ${
                msg.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-100"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.memoriesStored !== undefined && msg.memoriesStored > 0 && (
                <p className="text-xs text-indigo-300 mt-1">
                  {msg.memoriesStored} memory{msg.memoriesStored !== 1 ? " facts" : " fact"} stored
                </p>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-xl px-4 py-2.5 text-sm text-gray-400 animate-pulse">
              Thinking...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {/* Input */}
      <div className="flex gap-2">
        <input
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500"
          placeholder="Type a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}
