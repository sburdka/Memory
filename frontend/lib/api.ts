const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Memory {
  id: string;
  user_id: string;
  content: string;
  memory_type: "episodic" | "semantic" | "reflective";
  importance_score: number;
  created_at: string;
}

export interface ChatRequest {
  user_id: string;
  model: "openai" | "claude";
  message: string;
}

export interface ChatResponse {
  response: string;
  memories_stored: number;
}

export async function sendMessage(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Chat request failed");
  }
  return res.json();
}

export async function getMemories(userId: string): Promise<Memory[]> {
  const res = await fetch(`${API_URL}/memory/${userId}`);
  if (!res.ok) throw new Error("Failed to fetch memories");
  return res.json();
}

export async function createMemory(
  userId: string,
  content: string,
  memoryType: string = "episodic"
): Promise<Memory> {
  const res = await fetch(`${API_URL}/memory`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, content, memory_type: memoryType }),
  });
  if (!res.ok) throw new Error("Failed to create memory");
  return res.json();
}

export async function searchMemories(userId: string, query: string, topK = 10): Promise<string[]> {
  const res = await fetch(`${API_URL}/memory/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, query, top_k: topK }),
  });
  if (!res.ok) throw new Error("Search failed");
  return res.json();
}
