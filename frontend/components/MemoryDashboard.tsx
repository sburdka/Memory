"use client";

import { useState, useCallback } from "react";
import { getMemories, type Memory } from "@/lib/api";

const TYPE_COLORS: Record<string, string> = {
  episodic: "bg-blue-900 text-blue-300 border-blue-700",
  semantic: "bg-green-900 text-green-300 border-green-700",
  reflective: "bg-purple-900 text-purple-300 border-purple-700",
};

export default function MemoryDashboard() {
  const [userId, setUserId] = useState("demo_user");
  const [memories, setMemories] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);

  const fetchMemories = useCallback(async () => {
    if (!userId.trim()) return;
    setLoading(true);
    setError("");
    try {
      const data = await getMemories(userId);
      setMemories(data);
      setLoaded(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [userId]);

  const byType = memories.reduce<Record<string, Memory[]>>((acc, m) => {
    (acc[m.memory_type] ??= []).push(m);
    return acc;
  }, {});

  return (
    <div className="flex flex-col gap-6">
      {/* Controls */}
      <div className="flex gap-3 items-center">
        <label className="text-sm text-gray-400">User ID</label>
        <input
          className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm w-40 focus:outline-none focus:border-indigo-500"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && fetchMemories()}
        />
        <button
          onClick={fetchMemories}
          disabled={loading}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white px-4 py-1.5 rounded text-sm font-medium transition-colors"
        >
          {loading ? "Loading..." : "Load Memories"}
        </button>
        {loaded && (
          <span className="text-sm text-gray-500">{memories.length} total</span>
        )}
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {loaded && memories.length === 0 && (
        <p className="text-gray-500 text-sm">No memories found for this user.</p>
      )}

      {/* Memory groups */}
      {(["reflective", "semantic", "episodic"] as const).map((type) => {
        const group = byType[type] ?? [];
        if (!loaded || group.length === 0) return null;
        return (
          <div key={type} className="bg-gray-900 rounded-xl border border-gray-800 p-5">
            <div className="flex items-center gap-2 mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-300 capitalize">
                {type}
              </h2>
              <span className={`text-xs px-2 py-0.5 rounded-full border ${TYPE_COLORS[type]}`}>
                {group.length}
              </span>
            </div>
            <ul className="flex flex-col gap-2">
              {group.map((m) => (
                <li key={m.id} className="flex items-start gap-3 text-sm">
                  <span className="text-gray-600 mt-0.5 shrink-0">
                    {new Date(m.created_at).toLocaleDateString()}
                  </span>
                  <span className="text-gray-200">{m.content}</span>
                  {m.importance_score > 1 && (
                    <span className="ml-auto text-xs text-purple-400 shrink-0">
                      ★ {m.importance_score.toFixed(1)}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </div>
  );
}
