import ChatInterface from "@/components/ChatInterface";

export default function ChatPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Chat</h1>
        <p className="text-sm text-gray-400 mt-1">
          Switch between GPT-4o and Anthropic. Memories persist across both.
        </p>
      </div>
      <ChatInterface />
    </div>
  );
}
