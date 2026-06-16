import MemoryDashboard from "@/components/MemoryDashboard";

export default function DashboardPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Memory Dashboard</h1>
        <p className="text-sm text-gray-400 mt-1">
          View all stored memories grouped by type. Reflective memories are auto-generated insights.
        </p>
      </div>
      <MemoryDashboard />
    </div>
  );
}
