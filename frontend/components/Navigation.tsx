"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navigation() {
  const path = usePathname();

  const navLink = (href: string, label: string) => (
    <Link
      href={href}
      className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
        path === href
          ? "bg-indigo-600 text-white"
          : "text-gray-400 hover:text-white hover:bg-gray-800"
      }`}
    >
      {label}
    </Link>
  );

  return (
    <nav className="border-b border-gray-800 bg-gray-900">
      <div className="max-w-5xl mx-auto px-4 h-14 flex items-center gap-2">
        <span className="text-indigo-400 font-bold mr-6">MemoryOS</span>
        {navLink("/", "Chat")}
        {navLink("/dashboard", "Memory Dashboard")}
      </div>
    </nav>
  );
}
