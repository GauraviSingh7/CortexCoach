/** Coaching suggestions for the most recent turn. */

import { Card, EmptyState } from "./ui/primitives";

export function Suggestions({ suggestions }: { suggestions: string[] }) {
  return (
    <Card title="Suggestions" subtitle="For the latest turn">
      {suggestions.length === 0 ? (
        <EmptyState>No suggestions yet.</EmptyState>
      ) : (
        <ol className="flex flex-col gap-2">
          {suggestions.map((suggestion, index) => (
            <li
              key={index}
              className="rounded-lg bg-[--color-panel-soft] px-3 py-2 text-sm leading-relaxed"
            >
              {suggestion}
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}
