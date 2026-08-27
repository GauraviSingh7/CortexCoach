/** Prompts the coach might use next, for the most recent turn. */

import { Card, EmptyState } from "./ui/primitives";

export function Suggestions({ suggestions }: { suggestions: string[] }) {
  return (
    <Card title="What you might try" subtitle="Prompted by the last turn">
      {suggestions.length === 0 ? (
        <EmptyState>Nothing to suggest yet.</EmptyState>
      ) : (
        <ul className="flex flex-col gap-2.5">
          {suggestions.map((suggestion, index) => (
            <li
              key={index}
              className="border-l-2 border-sage/40 pl-3 text-[14px] leading-relaxed text-ink"
            >
              {suggestion}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
