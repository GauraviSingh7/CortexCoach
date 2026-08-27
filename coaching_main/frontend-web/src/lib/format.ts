/** Small formatting helpers shared across panels. */

/** Render a score, or say plainly that it is unavailable. */
export const metric = (value: number | undefined | null, digits = 2): string =>
  typeof value === "number" && value > 0 ? value.toFixed(digits) : "Not Available";

export const percent = (value: number | undefined | null, digits = 0): string =>
  typeof value === "number" ? `${(value * 100).toFixed(digits)}%` : "-";

export const clockTime = (epochSeconds: number): string =>
  new Date(epochSeconds * 1000).toLocaleTimeString([], {
    minute: "2-digit",
    second: "2-digit",
  });

/** Highest-scoring emotion in a distribution, or null when there is none. */
export function dominantEmotion(
  trend: Record<string, number>,
): { label: string; score: number } | null {
  const entries = Object.entries(trend ?? {});
  if (!entries.length) return null;
  const [label, score] = entries.reduce((a, b) => (b[1] > a[1] ? b : a));
  return { label, score };
}

export const titleCase = (value: string): string =>
  value.charAt(0).toUpperCase() + value.slice(1);
