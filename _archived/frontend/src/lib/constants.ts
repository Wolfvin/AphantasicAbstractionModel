export const TIER_COLORS: Record<string, string> = {
  seed: "#6b7280",
  novice: "#3b82f6",
  stable: "#10b981",
  confident: "#f59e0b",
  authority: "#ef4444",
};

export const TIER_LABELS: Record<string, string> = {
  seed: "Seed",
  novice: "Novice",
  stable: "Stable",
  confident: "Confident",
  authority: "Authority",
};

// Numeric tier colors/labels used across graph components
export const NUMERIC_TIER_COLORS: Record<number, string> = {
  1: '#00E5FF',
  2: '#FFB74D',
  3: '#FF5252',
};

export const NUMERIC_TIER_LABELS: Record<number, string> = {
  1: 'T1 — Stable',
  2: 'T2 — Alert',
  3: 'T3 — Critical',
};

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
