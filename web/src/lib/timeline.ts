import type { TimelineData, TimelineItem } from "../types";

export function isTimelineApproved(
  data: TimelineData | null | undefined,
): boolean {
  return Boolean(
    data &&
      data.status === "approved" &&
      Array.isArray(data.items) &&
      data.items.length > 0,
  );
}

export function getApprovedTimelineItems(
  data: TimelineData | null | undefined,
): TimelineItem[] {
  if (!isTimelineApproved(data)) {
    return [];
  }
  return data!.items;
}
