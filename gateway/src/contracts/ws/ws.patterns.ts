export const WS_PATTERNS = {
  WS_BROADCAST: 'ws.broadcast',
  WS_TICKET_VERIFY: 'auth.wsTicket.verify',
} as const;

export type TargetKind =
  | 'broadcast'
  | 'user'
  | 'role'
  | 'org'
  | 'group'
  | 'course'
  | 'spec';

export interface WsEvent<T = any> {
  type: string; // например: 'user.updated'
  target: { kind: TargetKind; ids?: (number | string)[] };
  payload: T;
  meta?: { requestId?: string; producer?: string; ts?: number };
}

export type WsClaims = {
  userId: number | string;
  role?: string;
  orgId?: number | string | null;
  specId?: number | string | null;
  groupIds?: (number | string)[];
};

export const room = {
  user: (id: number | string) => `user:${id}`,
  role: (r: string) => `role:${r}`,
  org: (id: number | string) => `org:${id}`,
  group: (id: number | string) => `group:${id}`,
  course: (id: number | string) => `course:${id}`,
  spec: (id: number | string) => `spec:${id}`,
};
