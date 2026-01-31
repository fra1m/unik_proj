export enum Role {
  USER = 'user',
  STUDENT = 'student',
  ADMIN = 'admin',
  TEACHER = 'teacher',
}

export type UserModel = {
  id: number;
  email: string;
  name: string;
  role: Role;
  specializationId: number;
  counterAgentSpecializationIds?: number[] | null;
  phone?: string | null;
  birthDate?: string | null;
  city?: string | null;
  address?: string | null;
  photoUrl?: string | null;
  archivedAt?: string | null;
};
