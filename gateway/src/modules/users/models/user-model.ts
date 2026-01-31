import { Role } from 'src/common/decorators/roles-auth.decorator';

export type UserModel = {
  sub: number;
  email: string;
  name: string;
  role: Role;
  specializationId?: number | null;
  counterAgentSpecializationIds?: number[] | null;
  phone?: string | null;
  birthDate?: string | null;
  city?: string | null;
  address?: string | null;
  photoUrl?: string | null;
  archivedAt?: string | null;
};
