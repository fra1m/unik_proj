export enum Role {
  USER = 'user',
  STUDENT = 'student',
  ADMIN = 'admin',
  TEACHER = 'teacher',
}

export type UserListItemDto = {
  id: number;
  name: string;
  role: Role;
  email: string;
};
