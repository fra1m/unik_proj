import { UserModel } from './user-model';

export type UserStatsModel = {
  // id: number;

  // user: UserModel;

  coursesEnrolled: number;

  coursesAuthored: number;

  lessonsTotal: number;

  lessonsCompleted: number;

  quizzesTotal: number;

  quizzesPassed: number;

  averageScore: string;

  streakDays: number;

  lastActiveAt: Date | null;
};
