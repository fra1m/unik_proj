import { IsDate, IsNumber } from 'class-validator';

export class ApplyQuizStatsDto {
  @IsNumber()
  quizzesTotal: number;

  @IsNumber()
  quizzesPassed: number;

  @IsNumber()
  averageScore: number;

  @IsNumber()
  lessonsTotal: number;

  @IsNumber()
  lessonsCompleted: number;

  @IsDate()
  lastActiveAt?: Date;
}
