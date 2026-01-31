import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  OneToOne,
  JoinColumn,
  Index,
  Check,
} from 'typeorm';
import { ApiProperty, ApiHideProperty } from '@nestjs/swagger';
import { UserEntity } from './user.entity';

@Entity({ name: 'user_stats' })
@Check(`"average_score" >= 0 AND "average_score" <= 100`)
export class UserStatsEntity {
  @PrimaryGeneratedColumn()
  @ApiProperty({ example: 1, description: 'ID записи статистики' })
  id: number;

  @OneToOne(() => UserEntity, (user) => user.stats, {
    onDelete: 'CASCADE', // при удалении пользователя — удаляем статистику
  })
  @JoinColumn({ name: 'user_id' })
  @Index({ unique: true })
  @ApiHideProperty()
  user: UserEntity;

  @Column({ type: 'int', default: 0 })
  @ApiProperty({ example: 3, description: 'Кол-во курсов, на которые записан' })
  coursesEnrolled: number;

  @Column({ type: 'int', default: 0 })
  @ApiProperty({
    example: 2,
    description: 'Кол-во курсов, созданных пользователем',
  })
  coursesAuthored: number;

  @Column({ type: 'int', default: 0 })
  @ApiProperty({ example: 24, description: 'Всего уроков (для пользователя)' })
  lessonsTotal: number;

  @Column({ type: 'int', default: 0 })
  @ApiProperty({ example: 18, description: 'Пройденных уроков' })
  lessonsCompleted: number;

  @Column({ type: 'int', default: 0 })
  @ApiProperty({ example: 10, description: 'Всего тестов' })
  quizzesTotal: number;

  @Column({ type: 'int', default: 0 })
  @ApiProperty({ example: 8, description: 'Пройденных тестов' })
  quizzesPassed: number;

  @Column({
    name: 'average_score',
    type: 'numeric',
    precision: 5,
    scale: 2,
    default: 0,
  })
  @ApiProperty({ example: 76.5, description: 'Средний балл (0..100)' })
  averageScore: string; // numeric в PG -> string в TypeORM; при отдаче можно парсить в number

  @Column({ type: 'int', default: 0 })
  @ApiProperty({ example: 5, description: 'Серия дней подряд' })
  streakDays: number;

  @Column({ type: 'timestamptz', nullable: true })
  @ApiProperty({
    example: '2025-08-25T12:34:56.000Z',
    description: 'Последняя активность',
  })
  lastActiveAt: Date | null;
}
