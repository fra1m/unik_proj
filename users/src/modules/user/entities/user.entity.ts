import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  BaseEntity,
  OneToOne,
  CreateDateColumn,
  UpdateDateColumn,
  DeleteDateColumn,
} from 'typeorm';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { Role } from '../dto/userListItem.dto';
import { UserStatsEntity } from './user-stats.entity';

// import { UserStatsEntity } from './user-stats.entity';
// import { TokenEntity } from 'src/modules/auth/entities/token.entity';
// import { CourseEntity } from 'src/modules/courses/entities/course.entity';
// import { QuizEntity } from 'src/modules/quiz/entities/quiz.entity';
// import { SpecializationEntity } from 'src/modules/specialization/entities/specialization.entity';
// import { QuizAttemptEntity } from 'src/modules/analytics/entities/quiz-attempt.entity';

@Entity({ name: 'users' })
export class UserEntity extends BaseEntity {
  @ApiProperty({ example: '1', description: 'Уникальный идентификатор' })
  @PrimaryGeneratedColumn()
  id: number;

  @ApiProperty({
    example: 'TestDeveloper',
    description: 'Имя пользователя',
  })
  @Column({ nullable: false })
  name: string;

  @ApiProperty({
    example: `user_${Math.random().toString(36).substring(7)}@example.com`,
    description: 'Почта пользователя',
  })
  @Column({ unique: true, nullable: false })
  email: string;

  @ApiProperty({
    enum: Role,
    example: Role.USER,
    description: 'Роль пользователя',
  })
  @Column({ type: 'enum', enum: Role, default: Role.USER })
  role: Role;

  @ApiProperty({ description: 'ID специализации пользователя' })
  @Column({ type: 'integer' })
  specializationId: number | null;

  @ApiPropertyOptional({
    description: 'ID специализаций контр-агента пользователя',
    type: [Number],
  })
  @Column('int', {
    array: true,
    nullable: true,
    default: () => 'ARRAY[]::INTEGER[]',
  })
  counterAgentSpecializationIds: number[];

  @ApiPropertyOptional({ description: 'Телефон пользователя' })
  @Column({ type: 'varchar', length: 32, nullable: true })
  phone?: string | null;

  @ApiPropertyOptional({ description: 'Дата рождения пользователя' })
  @Column({ type: 'date', nullable: true })
  birthDate?: string | null;

  @ApiPropertyOptional({ description: 'Город пользователя' })
  @Column({ type: 'varchar', length: 120, nullable: true })
  city?: string | null;

  @ApiPropertyOptional({ description: 'Адрес пользователя' })
  @Column({ type: 'varchar', length: 255, nullable: true })
  address?: string | null;

  @ApiPropertyOptional({ description: 'URL фото пользователя' })
  @Column({ type: 'text', nullable: true })
  photoUrl?: string | null;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @DeleteDateColumn()
  archivedAt?: Date | null;

  // @OneToMany(() => QuizAttemptEntity, (att) => att.user, {
  //   onDelete: 'CASCADE',
  // })
  // attemt: QuizAttemptEntity | null;

  // @ManyToMany(() => CourseEntity, (course) => course.students)
  // @JoinTable({
  //   name: 'user_courses',
  //   joinColumns: [{ name: 'user_id', referencedColumnName: 'id' }],
  //   inverseJoinColumns: [{ name: 'course_id', referencedColumnName: 'id' }],
  // })
  // enrolledCourses: CourseEntity[];

  // @ApiHideProperty()
  // @OneToMany(() => CourseEntity, (course) => course.teacher)
  // authoredCourses: CourseEntity[];

  // @ApiProperty({
  //   example: [QuizEntity],
  //   description: 'Массив токенов пользователя',
  // })
  // @OneToMany(() => QuizEntity, (quiz) => quiz.user, {
  //   cascade: true,
  //   onDelete: 'CASCADE',
  // })
  // quizzes: QuizEntity[];

  @OneToOne(() => UserStatsEntity, (stats) => stats.user, {
    cascade: ['insert', 'update'], // создаём/обновляем stats вместе с пользователем
    eager: true, // автоматически подтягивать stats (опционально)
  })
  @ApiProperty({
    type: () => UserStatsEntity,
    description: 'Статистика пользователя',
  })
  stats: UserStatsEntity;
}
