import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  BaseEntity,
} from 'typeorm';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

export enum UserHistoryAction {
  CREATED = 'created',
  UPDATED = 'updated',
  ARCHIVED = 'archived',
  RESTORED = 'restored',
}

@Entity({ name: 'user_history' })
export class UserHistoryEntity extends BaseEntity {
  @ApiProperty({ example: 1, description: 'ID записи истории' })
  @PrimaryGeneratedColumn()
  id: number;

  @ApiProperty({ example: 12, description: 'ID пользователя' })
  @Column({ type: 'int' })
  userId: number;

  @ApiPropertyOptional({
    example: 1,
    description: 'ID пользователя, выполнившего действие',
  })
  @Column({ type: 'int', nullable: true })
  actorId?: number | null;

  @ApiProperty({
    enum: UserHistoryAction,
    description: 'Тип события',
  })
  @Column({ type: 'enum', enum: UserHistoryAction })
  action: UserHistoryAction;

  @ApiPropertyOptional({ description: 'Изменения или метаданные события' })
  @Column({ type: 'jsonb', nullable: true })
  changes?: Record<string, unknown> | null;

  @CreateDateColumn()
  createdAt: Date;
}
