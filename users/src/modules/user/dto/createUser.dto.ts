import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import {
  IsEmail,
  IsNumber,
  IsOptional,
  IsString,
  IsArray,
  IsInt,
  IsDateString,
} from 'class-validator';
import { Role } from './userListItem.dto';

// import { Role } from '../entities/user.entity';

export class CreateUserDto {
  @ApiProperty({
    example: 'user_uf3h4u@example.com',
    description: 'Почта пользователя',
  })
  @IsString({ message: 'Должно быть строкой' })
  @IsEmail({}, { message: 'Не корректный email' })
  email: string;

  @ApiProperty({ example: 'Антон', description: 'Имя пользователя' })
  @IsString({ message: 'Должно быть строкой' })
  name: string;

  @ApiProperty({
    enum: Role,
    example: Role.USER,
    description: 'Роль пользователя',
  })
  @IsString({ message: 'Должно быть строкой' })
  role: Role;

  @ApiProperty({
    example: 1,
    description: 'Роль пользователя',
  })
  @IsNumber({}, { message: 'Должно быть числом' })
  specializationId: number | null;

  @ApiPropertyOptional({
    example: [3, 4],
    description: 'ID специализаций контр-агента',
  })
  @IsOptional()
  @IsArray()
  @IsInt({ each: true })
  counterAgentSpecializationIds?: number[] | null;

  @ApiPropertyOptional({ example: '+79990001122', description: 'Телефон' })
  @IsOptional()
  @IsString()
  phone?: string | null;

  @ApiPropertyOptional({ example: '1992-05-12', description: 'Дата рождения' })
  @IsOptional()
  @IsDateString()
  birthDate?: string | null;

  @ApiPropertyOptional({ example: 'Москва', description: 'Город' })
  @IsOptional()
  @IsString()
  city?: string | null;

  @ApiPropertyOptional({ example: 'ул. Пример, 1', description: 'Адрес' })
  @IsOptional()
  @IsString()
  address?: string | null;

  @ApiPropertyOptional({ example: 'https://...', description: 'Фото (URL)' })
  @IsOptional()
  @IsString()
  photoUrl?: string | null;
}
