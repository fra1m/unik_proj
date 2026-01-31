import {
  IsInt,
  IsOptional,
  IsString,
  Min,
  IsArray,
  IsDateString,
} from 'class-validator';
import { ApiPropertyOptional } from '@nestjs/swagger';
import { Role } from './userListItem.dto';

export class UpdateUserDto {
  @ApiPropertyOptional({
    example: Role.USER,
    description: 'Роль пользователя (опционально)',
  })
  @IsString({ message: 'Должно быть строкой' })
  @IsOptional()
  role?: Role;

  @ApiPropertyOptional({
    example: 'Антон',
    description: 'Имя пользователя (опционально)',
  })
  @IsString({ message: 'Должно быть строкой' })
  @IsOptional()
  name?: string;

  @ApiPropertyOptional({
    example: 'example@asd.com',
    description: 'Почта пользователя (опционально)',
  })
  @IsString({ message: 'Должно быть строкой' })
  @IsOptional()
  email: string;

  @ApiPropertyOptional({
    example: 3,
    description: 'ID специализации (опционально)',
  })
  @IsOptional()
  @IsInt()
  @Min(1)
  specializationId?: number;

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
