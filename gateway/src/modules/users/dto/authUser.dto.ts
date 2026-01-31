import { ApiProperty } from '@nestjs/swagger';
import { IsEmail, IsString, Length } from 'class-validator';

export class AuthUserDto {
  @ApiProperty({
    description: 'Почта пользователя',
  })
  @IsString({ message: 'Должно быть строкой' })
  @IsEmail({}, { message: 'Не корректный логин' })
  email: string;

  @ApiProperty({
    example: 'pass123',
    description: 'Пароль пользователя',
    minLength: 6,
    maxLength: 16,
  })
  @IsString({ message: 'Должно быть строкой' })
  @Length(6, 16, {
    message: 'Не корректный пароль',
  })
  password: string;
}
