import { ApiProperty } from '@nestjs/swagger';
import { IsNumber } from 'class-validator';

export class DeleteUserDto {
  @ApiProperty({
    example: '1',
    description: 'Id пользователя',
  })
  @IsNumber({}, { message: 'Должно быть ичслом' })
  id: number;
}
