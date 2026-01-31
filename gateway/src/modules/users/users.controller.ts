import {
  Body,
  Controller,
  Get,
  Param,
  Post,
  Query,
} from '@nestjs/common';
import { ApiOperation, ApiTags } from '@nestjs/swagger';
import { ReqId } from 'src/common/http/req-id.decorator';
import { UsersService } from './users.service';
import { Role } from 'src/common/decorators/roles-auth.decorator';
import { UserModel } from './models/user-model';

class CreateUserMinimalDto {
  email!: string;
  name!: string;
  role?: Role;
  specializationId?: number;
}

@ApiTags('users')
@Controller('users')
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @ApiOperation({ summary: 'Создать пользователя (минимально)' })
  @Post()
  async create(
    @Body() dto: CreateUserMinimalDto,
    @ReqId() reqId: string,
  ): Promise<UserModel> {
    const meta = { requestId: reqId };
    const role = dto.role ?? Role.USER;
    const specializationId = Number.isFinite(dto.specializationId)
      ? Number(dto.specializationId)
      : 0;

    return this.usersService.createUser(meta, {
      email: dto.email.trim().toLowerCase(),
      name: dto.name,
      role,
      specializationId,
      counterAgentSpecializationIds: [],
      phone: null,
      birthDate: null,
      city: null,
      address: null,
      photoUrl: null,
    });
  }

  @ApiOperation({ summary: 'Найти пользователя по email' })
  @Get('by-email')
  async byEmail(
    @Query('email') email: string,
    @ReqId() reqId: string,
  ): Promise<UserModel> {
    const meta = { requestId: reqId };
    return this.usersService.getByEmail(meta, {
      email: email.trim().toLowerCase(),
    });
  }

  @ApiOperation({ summary: 'Найти пользователя по id' })
  @Get(':id')
  async byId(
    @Param('id') id: string,
    @ReqId() reqId: string,
  ): Promise<UserModel> {
    const meta = { requestId: reqId };
    return this.usersService.getUserById(meta, Number(id));
  }
}
