import { Controller } from '@nestjs/common';
import { MessagePattern, Payload, RpcException } from '@nestjs/microservices';
import { UserService } from './user.service';
import { CreateUserDto } from './dto/createUser.dto';
import { InjectPinoLogger, PinoLogger } from 'nestjs-pino';
import { USERS_PATTERNS } from 'src/contracts/users.patterns';
import { AuthUserDto } from './dto/authUser.dto';
import { ApplyQuizStatsDto } from './dto/applyQuizStats.dto';
import { UpdateUserDto } from './dto/updateUser.dto';

@Controller()
export class UserController {
  constructor(
    @InjectPinoLogger(UserController.name) private readonly logger: PinoLogger,
    private readonly userService: UserService,
  ) {}

  @MessagePattern(USERS_PATTERNS.GET_BY_EMAIL)
  async getByEmail(
    @Payload() data: { meta: { requestId: string }; authUserDto: AuthUserDto },
  ) {
    this.logger.info(
      {
        rid: data.meta?.requestId,
        dto: { ...data.authUserDto },
      },
      `${USERS_PATTERNS.GET_BY_EMAIL} received`,
    );

    try {
      const user = await this.userService.getUserByEmail(
        data.authUserDto.email,
      );

      this.logger.info(
        { rid: data.meta?.requestId, user },
        `${USERS_PATTERNS.GET_BY_EMAIL} succeeded`,
      );

      return user;
    } catch (e: any) {
      this.logger.error(
        { rid: data.meta?.requestId, err: e },
        `${USERS_PATTERNS.GET_BY_EMAIL} failed`,
      );
      throw new RpcException({ message: e?.message ?? 'Get user failed' });
    }
  }

  @MessagePattern(USERS_PATTERNS.CREATE)
  async create(
    @Payload()
    data: {
      meta: { requestId: string };
      createUserDto: CreateUserDto;
    },
  ) {
    this.logger.info(
      {
        rid: data.meta?.requestId,
        dto: { ...data.createUserDto, password: '[REDACTED]' },
      },
      `${USERS_PATTERNS.CREATE} received`,
    );

    try {
      // data.createUserDto.role = Role.USER;
      const user = await this.userService.createUser(data.createUserDto);

      return user;
    } catch (e: any) {
      this.logger.error(
        { rid: data.meta?.requestId, err: e },
        `${USERS_PATTERNS.CREATE} failed`,
      );
      throw new RpcException({ message: e?.message ?? 'Create users failed' });
    }
  }

  @MessagePattern(USERS_PATTERNS.BY_ID)
  async getUserById(
    @Payload() data: { meta: { requestId: string }; id: number },
  ) {
    this.logger.info(
      {
        rid: data.meta?.requestId,
        dto: data.id,
      },
      `${USERS_PATTERNS.GET_BY_EMAIL} received`,
    );

    try {
      const user = await this.userService.getUserById(data.id);

      this.logger.info(
        { rid: data.meta?.requestId, user },
        `${USERS_PATTERNS.GET_BY_EMAIL} succeeded`,
      );

      return user;
    } catch (e: any) {
      this.logger.error(
        { rid: data.meta?.requestId, err: e },
        `${USERS_PATTERNS.GET_BY_EMAIL} failed`,
      );
      throw new RpcException({ message: e?.message ?? 'Get user failed' });
    }
  }

  //TODO: Добавить пагинацию и исправить getAllUsers
  @MessagePattern(USERS_PATTERNS.GET_ALL)
  async getAll(
    @Payload()
    data: {
      meta: { requestId: string };
      includeArchived?: boolean;
      onlyArchived?: boolean;
    },
  ) {
    try {
      return await this.userService.getAllUsers({
        includeArchived: data.includeArchived,
        onlyArchived: data.onlyArchived,
      });
    } catch (e: any) {
      this.logger.error(
        { rid: data.meta?.requestId, err: e },
        `${USERS_PATTERNS.GET_ALL} failed`,
      );
      throw new RpcException({ message: e?.message ?? 'Get all users failed' });
    }
  }

  @MessagePattern(USERS_PATTERNS.GET_STATS)
  async getUserStats(
    @Payload() data: { meta: { requestId: string }; id: number },
  ) {
    try {
      return await this.userService.getUserStatsById(data.id);
    } catch (e: any) {
      this.logger.error(
        { rid: data.meta?.requestId, err: e },
        `${USERS_PATTERNS.GET_STATS} failed`,
      );
      throw new RpcException({
        message: e?.message ?? "Get user's stats failed",
      });
    }
  }

  @MessagePattern(USERS_PATTERNS.APPLY_QUIZ_STATS)
  async applyQuizStats(
    @Payload()
    data: {
      meta: { requestId: string };
      userId: number;
      patch: ApplyQuizStatsDto;
    },
  ) {}

  @MessagePattern(USERS_PATTERNS.UPDATE)
  async updateUser(
    @Payload()
    data: {
      meta: { requestId: string };
      id: number;
      dto: UpdateUserDto;
      actorId?: number;
    },
  ) {
    try {
      return await this.userService.updateUser(
        data.id,
        data.dto,
        data.actorId ?? null,
      );
    } catch (e: any) {
      this.logger.error(
        { rid: data.meta?.requestId, err: e },
        `${USERS_PATTERNS.UPDATE} failed`,
      );
      throw new RpcException({ message: e?.message ?? 'Update user failed' });
    }
  }

  @MessagePattern(USERS_PATTERNS.ARCHIVE)
  async archiveUser(
    @Payload()
    data: { meta: { requestId: string }; id: number; actorId?: number },
  ) {
    try {
      return await this.userService.archiveUser(data.id, data.actorId ?? null);
    } catch (e: any) {
      this.logger.error(
        { rid: data.meta?.requestId, err: e },
        `${USERS_PATTERNS.ARCHIVE} failed`,
      );
      throw new RpcException({ message: e?.message ?? 'Archive user failed' });
    }
  }

  @MessagePattern(USERS_PATTERNS.RESTORE)
  async restoreUser(
    @Payload()
    data: { meta: { requestId: string }; id: number; actorId?: number },
  ) {
    try {
      return await this.userService.restoreUser(data.id, data.actorId ?? null);
    } catch (e: any) {
      this.logger.error(
        { rid: data.meta?.requestId, err: e },
        `${USERS_PATTERNS.RESTORE} failed`,
      );
      throw new RpcException({ message: e?.message ?? 'Restore user failed' });
    }
  }

  @MessagePattern(USERS_PATTERNS.HISTORY)
  async history(
    @Payload() data: { meta: { requestId: string }; id: number },
  ) {
    try {
      return await this.userService.getUserHistory(data.id);
    } catch (e: any) {
      this.logger.error(
        { rid: data.meta?.requestId, err: e },
        `${USERS_PATTERNS.HISTORY} failed`,
      );
      throw new RpcException({ message: e?.message ?? 'History failed' });
    }
  }
}
// @MessagePattern('user.update') update(
//   @Payload() { id, ...dto }: { id: number } & UpdateUserDto,
// ) {
//   return this.userService.updateUser(id, dto);
// }

// @MessagePattern('user.delete') del(@Payload() dto: DeleteUserDto) {
//   return this.userService.deleteUserById(dto);
// }

// @MessagePattern('user.by_id') byId(@Payload() id: number) {
//   return this.userService.getUserById(id);
// }

// @MessagePattern('user.by_email') byEmail(@Payload() email: string) {
//   return this.userService.getUserByEmail(email);
// }
