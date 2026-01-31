//TODO: Проверить название патернов. Провить их логику в микросервисе users

import { Inject, Injectable } from '@nestjs/common';
import { CreateUserDto } from './dto/create-user.dto';
import { ClientProxy } from '@nestjs/microservices';
import { USERS_CLIENT } from 'src/common/rmq/rmq.module';
import { rpc } from 'src/common/rpc/rpc.util';
import { UserModel } from './models/user-model';
import { PATTERNS } from 'src/contracts/patterns';
import { AuthUserDto } from './dto/authUser.dto';
import { UserStatsModel } from './models/user-stats-model';
import { ApplyQuizStatsDto } from './dto/apply-quiz-stats.dto';
import { UpdateUserDto } from './dto/update-user.dto';

@Injectable()
export class UsersService {
  constructor(@Inject(USERS_CLIENT) private readonly users: ClientProxy) {}

  private withSub<T extends { id: number }>(user: T): T & { sub: number } {
    return { ...user, sub: user.id };
  }

  private withSubList<T extends { id: number }>(
    users: T[],
  ): Array<T & { sub: number }> {
    return users.map((user) => this.withSub(user));
  }

  async getByEmail(
    meta: { requestId: string },
    authUserDto: Omit<AuthUserDto, 'password'>,
  ): Promise<UserModel> {
    const exists = await rpc<UserModel & { id: number }>(
      this.users,
      PATTERNS.USERS_BY_EMAIL,
      {
        meta,
        authUserDto,
      },
    );

    return this.withSub(exists);
  }

  async createUser(
    meta: { requestId: string },
    createUserDto: Omit<CreateUserDto, 'password'>,
  ): Promise<UserModel> {
    const exists = await rpc<UserModel & { id: number }>(
      this.users,
      PATTERNS.USERS_CREATE,
      {
        meta,
        createUserDto,
      },
    );

    return this.withSub(exists);
  }

  async getAllUsers(): Promise<UserModel[]> {
    const users = await rpc<Array<UserModel & { id: number }>>(
      this.users,
      PATTERNS.USERS_ALL,
      {},
    );
    return this.withSubList(users);
  }

  async getAllUsersWithFilters(params?: {
    includeArchived?: boolean;
    onlyArchived?: boolean;
  }): Promise<UserModel[]> {
    const users = await rpc<Array<UserModel & { id: number }>>(
      this.users,
      PATTERNS.USERS_ALL,
      {
        includeArchived: params?.includeArchived,
        onlyArchived: params?.onlyArchived,
      },
    );
    return this.withSubList(users);
  }

  async getUserById(
    meta: { requestId: string },
    id: number,
  ): Promise<UserModel> {
    const user = await rpc<UserModel & { id: number }>(
      this.users,
      PATTERNS.USERS_BY_ID,
      {
        meta,
        id,
      },
    );
    return this.withSub(user);
  }

  async getUserStatsById(
    meta: { requestId: string },
    id: number,
  ): Promise<UserStatsModel> {
    const stats = await rpc<UserStatsModel>(
      this.users,
      PATTERNS.USERS_GET_STATS,
      {
        meta,
        id,
      },
    );
    return stats;
  }

  async applyQuizStats(
    meta: { requestId: string },
    userId: number,
    patch: ApplyQuizStatsDto,
  ): Promise<UserStatsModel> {
    const stats = await rpc<UserStatsModel>(
      this.users,
      PATTERNS.USERS_APPLY_QUIZ_STATS,
      {
        meta,
        userId,
        patch,
      },
    );
    return stats;
  }

  async updateUser(
    meta: { requestId: string },
    id: number,
    dto: UpdateUserDto,
    actorId?: number,
  ): Promise<UserModel> {
    const updated = await rpc<UserModel & { id: number }>(
      this.users,
      PATTERNS.USERS_UPDATE,
      { meta, id, dto, actorId },
    );
    return this.withSub(updated);
  }

  async archiveUser(
    meta: { requestId: string },
    id: number,
    actorId?: number,
  ): Promise<{ id: number }> {
    return await rpc<{ id: number }>(this.users, PATTERNS.USERS_ARCHIVE, {
      meta,
      id,
      actorId,
    });
  }

  async restoreUser(
    meta: { requestId: string },
    id: number,
    actorId?: number,
  ): Promise<{ id: number }> {
    return await rpc<{ id: number }>(this.users, PATTERNS.USERS_RESTORE, {
      meta,
      id,
      actorId,
    });
  }

  async getUserHistory(
    meta: { requestId: string },
    id: number,
  ): Promise<any[]> {
    return await rpc<any[]>(this.users, PATTERNS.USERS_HISTORY, { meta, id });
  }
}
