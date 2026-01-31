import {
  BadRequestException,
  HttpException,
  HttpStatus,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { UserEntity } from './entities/user.entity';
import { CreateUserDto } from './dto/createUser.dto';
import { UpdateUserDto } from './dto/updateUser.dto';
import { DeleteUserDto } from './dto/deleteUser.dto';
import { AuthUserDto } from './dto/authUser.dto';
import { Role } from './dto/userListItem.dto';
import { UserStatsEntity } from './entities/user-stats.entity';
import { ApplyQuizStatsDto } from './dto/applyQuizStats.dto';
import {
  UserHistoryAction,
  UserHistoryEntity,
} from './entities/user-history.entity';

@Injectable()
export class UserService {
  constructor(
    @InjectRepository(UserEntity)
    private userRepository: Repository<UserEntity>,
    @InjectRepository(UserStatsEntity)
    private userStatsRepositorysitory: Repository<UserStatsEntity>,
    @InjectRepository(UserHistoryEntity)
    private userHistoryRepository: Repository<UserHistoryEntity>,
  ) {}

  private async validateNewUser(email: string) {
    const candidate = await this.getUserByEmail(email);

    if (candidate) {
      throw new HttpException(
        'Пользователь с таким email существует!',
        HttpStatus.BAD_REQUEST,
      );
    }
  }
  private async addHistory(
    userId: number,
    action: UserHistoryAction,
    actorId?: number | null,
    changes?: Record<string, unknown> | null,
  ) {
    const entry = this.userHistoryRepository.create({
      userId,
      action,
      actorId: actorId ?? null,
      changes: changes ?? null,
    });
    await this.userHistoryRepository.save(entry);
  }

  async getAllUsers(params?: {
    includeArchived?: boolean;
    onlyArchived?: boolean;
  }) {
    if (!params?.includeArchived && !params?.onlyArchived) {
      return await this.userRepository.find();
    }
    const qb = this.userRepository.createQueryBuilder('u').withDeleted();
    if (params?.onlyArchived) {
      qb.where('u.archivedAt IS NOT NULL');
    }
    return await qb.getMany();
  }

  // async getAllUsers(user: JwtPayload): Promise<UserListItemDto[]> {
  //   // Если у password стоит select:false — переключитесь на QB и .addSelect('u.password')
  //   const users = await this.userRepository.find({
  //     select: ['id', 'name', 'role', 'email'],
  //     order: { id: 'ASC' },
  //   });

  //   return users.map<UserListItemDto>((u) => {
  //     // email админам показываем ТОЛЬКО если это сам запрашивающий админ
  //     const emailForRole =
  //       u.role === Role.ADMIN && u.id !== Number(user.id) ? '' : u.email;

  //     // пароль отдаём только для role=user (и это будет хэш, если храните хэш)
  //     if (u.role === Role.USER) {
  //       return {
  //         id: u.id,
  //         name: u.name,
  //         role: u.role,
  //         email: emailForRole,
  //       };
  //     }

  //     // для остальных ролей — без пароля
  //     return {
  //       id: u.id,
  //       name: u.name,
  //       role: u.role,
  //       email: emailForRole,
  //     };
  //   });
  // }

  async deleteUserById(dto: DeleteUserDto) {
    const user = await this.userRepository.findOne({
      where: { id: dto.id },
    });
    if (!user) throw new NotFoundException('Пользователь не найден');
    await this.userRepository.softDelete(dto.id);
    await this.addHistory(dto.id, UserHistoryAction.ARCHIVED, null, {
      reason: 'soft delete',
    });
    return { message: `Пользователь с ID ${dto.id} архивирован.` };
  }

  async getUserByEmail(email: string) {
    const user = await this.userRepository.findOne({
      where: { email },
    });
    return user;
  }

  async getUserById(userId: number) {
    const user = await this.userRepository.findOne({
      where: { id: userId },
    });

    if (!user) {
      throw new HttpException(
        'Пользователь не найден!',
        HttpStatus.BAD_REQUEST,
      );
    }
    return user;
  }

  isUser(user: UserEntity) {
    return user.role === Role.USER ? user : null;
  }

  async createUser(createUserDto: CreateUserDto) {
    await this.validateNewUser(createUserDto.email);

    const entity = this.userRepository.create({
      ...createUserDto,
      counterAgentSpecializationIds:
        createUserDto.counterAgentSpecializationIds ?? [],
    });
    const user = await this.userRepository.save(entity);
    await this.addHistory(user.id, UserHistoryAction.CREATED, null, {
      email: user.email,
      role: user.role,
    });

    return user;
  }

  async authUser(authUserDto: AuthUserDto) {
    const candidate = await this.getUserByEmail(authUserDto.email);

    if (!candidate) {
      throw new HttpException(
        'Пользователь с таким email не существует',
        HttpStatus.BAD_REQUEST,
      );
    }

    //FIXME: Исправить авторизацию - костыль, правильное решение ниже
    const user = this.isUser(candidate) ?? '';
    // TODO: (АВТОРИЗАЦИЯ): убрать генерацию токена и проверку пароля - это в микросервис auth
    // (await this.authService.auth(authUserDto, candidate));
    // const tokens = await this.authService.generateToken(candidate);
    // await this.authService.saveToken(user ?? candidate, tokens.refreshToken);

    return { user };
  }

  async updateUser(
    userId: number,
    dto: UpdateUserDto,
    actorId?: number | null,
  ) {
    const user = await this.userRepository.findOne({ where: { id: userId } });
    if (!user) throw new NotFoundException('Пользователь не найден');

    const changes: Record<string, unknown> = {};

    // Разрешённые поля — имя, роль (если уже проверено), возможно email (с пересчётом emailLower)
    if (dto.name !== undefined) {
      const name = dto.name.trim();
      if (name.length < 2 || name.length > 50) {
        throw new BadRequestException('Имя: от 2 до 50 символов');
      }
      if (name !== user.name) changes.name = { from: user.name, to: name };
      user.name = name;
    }

    if (dto.email !== undefined) {
      const email = dto.email.trim();
      if (!email) throw new BadRequestException('Email пуст');
      if (email !== user.email) changes.email = { from: user.email, to: email };
      user.email = email;
    }

    if (dto.role !== undefined) {
      if (dto.role !== user.role)
        changes.role = { from: user.role, to: dto.role };
      user.role = dto.role;
    }

    if (dto.specializationId !== undefined) {
      const next = dto.specializationId ?? null;
      if (next !== user.specializationId) {
        changes.specializationId = {
          from: user.specializationId,
          to: next,
        };
      }
      user.specializationId = next;
    }

    if (dto.counterAgentSpecializationIds !== undefined) {
      user.counterAgentSpecializationIds =
        dto.counterAgentSpecializationIds ?? [];
      changes.counterAgentSpecializationIds =
        user.counterAgentSpecializationIds;
    }

    if (dto.phone !== undefined) {
      if (dto.phone !== user.phone) {
        changes.phone = { from: user.phone ?? null, to: dto.phone ?? null };
      }
      user.phone = dto.phone ?? null;
    }

    if (dto.birthDate !== undefined) {
      if (dto.birthDate !== user.birthDate) {
        changes.birthDate = {
          from: user.birthDate ?? null,
          to: dto.birthDate ?? null,
        };
      }
      user.birthDate = dto.birthDate ?? null;
    }

    if (dto.city !== undefined) {
      if (dto.city !== user.city) {
        changes.city = { from: user.city ?? null, to: dto.city ?? null };
      }
      user.city = dto.city ?? null;
    }

    if (dto.address !== undefined) {
      if (dto.address !== user.address) {
        changes.address = {
          from: user.address ?? null,
          to: dto.address ?? null,
        };
      }
      user.address = dto.address ?? null;
    }

    if (dto.photoUrl !== undefined) {
      if (dto.photoUrl !== user.photoUrl) {
        changes.photoUrl = {
          from: user.photoUrl ?? null,
          to: dto.photoUrl ?? null,
        };
      }
      user.photoUrl = dto.photoUrl ?? null;
    }

    try {
      const saved = await this.userRepository.save(user);
      if (Object.keys(changes).length > 0) {
        await this.addHistory(
          userId,
          UserHistoryAction.UPDATED,
          actorId,
          changes,
        );
      }
      return saved;
    } catch (e: any) {
      if (e?.code === '23505') {
        throw new BadRequestException('Email уже занят');
      }
      throw e;
    }
  }

  async archiveUser(userId: number, actorId?: number | null) {
    const user = await this.userRepository.findOne({ where: { id: userId } });
    if (!user) throw new NotFoundException('Пользователь не найден');
    await this.userRepository.softDelete(userId);
    await this.addHistory(userId, UserHistoryAction.ARCHIVED, actorId);
    return { id: userId };
  }

  async restoreUser(userId: number, actorId?: number | null) {
    const user = await this.userRepository.findOne({
      where: { id: userId },
      withDeleted: true,
    });
    if (!user) throw new NotFoundException('Пользователь не найден');
    await this.userRepository.restore(userId);
    await this.addHistory(userId, UserHistoryAction.RESTORED, actorId);
    return { id: userId };
  }

  async getUserHistory(userId: number) {
    return this.userHistoryRepository.find({
      where: { userId },
      order: { createdAt: 'DESC' },
    });
  }

  //FIXME: надо чтобы статистика не возращала пользователя
  async getUserStatsById(userId: number) {
    const user = await this.userRepository.findOne({
      where: { id: userId },
      relations: ['stats'],
    });

    if (!user) {
      throw new HttpException(
        'Пользователь не найден!',
        HttpStatus.BAD_REQUEST,
      );
    }
    return user.stats;
  }

  //FIXME: надо чтобы статистика не возращала пользователя
  /**
   * Обновить агрегаты статистики пользователя (upsert).
   * averageScore — число 0..100 (мы храним как numeric -> string).
   */
  async applyQuizStats(userId: number, patch: ApplyQuizStatsDto) {
    const user = await this.getUserById(userId);

    const stats = await this.getUserStatsById(user.id);

    if (!stats) {
      user.stats = this.userStatsRepositorysitory.create({});
    }

    stats.quizzesTotal = patch.quizzesTotal;
    stats.quizzesPassed = patch.quizzesPassed;
    stats.lessonsTotal = patch.lessonsTotal;
    stats.lessonsCompleted = patch.lessonsCompleted;

    const avg = Math.min(100, Math.max(0, Number(patch.averageScore) || 0));
    stats.averageScore = String(Math.round(avg * 100) / 100);

    if (patch.lastActiveAt) stats.lastActiveAt = patch.lastActiveAt;

    const stats$ = await this.userStatsRepositorysitory.save(stats);

    return {
      quizzesTotal: stats$.quizzesTotal,
      quizzesPassed: stats$.quizzesPassed,
      averageScore: Number(stats$.averageScore),
      coursesEnrolled: stats$.coursesEnrolled,
      coursesAuthored: stats$.coursesAuthored,
      lessonsTotal: stats$.lessonsTotal,
      lessonsCompleted: stats$.lessonsCompleted,
      streakDays: stats$.streakDays,
      lastActiveAt: stats$.lastActiveAt?.toISOString() ?? null,
    };
  }
  //TODO: Добавить обновление имени/email и тд
  //   // Нечего обновлять?
  //   if (
  //     updateUserDto.role === undefined &&
  //     updateUserDto.specializationId === undefined
  //   ) {
  //     throw new BadRequestException('Нет полей для обновления');
  //   }

  //   // Имя — всем можно править себя, админ — любого
  //   // if (updateUserDto.name !== undefined) {
  //   //   const name = updateUserDto.name.trim();
  //   //   if (name.length < 2 || name.length > 50) {
  //   //     throw new BadRequestException('Имя: от 2 до 50 символов');
  //   //   }
  //   //   user.name = name;
  //   // }

  //   // Роль — только админ
  //   if (updateUserDto.role !== undefined) {
  //     if (!isAdmin) {
  //       throw new ForbiddenException('Недостаточно прав для смены роли');
  //     }
  //     user.role = updateUserDto.role;
  //   }
}
