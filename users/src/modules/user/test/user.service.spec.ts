import { Test } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import {
  BadRequestException,
  HttpException,
  HttpStatus,
  NotFoundException,
} from '@nestjs/common';

import { UserEntity } from '../entities/user.entity';
import { UserService } from '../user.service';
import { Role } from '../dto/userListItem.dto';

// Удобная фабрика под пользователя
const makeUser = (patch: Partial<UserEntity> = {}): UserEntity => {
  return {
    id: patch.id ?? 1,
    email: patch.email ?? 'user@example.com',
    name: patch.name ?? 'User',
    role: patch.role ?? Role.USER,
    // любые поля, которые есть в сущности, но нам не нужны — можно опустить
  } as UserEntity;
};

type MockRepo = {
  find: jest.Mock;
  findOne: jest.Mock;
  save: jest.Mock;
  delete: jest.Mock;
  softDelete: jest.Mock;
};

const createMockRepo = (): MockRepo => ({
  find: jest.fn(),
  findOne: jest.fn(),
  save: jest.fn(),
  delete: jest.fn(),
  softDelete: jest.fn(),
});

describe('UserService (functional with mocked repository)', () => {
  let service: UserService;
  let repo: MockRepo;

  beforeEach(async () => {
    repo = createMockRepo();

    const moduleRef = await Test.createTestingModule({
      providers: [
        UserService,
        {
          provide: getRepositoryToken(UserEntity),
          useValue: repo,
        },
      ],
    }).compile();

    service = moduleRef.get(UserService);
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  // ---------- createUser ----------
  describe('createUser', () => {
    it('должен создать пользователя (happy path)', async () => {
      const dto = { email: 'new@ex.com', name: 'Neo', role: Role.USER };
      repo.findOne.mockResolvedValueOnce(null); // validateNewUser -> getUserByEmail -> null
      repo.save.mockResolvedValueOnce(makeUser({ id: 42, ...dto }));

      const res = await service.createUser(dto);

      expect(repo.findOne).toHaveBeenCalledWith({
        where: { email: dto.email },
      });
      expect(repo.save).toHaveBeenCalledWith(expect.objectContaining(dto));
      expect(res).toEqual(makeUser({ id: 42, ...dto }));
    });

    it('должен кинуть 400, если email уже занят', async () => {
      const dto = { email: 'dup@ex.com', name: 'Dup', role: Role.USER };
      repo.findOne.mockResolvedValueOnce(makeUser({ email: dto.email })); // validateNewUser -> найден

      await expect(service.createUser(dto as any)).rejects.toEqual(
        new HttpException(
          'Пользователь с таким email существует!',
          HttpStatus.BAD_REQUEST,
        ),
      );

      expect(repo.save).not.toHaveBeenCalled();
    });
  });

  // ---------- getUserById ----------
  describe('getUserById', () => {
    it('возвращает пользователя', async () => {
      const user = makeUser({ id: 7, email: 'a@a.a' });
      repo.findOne.mockResolvedValueOnce(user);

      const res = await service.getUserById(7);
      expect(repo.findOne).toHaveBeenCalledWith({ where: { id: 7 } });
      expect(res).toBe(user);
    });

    it('404 если не найден', async () => {
      repo.findOne.mockResolvedValueOnce(null);

      await expect(service.getUserById(123)).rejects.toEqual(
        new HttpException('Пользователь не найден!', HttpStatus.BAD_REQUEST),
      );
    });
  });

  // ---------- getAllUsers ----------
  describe('getAllUsers', () => {
    it('админу скрывает email других админов (кроме самого себя)', async () => {
      const admin1 = makeUser({
        id: 1,
        email: 'admin1@ex.com',
        role: Role.ADMIN,
        name: 'A1',
      });
      const admin2 = makeUser({
        id: 2,
        email: 'admin2@ex.com',
        role: Role.ADMIN,
        name: 'A2',
      });
      const user = makeUser({
        id: 3,
        email: 'user@ex.com',
        role: Role.USER,
        name: 'U',
      });

      repo.find.mockResolvedValueOnce([admin1, admin2, user]);

      const res = await service.getAllUsers();

      // admin1 (сам запрашивает) — email должен остаться
      // admin2 (другой админ) — email должен превратиться в ''
      // user — email остаётся
      expect(res).toEqual([
        { id: 1, name: 'A1', role: Role.ADMIN, email: 'admin1@ex.com' },
        { id: 2, name: 'A2', role: Role.ADMIN, email: '' },
        { id: 3, name: 'U', role: Role.USER, email: 'user@ex.com' },
      ]);

      expect(repo.find).toHaveBeenCalledWith({
        select: ['id', 'name', 'role', 'email'],
        order: { id: 'ASC' },
      });
    });
  });

  // ---------- updateUser ----------
  describe('updateUser', () => {
    it('обновляет имя/email/роль (happy path)', async () => {
      const existing = makeUser({
        id: 10,
        email: 'old@ex.com',
        name: 'Old',
        role: Role.USER,
      });
      repo.findOne.mockResolvedValueOnce(existing);

      const dto = { name: 'New Name', email: 'new@ex.com', role: Role.ADMIN };
      const saved = makeUser({
        id: 10,
        email: dto.email,
        name: dto.name,
        role: dto.role,
      });
      repo.save.mockResolvedValueOnce(saved);

      const res = await service.updateUser(10, dto as any);

      expect(repo.findOne).toHaveBeenCalledWith({ where: { id: 10 } });
      expect(repo.save).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 10,
          email: 'new@ex.com',
          name: 'New Name',
          role: Role.ADMIN,
        }),
      );
      expect(res).toEqual({
        id: 10,
        email: 'new@ex.com',
        name: 'New Name',
        role: Role.ADMIN,
      });
    });

    it('кидает 400, если имя слишком короткое', async () => {
      const existing = makeUser({ id: 11 });
      repo.findOne.mockResolvedValueOnce(existing);

      await expect(
        service.updateUser(11, { name: 'A' } as any),
      ).rejects.toEqual(new BadRequestException('Имя: от 2 до 50 символов'));

      expect(repo.save).not.toHaveBeenCalled();
    });

    it('пробрасывает 400 при конфликте email (23505)', async () => {
      const existing = makeUser({ id: 12 });
      repo.findOne.mockResolvedValueOnce(existing);

      const err: any = new Error('unique_violation');
      err.code = '23505';
      repo.save.mockRejectedValueOnce(err);

      await expect(
        service.updateUser(12, { email: 'dup@ex.com' } as any),
      ).rejects.toEqual(new BadRequestException('Email уже занят'));
    });

    it('404 если пользователя нет', async () => {
      repo.findOne.mockResolvedValueOnce(null);
      await expect(
        service.updateUser(999, { name: 'X' } as any),
      ).rejects.toEqual(new NotFoundException('Пользователь не найден'));
    });
  });

  // ---------- deleteUserById ----------
  describe('deleteUserById', () => {
    it('404 если пользователя нет', async () => {
      repo.findOne.mockResolvedValueOnce(null);

      await expect(service.deleteUserById({ id: 777 } as any)).rejects.toEqual(
        new NotFoundException('Пользователь не найден'),
      );
    });

    it('возвращает message при «успешном удалении» (по текущей реализации)', async () => {
      repo.findOne.mockResolvedValueOnce(makeUser({ id: 5 }));
      // метод сейчас не вызывает ни delete, ни softDelete — просто возвращает message
      const res = await service.deleteUserById({ id: 5 } as any);
      expect(res).toEqual({ message: 'Пользователь с ID 5 удалён.' });
      expect(repo.delete).not.toHaveBeenCalled();
      expect(repo.softDelete).not.toHaveBeenCalled();
    });
  });

  // ---------- authUser ----------
  describe('authUser', () => {
    it('400 если пользователь не найден по email', async () => {
      repo.findOne.mockResolvedValueOnce(null);
      await expect(
        service.authUser({ email: 'none@ex.com' } as any),
      ).rejects.toEqual(
        new HttpException(
          'Пользователь с таким email не существует',
          HttpStatus.BAD_REQUEST,
        ),
      );
    });

    it('возвращает { user } если роль USER (текущая логика с isUser)', async () => {
      const u = makeUser({ id: 1, role: Role.USER });
      repo.findOne.mockResolvedValueOnce(u);

      const res = await service.authUser({ email: u.email } as any);
      expect(res).toEqual({ user: u });
    });

    it('возвращает { user: "" } если роль не USER (текущая логика)', async () => {
      const u = makeUser({ id: 2, role: Role.ADMIN });
      repo.findOne.mockResolvedValueOnce(u);

      const res = await service.authUser({ email: u.email } as any);
      expect(res).toEqual({ user: '' });
    });
  });
});
