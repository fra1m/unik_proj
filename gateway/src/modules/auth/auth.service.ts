import { Inject, Injectable } from '@nestjs/common';
import { ClientProxy } from '@nestjs/microservices';
import { AUTH_CLIENT } from 'src/common/rmq/rmq.module';
import { rpc } from 'src/common/rpc/rpc.util';
import { Tokens } from './models/auth-model';
import { PATTERNS } from 'src/contracts/patterns';
import { UserModel } from '../users/models/user-model';

@Injectable()
export class AuthService {
  constructor(@Inject(AUTH_CLIENT) private readonly auth: ClientProxy) {}

  // async createHash(
  //   meta: { requestId: string },
  //   password: string,
  // ): Promise<string> {
  //   return await rpc<string>(this.auth, PATTERNS.AUTH_HASH, { meta, password });
  // }

  async authByPassword(
    meta: { requestId: string },
    params: { user: UserModel; password: string },
  ): Promise<Tokens> {
    return await rpc<Tokens>(this.auth, PATTERNS.AUTH_LOGIN_BY_PASSWORD, {
      meta,
      ...params,
    });
  }

  // TODO: реализовать в auth микросервисе или удалить
  async createCredentials(
    meta: { requestId: string },
    userId: number,
    password: string,
  ): Promise<{ ok: true }> {
    return await rpc<{ ok: true }>(this.auth, PATTERNS.AUTH_CREDENTIALS, {
      meta,
      userId,
      password,
    });
  }

  async generateTokens(
    meta: { requestId: string },
    user: UserModel,
    password?: string,
  ): Promise<Tokens> {
    return await rpc<Tokens>(this.auth, PATTERNS.AUTH_GENERATE_TOKENS, {
      meta,
      user,
      password,
    });
  }

  async validRefreshToken(
    meta: { requestId: string },
    refreshToken: string,
  ): Promise<{ userId: number }> {
    return await rpc<{ userId: number }>(
      this.auth,
      PATTERNS.AUTH_REFRESH_VALIDATE,
      {
        meta,
        // user,
        token: refreshToken,
      },
    );
  }

  //В auth microservice реализовать логику
  async validateAccessToken(
    meta: { requestId: string },
    accessToken: string,
  ): Promise<{ userId: number }> {
    return await rpc<{ userId: number }>(
      this.auth,
      PATTERNS.AUTH_ACCESS_VALIDATE,
      {
        meta,
        // user,
        token: accessToken,
      },
    );
  }

  async removeToken(
    meta: { requestId: string },
    refreshToken: string,
  ): Promise<boolean> {
    return await rpc<boolean>(this.auth, PATTERNS.AUTH_REMOVE_TOKEN, {
      meta,
      refreshToken,
    });
  }

  async enrollFace(
    meta: { requestId: string },
    params: { userId: number; embedding: number[] },
  ): Promise<{ userId: number; samples: number }> {
    return await rpc<{ userId: number; samples: number }>(
      this.auth,
      PATTERNS.AUTH_FACE_ENROLL,
      {
        meta,
        ...params,
      },
    );
  }

  async verifyFace(
    meta: { requestId: string },
    params: { embedding: number[]; threshold?: number },
  ): Promise<{
    matched: boolean;
    userId?: number;
    score: number;
    threshold: number;
    samples: number;
  }> {
    return await rpc<{
      matched: boolean;
      userId?: number;
      score: number;
      threshold: number;
      samples: number;
    }>(this.auth, PATTERNS.AUTH_FACE_VERIFY, {
      meta,
      ...params,
    });
  }
}
