import {
  CanActivate,
  ExecutionContext,
  HttpException,
  HttpStatus,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { Request } from 'express'; // Добавь вверху
import { Reflector } from '@nestjs/core';
import { JwtService } from '@nestjs/jwt';

import { ROLES_KEY } from 'src/common/decorators/roles-auth.decorator';
import { InjectPinoLogger, PinoLogger } from 'nestjs-pino';
import { UserModel } from 'src/modules/users/models/user-model';

@Injectable()
export class RolesGuard implements CanActivate {
  constructor(
    private readonly jwtService: JwtService,
    private readonly reflector: Reflector,
    @InjectPinoLogger('Role Guard') private readonly logger: PinoLogger,
  ) {}

  canActivate(context: ExecutionContext): boolean {
    const requiredRoles = this.reflector.getAllAndOverride<string[]>(
      ROLES_KEY,
      [context.getHandler(), context.getClass()],
    );

    if (!requiredRoles || requiredRoles.length === 0) {
      return true;
    }

    const req = context.switchToHttp().getRequest<Request>() as Request & {
      user?: UserModel;
    };
    const authHeader = req.headers.authorization;

    if (!authHeader) {
      throw new UnauthorizedException('Нет заголовка авторизации');
    }

    const [bearer, token] = authHeader.split(' ');

    if (bearer !== 'Bearer' || !token) {
      throw new UnauthorizedException('Вам необходимо авторизоваться');
    }

    try {
      const user = this.jwtService.verify<UserModel>(token);
      req.user = user;

      const userRoles = Array.isArray(user.role) ? user.role : [user.role];

      const hasRole = userRoles.some((role: string) =>
        requiredRoles.includes(role),
      );

      if (!hasRole) {
        throw new HttpException('Недостаточно прав', HttpStatus.FORBIDDEN);
      }

      return true;
    } catch {
      // if (e.name === 'TokenExpiredError') {
      //   throw new UnauthorizedException('Токен истёк');
      // }
      this.logger.error('UNATARIZED');
      throw new HttpException('Доступ запрещён', HttpStatus.FORBIDDEN);
    }
  }
}
