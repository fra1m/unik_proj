import {
  BadRequestException,
  Body,
  Controller,
  Get,
  Post,
  Res,
  UnauthorizedException,
} from '@nestjs/common';
import { ApiOperation, ApiTags } from '@nestjs/swagger';
import { Response } from 'express';
import { ReqId } from 'src/common/http/req-id.decorator';
import { Role } from 'src/common/decorators/roles-auth.decorator';
import { UsersService } from '../users/users.service';
import { UserModel } from '../users/models/user-model';
import { AuthService } from './auth.service';
import { Tokens } from './models/auth-model';
import { FaceCoreService } from './face-core.service';

class FaceRegisterDto {
  email!: string;
  name!: string;
  embedding!: number[];
  role?: Role;
  specializationId?: number;
}

class FaceLoginDto {
  embedding!: number[];
  email?: string;
  threshold?: number;
}

class FaceEnrollDto {
  userId!: number;
  embedding!: number[];
}

class FaceVerifyDto {
  embedding!: number[];
  threshold?: number;
}

class FaceCameraDto {
  imageBase64!: string;
  email?: string;
  name?: string;
  threshold?: number;
}

class FaceCaptureDto {
  imageBase64!: string;
}

class FaceEmbeddingsRegisterDto {
  email!: string;
  name!: string;
  embeddings!: number[][];
  role?: Role;
  specializationId?: number;
}

class FaceEmbeddingsLoginDto {
  email?: string;
  embeddings!: number[][];
  threshold?: number;
}

@ApiTags('face-auth')
@Controller('auth/face')
export class FaceAuthController {
  constructor(
    private readonly authService: AuthService,
    private readonly usersService: UsersService,
    private readonly faceCoreService: FaceCoreService,
  ) {}

  private async resolveEmbedding(
    embedding?: number[],
    useCore?: boolean,
  ): Promise<number[]> {
    if (Array.isArray(embedding) && embedding.length >= 8) {
      return embedding;
    }
    if (useCore) {
      return this.faceCoreService.getLatestEmbedding();
    }
    throw new BadRequestException('Embedding is required');
  }

  private resolveEmbeddings(embeddings: number[][]): number[][] {
    if (!Array.isArray(embeddings) || embeddings.length === 0) {
      throw new BadRequestException('Embeddings are required');
    }
    const cleaned = embeddings
      .filter((item) => Array.isArray(item))
      .map((item) => item.map((value) => Number(value)).filter(Number.isFinite))
      .filter((item) => item.length >= 8);
    if (cleaned.length === 0) {
      throw new BadRequestException('No valid embeddings provided');
    }
    return cleaned;
  }

  private setRefreshCookie(res: Response, refreshToken: string) {
    const isProd = process.env.NODE_ENV === 'production';
    res.cookie('refreshToken', refreshToken, {
      httpOnly: true,
      secure: isProd,
      sameSite: isProd ? 'none' : 'lax',
      path: '/',
      maxAge: 30 * 24 * 60 * 60 * 1000,
    });
  }

  @ApiOperation({ summary: 'Статус FaceID приложения' })
  @Get('core/status')
  async coreStatus() {
    return this.faceCoreService.getAppStatus();
  }

  @ApiOperation({ summary: 'Запустить FaceID приложение' })
  @Post('core/start')
  async coreStart() {
    return this.faceCoreService.startApp();
  }

  @ApiOperation({ summary: 'Остановить FaceID приложение' })
  @Post('core/stop')
  async coreStop() {
    return this.faceCoreService.stopApp();
  }

  @ApiOperation({ summary: 'Последний эмбеддинг (метаданные)' })
  @Get('core/latest')
  async coreLatest() {
    return this.faceCoreService.getLatestEmbeddingInfo();
  }

  @ApiOperation({ summary: 'Регистрация по FaceID' })
  @Post('register')
  async register(
    @Body() dto: FaceRegisterDto,
    @ReqId() reqId: string,
    @Res({ passthrough: true }) res: Response,
  ): Promise<{ user: UserModel; tokens: Tokens; face: { samples: number } }> {
    const meta = { requestId: reqId };
    const role = dto.role ?? Role.USER;
    const specializationId = Number.isFinite(dto.specializationId)
      ? Number(dto.specializationId)
      : 0;
    const embedding = await this.resolveEmbedding(dto.embedding, false);

    const created = await this.usersService.createUser(meta, {
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

    const face = await this.authService.enrollFace(meta, {
      userId: created.sub,
      embedding,
    });

    const tokens = await this.authService.generateTokens(meta, created);
    this.setRefreshCookie(res, tokens.refreshToken);

    return { user: created, tokens, face };
  }

  @ApiOperation({ summary: 'Логин по FaceID' })
  @Post('login')
  async login(
    @Body() dto: FaceLoginDto,
    @ReqId() reqId: string,
    @Res({ passthrough: true }) res: Response,
  ): Promise<{
    user: UserModel;
    tokens: Tokens;
    verify: { matched: boolean; score: number; threshold: number };
  }> {
    const meta = { requestId: reqId };
    const embedding = await this.resolveEmbedding(dto.embedding, false);
    const verify = await this.authService.verifyFace(meta, {
      embedding,
      threshold: dto.threshold,
    });

    if (!verify.matched || !verify.userId) {
      throw new UnauthorizedException('Face not recognized');
    }

    const user = await this.usersService.getUserById(meta, verify.userId);
    if (dto.email && user.email !== dto.email.trim().toLowerCase()) {
      throw new UnauthorizedException('Face does not match this email');
    }

    const tokens = await this.authService.generateTokens(meta, user);
    this.setRefreshCookie(res, tokens.refreshToken);

    return {
      user,
      tokens,
      verify: {
        matched: verify.matched,
        score: verify.score,
        threshold: verify.threshold,
      },
    };
  }

  @ApiOperation({ summary: 'Добавить face-эмбеддинг пользователю' })
  @Post('enroll')
  async enroll(@Body() dto: FaceEnrollDto, @ReqId() reqId: string) {
    const meta = { requestId: reqId };
    const embedding = await this.resolveEmbedding(dto.embedding, false);
    return this.authService.enrollFace(meta, {
      userId: dto.userId,
      embedding,
    });
  }

  @ApiOperation({ summary: 'Проверить face-эмбеддинг' })
  @Post('verify')
  async verify(@Body() dto: FaceVerifyDto, @ReqId() reqId: string) {
    const meta = { requestId: reqId };
    const embedding = await this.resolveEmbedding(dto.embedding, false);
    const verify = await this.authService.verifyFace(meta, {
      embedding,
      threshold: dto.threshold,
    });
    if (verify.userId) {
      const user = await this.usersService.getUserById(meta, verify.userId);
      return { ...verify, user };
    }
    return verify;
  }

  @ApiOperation({ summary: 'Регистрация по FaceID (через faceid-core)' })
  @Post('register-from-core')
  async registerFromCore(
    @Body() dto: Omit<FaceRegisterDto, 'embedding'>,
    @ReqId() reqId: string,
    @Res({ passthrough: true }) res: Response,
  ) {
    const embedding = await this.faceCoreService.getLatestEmbedding();
    return this.register({ ...dto, embedding }, reqId, res);
  }

  @ApiOperation({ summary: 'Логин по FaceID (через faceid-core)' })
  @Post('login-from-core')
  async loginFromCore(
    @Body() dto: Omit<FaceLoginDto, 'embedding'>,
    @ReqId() reqId: string,
    @Res({ passthrough: true }) res: Response,
  ) {
    const embedding = await this.faceCoreService.getLatestEmbedding();
    return this.login({ ...dto, embedding }, reqId, res);
  }

  @ApiOperation({ summary: 'Регистрация по FaceID (камера → core → C++)' })
  @Post('register-from-camera')
  async registerFromCamera(
    @Body() dto: FaceCameraDto,
    @ReqId() reqId: string,
    @Res({ passthrough: true }) res: Response,
  ) {
    if (!dto.email || !dto.name) {
      throw new BadRequestException('email and name are required');
    }
    const { embedding } = await this.faceCoreService.getEmbeddingFromImage(
      dto.imageBase64,
    );
    return this.register(
      {
        email: dto.email,
        name: dto.name,
        embedding,
      },
      reqId,
      res,
    );
  }

  @ApiOperation({ summary: 'Логин по FaceID (камера → core → C++)' })
  @Post('login-from-camera')
  async loginFromCamera(
    @Body() dto: FaceCameraDto,
    @ReqId() reqId: string,
    @Res({ passthrough: true }) res: Response,
  ) {
    const { embedding } = await this.faceCoreService.getEmbeddingFromImage(
      dto.imageBase64,
    );
    return this.login(
      {
        email: dto.email,
        embedding,
        threshold: dto.threshold,
      },
      reqId,
      res,
    );
  }

  @ApiOperation({ summary: 'Снять эмбеддинг из кадра (камера → core → C++)' })
  @Post('capture')
  async capture(@Body() dto: FaceCaptureDto) {
    const { embedding, pose } = await this.faceCoreService.getEmbeddingFromImage(
      dto.imageBase64,
    );
    return { embedding, length: embedding.length, pose };
  }

  @ApiOperation({ summary: 'Регистрация по набору эмбеддингов' })
  @Post('register-with-embeddings')
  async registerWithEmbeddings(
    @Body() dto: FaceEmbeddingsRegisterDto,
    @ReqId() reqId: string,
    @Res({ passthrough: true }) res: Response,
  ) {
    const meta = { requestId: reqId };
    const role = dto.role ?? Role.USER;
    const specializationId = Number.isFinite(dto.specializationId)
      ? Number(dto.specializationId)
      : 0;
    const embeddings = this.resolveEmbeddings(dto.embeddings);

    const created = await this.usersService.createUser(meta, {
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

    let samples = 0;
    for (const embedding of embeddings) {
      const face = await this.authService.enrollFace(meta, {
        userId: created.sub,
        embedding,
      });
      samples = face.samples;
    }

    const tokens = await this.authService.generateTokens(meta, created);
    this.setRefreshCookie(res, tokens.refreshToken);

    return { user: created, tokens, face: { samples } };
  }

  @ApiOperation({ summary: 'Логин по набору эмбеддингов' })
  @Post('login-with-embeddings')
  async loginWithEmbeddings(
    @Body() dto: FaceEmbeddingsLoginDto,
    @ReqId() reqId: string,
    @Res({ passthrough: true }) res: Response,
  ) {
    const meta = { requestId: reqId };
    const embeddings = this.resolveEmbeddings(dto.embeddings);

    let best:
      | {
          matched: boolean;
          userId?: number;
          score: number;
          threshold: number;
          samples: number;
        }
      | null = null;

    for (const embedding of embeddings) {
      const verify = await this.authService.verifyFace(meta, {
        embedding,
        threshold: dto.threshold,
      });
      if (!best || verify.score > best.score) {
        best = verify;
      }
    }

    if (!best || !best.userId || !best.matched) {
      throw new UnauthorizedException('Face not recognized');
    }

    const user = await this.usersService.getUserById(meta, best.userId);
    if (dto.email && user.email !== dto.email.trim().toLowerCase()) {
      throw new UnauthorizedException('Face does not match this email');
    }

    const tokens = await this.authService.generateTokens(meta, user);
    this.setRefreshCookie(res, tokens.refreshToken);

    return {
      user,
      tokens,
      verify: {
        matched: best.matched,
        score: best.score,
        threshold: best.threshold,
      },
    };
  }
}
