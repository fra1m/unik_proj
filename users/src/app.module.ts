import { Module } from '@nestjs/common';
import { UserModule } from './modules/user/user.module';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { envSchema } from './config/validation';
import { HealthModule } from './modules/health/health.module';
import { LoggerModule } from './common/logger/logger.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      // cache: true,
      isGlobal: true,
      validationSchema: envSchema,
      envFilePath:
        process.env.NODE_ENV === 'production' ? [] : ['.env', '../.env'],
      expandVariables: true,
    }),
    TypeOrmModule.forRootAsync({
      imports: [ConfigModule],
      inject: [ConfigService],

      // TODO: Использовать когда перейдешь на урл к бд
      // useFactory: (cfg: ConfigService) => ({
      //   type: 'postgres',
      //   url: cfg.get<string>('DATABASE_URL', { infer: true }),
      //   autoLoadEntities: true,
      //   synchronize: process.env.NODE_ENV !== 'production',
      //   maxQueryExecutionTime: 500,
      // }),

      useFactory: (cfg: ConfigService) => ({
        type: 'postgres',
        host: cfg.get<string>('POSTGRES_HOST', { infer: true }),
        port: Number(cfg.get<number>('POSTGRES_PORT', { infer: true })),
        database: cfg.get<string>('POSTGRES_DB', { infer: true }),
        username: cfg.get<string>('POSTGRES_USER', { infer: true }),
        password: cfg.get<string>('POSTGRES_PASSWORD', { infer: true }),
        autoLoadEntities: true,
        synchronize: cfg.get<string>('NODE_ENV') !== 'production',
        maxQueryExecutionTime: 500,
      }),
    }),
    LoggerModule,
    HealthModule,
    UserModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
