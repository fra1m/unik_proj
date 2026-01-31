import { Module, DynamicModule } from '@nestjs/common';
import { ClientsModule, Transport, RmqOptions } from '@nestjs/microservices';
import { ConfigModule, ConfigService } from '@nestjs/config';

export const AUTH_CLIENT = 'AUTH_CLIENT';
export const USERS_CLIENT = 'USERS_CLIENT';
export const WS_CLIENT = 'WS_CLIENT';
export const COURSES_CLIENT = 'COURSES_CLIENT';
export const SPECIALIZATIONS_CLIENT = 'SPECIALIZATIONS_CLIENT';
export const ANALYTICS_CLIENT = 'ANALYTICS_CLIENT';
export const LESSONS_CLIENT = 'LESSONS_CLIENT';
export const QUIZZES_CLIENT = 'QUIZZES_CLIENT';
export const VIDEOS_CLIENT = 'VIDEOS_CLIENT';

export function buildRmqOptions(
  cfg: ConfigService,
  queueEnv: string,
  defQueue: string,
): RmqOptions {
  const url = cfg.getOrThrow<string>('RABBITMQ_URL');
  const queue = cfg.get<string>(queueEnv) ?? defQueue;
  const dlx = cfg.get<string>('RMQ_DLX') ?? 'dlx';
  const ttl = Number(cfg.get<string>('RMQ_MESSAGE_TTL_MS'));
  const maxLen = Number(cfg.get<string>('RMQ_MAX_LENGTH'));

  return {
    transport: Transport.RMQ,
    options: {
      urls: [url],
      queue,
      queueOptions: {
        durable: true, // очередь переживёт рестарт брокера (не volatile)
        // Аргументы уровня очереди (amqplib-поддержка)
        arguments: {
          'x-dead-letter-exchange': dlx, // куда падать «плохим» сообщениям
          ...(Number.isFinite(ttl) ? { 'x-message-ttl': ttl } : {}), // TTL сообщений в очереди
          ...(Number.isFinite(maxLen) ? { 'x-max-length': maxLen } : {}), // верхняя граница по количеству сообщений
        },
      },
      persistent: true,
    },
  };
}

function registerRmqClient(token: string, queueEnv: string, defQueue: string) {
  return ClientsModule.registerAsync([
    {
      name: token,
      imports: [ConfigModule],
      inject: [ConfigService],
      useFactory: (cfg: ConfigService): RmqOptions =>
        buildRmqOptions(cfg, queueEnv, defQueue),
    },
  ]);
}

@Module({})
export class RmqModule {
  // Для корневого модуля (если хочется подключить оба сразу)
  static forServices(): DynamicModule {
    const auth = registerRmqClient(AUTH_CLIENT, 'RMQ_AUTH_QUEUE', 'auth');
    const users = registerRmqClient(USERS_CLIENT, 'RMQ_USERS_QUEUE', 'users');
    const ws = registerRmqClient(WS_CLIENT, 'RMQ_WS_QUEUE', 'ws');
    const courses = registerRmqClient(
      COURSES_CLIENT,
      'RMQ_COURSES_QUEUE',
      'courses',
    );
    const specializations = registerRmqClient(
      SPECIALIZATIONS_CLIENT,
      'RMQ_SPECIALIZATIONS_QUEUE',
      'specializations',
    );
    const analytics = registerRmqClient(
      ANALYTICS_CLIENT,
      'RMQ_ANALYTICS_QUEUE',
      'analytics',
    );

    const lessons = registerRmqClient(
      LESSONS_CLIENT,
      'RMQ_LESSONS_QUEUE',
      'lessons',
    );
    const quizzes = registerRmqClient(
      QUIZZES_CLIENT,
      'RMQ_QUIZZES_QUEUE',
      'quizzes',
    );
    const videos = registerRmqClient(
      VIDEOS_CLIENT,
      'RMQ_VIDEOS_QUEUE',
      'videos',
    );

    return {
      module: RmqModule,
      imports: [
        ConfigModule,
        auth,
        users,
        ws,
        courses,
        specializations,
        analytics,
        lessons,
        quizzes,
        videos,
      ],
      exports: [
        auth,
        users,
        ws,
        courses,
        specializations,
        analytics,
        lessons,
        quizzes,
        videos,
      ],
    };
  }

  // Точечные фабрики — использовать внутри feature-модулей
  static forAuth(): DynamicModule {
    const auth = registerRmqClient(AUTH_CLIENT, 'RMQ_AUTH_QUEUE', 'auth');
    return {
      module: RmqModule,
      imports: [ConfigModule, auth],
      exports: [auth],
    };
  }

  static forUsers(): DynamicModule {
    const users = registerRmqClient(USERS_CLIENT, 'RMQ_USERS_QUEUE', 'users');
    return {
      module: RmqModule,
      imports: [ConfigModule, users],
      exports: [users],
    };
  }

  static forWs(): DynamicModule {
    const ws = registerRmqClient(WS_CLIENT, 'RMQ_WS_QUEUE', 'ws');
    return {
      module: RmqModule,
      imports: [ConfigModule, ws],
      exports: [ws],
    };
  }

  static forCourses(): DynamicModule {
    const courses = registerRmqClient(
      COURSES_CLIENT,
      'RMQ_COURSES_QUEUE',
      'courses',
    );
    return {
      module: RmqModule,
      imports: [ConfigModule, courses],
      exports: [courses],
    };
  }

  static forSpecialization(): DynamicModule {
    const specializations = registerRmqClient(
      SPECIALIZATIONS_CLIENT,
      'RMQ_SPECIALIZATION_QUEUE',
      'specializations',
    );
    return {
      module: RmqModule,
      imports: [ConfigModule, specializations],
      exports: [specializations],
    };
  }

  static forAnalytics(): DynamicModule {
    const analytics = registerRmqClient(
      ANALYTICS_CLIENT,
      'RMQ_ANALYTICS_QUEUE',
      'analytics',
    );
    return {
      module: RmqModule,
      imports: [ConfigModule, analytics],
      exports: [analytics],
    };
  }

  static forLessons(): DynamicModule {
    const lessons = registerRmqClient(
      LESSONS_CLIENT,
      'RMQ_LESSONS_QUEUE',
      'lessons',
    );
    return {
      module: RmqModule,
      imports: [ConfigModule, lessons],
      exports: [lessons],
    };
  }

  static forQuizzes(): DynamicModule {
    const quizzes = registerRmqClient(
      QUIZZES_CLIENT,
      'RMQ_QUIZZES_QUEUE',
      'quizzes',
    );
    return {
      module: RmqModule,
      imports: [ConfigModule, quizzes],
      exports: [quizzes],
    };
  }

  static forVideos(): DynamicModule {
    const videos = registerRmqClient(
      VIDEOS_CLIENT,
      'RMQ_VIDEOS_QUEUE',
      'videos',
    );
    return {
      module: RmqModule,
      imports: [ConfigModule, videos],
      exports: [videos],
    };
  }
}
