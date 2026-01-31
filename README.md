# FaceID Auth Platform (C++ + NestJS + React)

Полный стек авторизации по лицу: C++ вычисляет эмбеддинги, backend хранит шаблоны, frontend управляет сценарием регистрации и входа.

## Состав проекта

- **C++ ядро**: `src/` — FaceID приложение и CLI для получения эмбеддинга
- **faceid-core**: `faceid-core/` — HTTP сервис, который запускает C++ CLI
- **gateway**: `gateway/` — API шлюз (REST), принимает запросы от клиента
- **auth**: `auth/` — хранит face-шаблоны и выполняет проверку
- **users**: `users/` — хранит пользователей
- **client**: `client/` — React UI
- **docker-compose**: `docker-compose/` — запуск всех сервисов

## Диаграмма архитектуры

```mermaid
flowchart LR
  subgraph Client
    C[React UI]
  end

  subgraph Gateway
    G[API Gateway]
  end

  subgraph FaceCore
    F[faceid-core]
    CLI[FaceID CLI (C++)]
  end

  subgraph AuthService
    A[Auth Service]
    ADB[(Postgres auth-db)]
  end

  subgraph UsersService
    U[Users Service]
    UDB[(Postgres users-db)]
  end

  R[(RabbitMQ)]

  C -->|HTTP| G
  G -->|HTTP /auth/face/capture| F
  F -->|exec| CLI
  G -->|RPC via RMQ| A
  G -->|RPC via RMQ| U
  A --> ADB
  U --> UDB
  G <--> R
```

## Поток регистрации (FaceID)

```mermaid
sequenceDiagram
  participant UI as React UI
  participant GW as Gateway
  participant FC as faceid-core
  participant CLI as FaceID CLI (C++)
  participant AUTH as Auth Service
  participant USERS as Users Service

  UI->>GW: POST /auth/face/capture (imageBase64)
  GW->>FC: /embedding/from-image
  FC->>CLI: FaceIDCli <image>
  CLI-->>FC: { embedding }
  FC-->>GW: { embedding }
  UI->>GW: POST /auth/face/register-with-embeddings
  GW->>USERS: createUser
  GW->>AUTH: faceEnroll (embedding)
  AUTH-->>GW: samples
  GW-->>UI: user + tokens
```

## Поток входа (FaceID)

```mermaid
sequenceDiagram
  participant UI as React UI
  participant GW as Gateway
  participant FC as faceid-core
  participant CLI as FaceID CLI (C++)
  participant AUTH as Auth Service
  participant USERS as Users Service

  UI->>GW: POST /auth/face/capture (imageBase64)
  GW->>FC: /embedding/from-image
  FC->>CLI: FaceIDCli <image>
  CLI-->>FC: { embedding }
  FC-->>GW: { embedding }
  UI->>GW: POST /auth/face/login-with-embeddings
  GW->>AUTH: faceVerify (embedding)
  AUTH-->>GW: matched + userId
  GW->>USERS: getUserById
  GW-->>UI: user + tokens
```

## Автоматический сбор эмбеддингов на фронте

```mermaid
sequenceDiagram
  participant UI as React UI
  participant GW as Gateway
  participant FC as faceid-core
  participant CLI as FaceID CLI (C++)

  loop каждые N мс
    UI->>GW: /auth/face/capture (imageBase64)
    GW->>FC: /embedding/from-image
    FC->>CLI: FaceIDCli <image>
    CLI-->>FC: { embedding }
    FC-->>GW: { embedding }
    GW-->>UI: embedding
  end

  UI->>GW: /auth/face/register-with-embeddings|login-with-embeddings
```

## Компоненты C++

- `FaceID` — GUI/камера (используется локально, не в docker)
- `FaceIDCli` — CLI, который принимает путь до изображения и возвращает JSON с эмбеддингом

Пример вывода CLI:

```json
{
  "embedding": [0.0123, 0.0456, ...]
}
```

## Docker Compose

Основной запуск: `docker-compose/docker-compose.yml`

Сервисы:
- `rabbitmq`
- `faceid-core`
- `users-db`
- `auth-db`
- `users`
- `auth`
- `gateway`
- `client`

Запуск:

```bash
docker compose -f docker-compose/docker-compose.yml up --build
```

## Основные эндпоинты

- `POST /auth/face/capture`
- `POST /auth/face/register-with-embeddings`
- `POST /auth/face/login-with-embeddings`

## Где меняется логика

- C++ CLI: `src/cli.cpp`
- Face core (HTTP): `faceid-core/server.js`
- Gateway FaceID API: `gateway/src/modules/auth/face-auth.controller.ts`
- Auth сервис: `auth/src/modules/auth/auth.service.ts`
- UI: `client/src/App.tsx`

## Примечания

- В docker контейнере **нет GUI**, поэтому используется только `FaceIDCli`.
- В случае ошибки `413 Payload Too Large` фронт автоматически снижает качество кадра.
- Точность определяется порогом `FACE_MATCH_THRESHOLD` (см. `docker-compose.yml`).
