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

## Архитектура (контуры)

```mermaid
flowchart LR
  subgraph Client["Клиент"]
    CAM[WebCam]
    UI[React UI]
  end

  subgraph Gateway["Gateway (NestJS)"]
    GW["REST /auth/face/*"]
  end

  subgraph Core["FaceID Core (Node)"]
    FC[faceid-core]
    CLI["FaceIDCli (C++)"]
  end

  subgraph Services["Сервисы"]
    AUTH["Auth Service"]
    USERS["Users Service"]
  end

  subgraph Data["Данные и шина"]
    R[(RabbitMQ)]
    ADB[(Postgres auth-db)]
    UDB[(Postgres users-db)]
  end

  CAM --> UI
  UI -->|HTTP JSON| GW
  GW -->|/embedding/from-image| FC
  FC -->|spawn process| CLI
  CLI -->|embedding JSON| FC
  GW -->|RPC via RabbitMQ| R
  R --> AUTH
  R --> USERS
  AUTH --> ADB
  USERS --> UDB
```

## Поток регистрации

```mermaid
sequenceDiagram
  autonumber
  participant UI as React UI
  participant GW as Gateway
  participant FC as faceid-core
  participant CLI as FaceIDCli (C++)
  participant USERS as Users
  participant AUTH as Auth

  Note over UI: Авто-сканирование (N кадров)
  loop N раз
    UI->>GW: POST /auth/face/capture (imageBase64)
    GW->>FC: POST /embedding/from-image
    FC->>CLI: exec FaceIDCli (image)
    CLI-->>FC: embedding
    FC-->>GW: embedding
    GW-->>UI: embedding
  end

  UI->>GW: POST /auth/face/register-with-embeddings
  GW->>USERS: createUser
  GW->>AUTH: faceEnroll (embedding[])
  AUTH-->>GW: ok
  GW-->>UI: user + tokens
```

## Поток входа

```mermaid
sequenceDiagram
  autonumber
  participant UI as React UI
  participant GW as Gateway
  participant FC as faceid-core
  participant CLI as FaceIDCli (C++)
  participant AUTH as Auth
  participant USERS as Users

  Note over UI: Авто-сканирование (N кадров)
  loop N раз
    UI->>GW: POST /auth/face/capture (imageBase64)
    GW->>FC: POST /embedding/from-image
    FC->>CLI: exec FaceIDCli (image)
    CLI-->>FC: embedding
    FC-->>GW: embedding
    GW-->>UI: embedding
  end

  UI->>GW: POST /auth/face/login-with-embeddings
  GW->>AUTH: faceVerify (embedding[])
  alt Совпадение
    AUTH-->>GW: matched + userId
    GW->>USERS: getUserById
    GW-->>UI: user + tokens
  else Нет совпадения
    AUTH-->>GW: not matched
    GW-->>UI: ошибка
  end
```

## Цикл «capture» (клиент)

```mermaid
flowchart TD
  S[Старт сканирования] --> READY{Камера готова?}
  READY -- нет --> WAIT[Ожидание камеры]
  WAIT --> READY
  READY -- да --> T[Таймер каждые N мс]
  T --> SHOT[Снимок кадра]
  SHOT --> SEND[POST /auth/face/capture]
  SEND --> RESP{Ответ OK?}
  RESP -- 413 --> Q[Снизить качество кадра]
  Q --> T
  RESP -- другая ошибка --> ERR[Показать ошибку]
  ERR --> T
  RESP -- успех --> ADD[Добавить эмбеддинг]
  ADD --> DONE{Набран лимит?}
  DONE -- нет --> T
  DONE -- да --> SUBMIT[Отправить регистрацию/вход]
```

## Схема БД (упрощённо)

### auth-db

```mermaid
erDiagram
  TOKEN {
    int id PK
    varchar token
    int user_id "unique"
    varchar password_hash
    timestamptz created_at
  }

  FACE_TEMPLATES {
    int id PK
    int user_id
    jsonb embedding
    timestamptz created_at
    timestamptz updated_at
  }
```

### users-db

```mermaid
erDiagram
  USERS {
    int id PK
    varchar name
    varchar email "unique"
    enum role
    int specialization_id
    int[] counter_agent_specialization_ids
    varchar phone
    date birth_date
    varchar city
    varchar address
    text photo_url
    timestamptz created_at
    timestamptz updated_at
    timestamptz archived_at
  }

  USER_STATS {
    int id PK
    int user_id "unique"
    int courses_enrolled
    int courses_authored
    int lessons_total
    int lessons_completed
    int quizzes_total
    int quizzes_passed
    numeric average_score
    int streak_days
    timestamptz last_active_at
  }

  USER_HISTORY {
    int id PK
    int user_id
    int actor_id
    enum action
    jsonb changes
    timestamptz created_at
  }

  USERS ||--|| USER_STATS : "user_id"
  USERS ||--o{ USER_HISTORY : "user_id"
```

Примечание: `user_id` в `auth-db` логически ссылается на `users-db.users.id`, но физически это разные базы, поэтому FK между ними не создаётся.

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

## Тестирование (схема и отчёт)

### Схема прогона

```mermaid
flowchart TD
  S[Start] --> DC[docker compose up]
  DC --> H[/health checks/]
  H --> F1[Functional: capture]
  F1 --> F2[Functional: register-with-embeddings]
  F2 --> F3[Functional: login-with-embeddings]
  F3 --> N1[Negative: чужое лицо/порог]
  N1 --> L1[Load: capture scenario]
  L1 --> L2[Load: login scenario]
  L2 --> R[Report]
```

### Отчёт (пример, как после прогона)

Функциональные тесты (API):
- Пройдено: **18/18**
- Критических ошибок: **0**
- Основные сценарии:
  - capture → register-with-embeddings → login-with-embeddings (успех)
  - login с чужим лицом (ожидаемый отказ)
  - неверный формат изображения (400)
  - превышение размера кадра (413, корректная обработка на клиенте)

Нагрузочные тесты (пример метрик):

| Сценарий | Профиль | p95 | p99 | Ошибки |
|---|---|---|---|---|
| `/auth/face/capture` | 10 VU, 3 мин | 520 ms | 780 ms | 0.6% |
| `/auth/face/login-with-embeddings` | 5 VU, 3 мин | 240 ms | 360 ms | 0% |
| `/auth/face/register-with-embeddings` | 2 VU, 2 мин | 410 ms | 610 ms | 0% |

Примечание: значения выше — шаблон отчёта и заполняются фактическими цифрами после реального прогона.

## Примечания

- В docker контейнере **нет GUI**, поэтому используется только `FaceIDCli`.
- В случае ошибки `413 Payload Too Large` фронт автоматически снижает качество кадра.
- Точность определяется порогом `FACE_MATCH_THRESHOLD` (см. `docker-compose.yml`).
