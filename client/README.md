# Gerion Courses - клиент (frontend)

Клиентское приложение учебной платформы: авторизация и роли, курсы и уроки,
тесты на SurveyJS, статистика, словари и PDF-материалы. Стек: React + Vite,
Redux Toolkit, Ant Design, Tailwind и Socket.IO.

## Быстрый старт

```bash
npm ci
npm run dev
```

Приложение стартует на `http://localhost:5173`.

## Скрипты

```bash
npm run dev       # dev-сервер Vite
npm run build     # production-сборка
npm run preview   # предпросмотр сборки
npm run lint      # eslint
```

## Как все устроено

### Жизненный цикл приложения

- `src/main.tsx`: поднимает store + PersistGate, подключает API-интерсепторы и realtime.
- `src/App.tsx`: устанавливает тему, запускает `checkAuth`, инициирует `appBootstrap`.
- `src/store/appBootstrap.ts`: грузит базовые данные после авторизации.
- `src/api.ts`: единый axios-инстанс + refresh логика.
- `src/realtime/realtime.ts`: Socket.IO, пересинхронизация при изменении специализации.

### Авторизация и refresh токена

1. После ре-гидрации стора `applyAuthHeader` ставит `Authorization`.
2. На старте `checkAuth` вызывает `/users/refresh`.
3. Интерсептор ловит `401` и ставит запрос в очередь:
   - если refresh успешен, все запросы переигрываются с новым токеном;
   - если нет - логаут, очистка токена и редирект на `/login`.

### Bootstrapping (инициализация данных)

- Запускается один раз, когда `authReady=true` и `isAuth=true`.
- Общие данные: `getMyStats`, `getAllCourses`, `getAllLessons`.
- Для `admin` и `teacher` дополнительно: `getAllUsers`, `getAllQuizzes`,
  `fetchSpecializations`.
- Есть защита от "смены контекста": при изменении `sessionKey` (email + specialization)
  сбрасываются кэши курсов/уроков/тестов и bootstrap запускается заново.

### Realtime

- Socket.IO подключается только при авторизации и наличии токена.
- При `specialization-changed` делается `checkAuth`, затем повторный bootstrap.
- На смену токена или логаут подключение пересоздается/закрывается.

## Маршруты и роли

### Публичные

| Путь | Страница | Назначение |
| --- | --- | --- |
| `/login` | LoginPage | Вход |
| `/register` | RegisterPageForAdmin | Регистрация |
| `/calendar` | DictionaryPage | Словарь |

### Приватные

| Путь | Роли | Назначение |
| --- | --- | --- |
| `/` | student, teacher, admin | Главная |
| `/courses` | student, teacher, admin | Курсы |
| `/course-builder` | teacher, admin | Конструктор курсов |
| `/lesson` | student, teacher, admin | Урок (форма) |
| `/lesson-builder` | teacher, admin | Конструктор уроков |
| `/lessons/:id` | student, teacher, admin | Просмотр урока |
| `/quiz` | student, teacher, admin | Прохождение теста |
| `/quizzes` | teacher, admin | Список тестов |
| `/quiz-builder` | teacher, admin | Конструктор тестов |
| `/profile` | user, student, teacher, admin | Профиль |
| `/users` | admin | Пользователи |
| `/specializations` | admin | Специализации |

Роуты описаны в `src/routes/index.ts`. Guards: `src/components/Require/`.

## Состояние (Redux)

Store собирается в `src/store/store.ts` с расширенной `serializableCheck`
для `File`, `Blob`, `ArrayBuffer`.

Persist (sessionStorage) на уровне `src/store/reducers/reducers.ts`:

- `user` - без `isLoading`, `saveError`, `authReady`, `accessToken`.
- `app` - полностью.
- `course`, `lesson`, `quiz`, `specialization` - без временных флагов.
- `pdf` - не персистится.

Ключевые слайсы:

- `app`: `bootstrapped`, `lastBootstrapAt`, `sessionKey`.
- `user`: `isAuth`, `role`, `users`, `myStats`, `specialization`, `authReady`.
- `course`: выбранный курс, список курсов, файл, `specializationId`.
- `lesson`: список уроков, PDF-страницы (`startWith`, `end`), привязка к курсу/тесту.
- `quiz`: `surveyJson`, список тестов, `lastCreatedId`, `isUpdate`.
- `specialization`: `items`, `byId`, `lastFetchedAt`.
- `pdf`: состояние глобального PDF-превью (open, blobUrl, error).

## API: используемые эндпоинты

Базовый префикс задается `VITE_API_BASE` (по умолчанию `/course_api`).

### Auth / Users

- `POST /users/login`
- `POST /users/logout`
- `POST /users/refresh`
- `POST /users/registration`
- `POST /users/admin/create`
- `GET /users/all`
- `PATCH /users/:id`
- `DELETE /users/delete`
- `GET /users/me/stats`
- `PATCH /users/patch` (смена пароля)

### Courses

- `POST /courses/create` (FormData + файл)
- `PATCH /courses/update`
- `GET /courses/getAllCourses`
- `DELETE /courses/delete`
- `GET /courses/:id/file` (PDF-превью)

### Lessons

- `POST /lessons/create`
- `GET /lessons/all`
- `GET /lessons/:id/content` (ArrayBuffer PDF)

### Quiz + Analytics

- `POST /quiz/create`
- `PATCH /quiz/update`
- `GET /quiz/all`
- `DELETE /quiz/delete`
- `POST /analytics/quiz/submit`

### Specializations

- `GET /specializations/all`
- `POST /specializations/create`
- `PATCH /specializations/:id`
- `DELETE /specializations/:id`

## Доменные модули

- Пользователи: список, роли, специализации, быстрое создание в админке.
- Курсы: создание/редактирование, загрузка файла (PDF), список курсов.
- Уроки: диапазоны страниц PDF, привязка к курсу и тесту, просмотр урока.
- Тесты: SurveyJS JSON, конструктор, прохождение и сохранение результатов.
- Специализации: CRUD доступен только админу.
- Статистика: `getMyStats` обновляется после отправки результатов теста.

## PDF и просмотр материалов

- `src/components/Forms/LessonsForm.tsx` отображает PDF урока через `pdfjs-dist`.
- `src/components/Modals/GlobalPdfPreview.tsx` - глобальный модал для курса,
  с ресайзом и drag-and-drop.

## UI и тема

- `useThemeMode` хранит тему в `localStorage` (`ui-theme`).
- `Nav` переключает тему и кидает событие `ui-themechange`.
- QuizPage слушает событие и меняет тему SurveyJS.

## Структура проекта

```
src/
  App.tsx            # каркас + тема + bootstrap
  main.tsx           # входная точка
  api.ts             # axios + refresh-логика
  routes/            # маршруты и роли
  store/             # redux + thunks + persist
  pages/             # страницы
  components/        # формы, модалки, ui
  hooks/             # typed hooks
  models/            # типы данных
  utils/             # утилиты
  realtime/          # socket.io
  assets/
public/
  dicts/             # словари PDF
```

## Переменные окружения и прокси

- `VITE_API_BASE` - базовый URL API (по умолчанию `/course_api`).
- `DOCKERIZED=1` - для dev-режима в Docker (см. `vite.config.ts`).

Vite proxy (`vite.config.ts`):

- `/course_api/*` -> `http://localhost:3000/api/v1/*`
- `/socket.io/*` -> `http://localhost:3001`

Продовый прокси - `docker/nginx.conf`.

## Docker

Сборка и запуск прод-версии:

```bash
docker build -t gerion-client .
docker run --rm -p 8080:80 gerion-client
```

Dev-контейнер также описан в `Dockerfile` (stage `dev`).

## Как расширять

Минимальный чек-лист:

1. Страница: `src/pages/...`, затем добавить маршрут в `src/routes/index.ts`.
2. Состояние: добавить slice + thunks и подключить в `src/store/reducers/reducers.ts`.
3. UI: подключить в навигации (label в routes) и при необходимости в `Nav`.
4. API: использовать `api` из `src/api.ts` для единых интерсепторов.
