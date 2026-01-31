// TODO: под каждый сервис сделать отдельные патерны - вынести auth и тд
export const PATTERNS = {
  /** 'auth.hashPassword' */
  AUTH_HASH: 'auth.hashPassword',
  /** 'auth.createCredentials' */
  AUTH_CREDENTIALS: 'auth.createCredentials',
  /** 'auth.generateTokens' */
  AUTH_GENERATE_TOKENS: 'auth.generateTokens',
  /** */
  AUTH_LOGIN_BY_PASSWORD: 'auth.authByPassword',
  /** 'auth.validateAccess' */
  AUTH_ACCESS_VALIDATE: 'auth.validateAccess',
  /** 'auth.validateRefresh'*/
  AUTH_REFRESH_VALIDATE: 'auth.validateRefresh',
  /** 'auth.removeToken' */
  AUTH_REMOVE_TOKEN: 'auth.removeToken',
  /** 'auth.faceEnroll' */
  AUTH_FACE_ENROLL: 'auth.faceEnroll',
  /** 'auth.faceVerify' */
  AUTH_FACE_VERIFY: 'auth.faceVerify',

  /** 'users.applyQuizStats' */
  USERS_APPLY_QUIZ_STATS: 'users.applyQuizStats',
  /** 'users.create' */
  USERS_CREATE: 'users.create',
  /** 'users.getByEmail' */
  USERS_BY_EMAIL: 'users.getByEmail',
  /** 'users.getAll' */
  USERS_ALL: 'users.getAll',
  /** 'users.getUserById' */
  USERS_BY_ID: 'users.getUserById',
  /** 'user.getStats' */
  USERS_GET_STATS: 'users.getStats',
  /** 'users.update' */
  USERS_UPDATE: 'users.update',
  /** 'users.archive' */
  USERS_ARCHIVE: 'users.archive',
  /** 'users.restore' */
  USERS_RESTORE: 'users.restore',
  /** 'users.history' */
  USERS_HISTORY: 'users.history',

  /** 'courses.create' */
  COURSES_CREATE: 'courses.create',
  /** 'courses.getAll' */
  COURSES_ALL: 'courses.getAll',
  /** 'courses.getById' */
  COURSES_GET_BY_ID: 'courses.getById',
  /** 'courses.delete' */
  COURSES_DELETE: 'courses.delete',
  /** 'courses.update' */
  COURSES_UPDATE: 'courses.update',

  /** 'specializations.getAll' */
  SPECIALIZATIONS_ALL: 'specializations.getAll',
  /** 'specializations.create' */
  SPECIALIZATIONS_CREATE: 'specializations.create',
  /** 'specializations.update' */
  SPECIALIZATIONS_UPDATE: 'specializations.update',

  /** 'analytics.submit' */
  ANALYTICS_SUBMIT: 'analytics.submit',

  /** 'lessons.create' */
  LESSONS_CREATE: 'lessons.create',
  /** 'lessons.getById' */
  LESSONS_GET_BY_ID: 'lessons.getById',
  /** 'lessons.getAllLite' */
  LESSONS_ALL_LITE: 'lessons.getAllLite',
  /** 'lessons.getQuizzesAndLessonsTotals' */
  LESSONS_GET_TOTALS: 'lessons.getQuizzesAndLessonsTotals',

  /** 'quizzes.create' */
  QUIZZES_CREATE: 'quizzes.create',
  /** 'quizzes.getByUserId' */
  QUIZZES_GET_BY_USER_ID: 'quizzes.getByUserId',
  /** 'quizzes.updateByUserId' */
  QUIZZES_UPDATE_BY_USER_ID: 'quizzes.updateByUserId',
  /** 'quizzes.delete' */
  QUIZZES_DELETE: 'quizzes.delete',

  /** 'videos.create' */
  VIDEOS_CREATE: 'videos.create',
  /** 'videos.getAll' */
  VIDEOS_GET_ALL: 'videos.getAll',
  /** 'videos.getById' */
  VIDEOS_GET_BY_ID: 'videos.getById',
  /** 'videos.update' */
  VIDEOS_UPDATE: 'videos.update',
  /** 'videos.delete' */
  VIDEOS_DELETE: 'videos.delete',
} as const;
