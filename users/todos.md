### TODOs
| Filename | line # | TODO |
|:------|:------:|:------|
| [src/app.module.ts](src/app.module.ts#L22) | 22 | Использовать когда перейдешь на урл к бд |
| [src/config/validation.ts](src/config/validation.ts#L23) | 23 | Убрать Postgres переменные и оставить только ссылку на бд, так же в .env тоже изменить - будет урл на БД |
| [src/modules/user/user.controller.ts](src/modules/user/user.controller.ts#L1) | 1 | Протестируй 'users.create' |
| [src/modules/user/user.controller.ts](src/modules/user/user.controller.ts#L52) | 52 | Добавить пагинацию и исправить getAllUsers |
| [src/modules/user/user.service.ts](src/modules/user/user.service.ts#L132) | 132 | (АВТОРИЗАЦИЯ): убрать генерацию токена и проверку пароля - это в микросервис auth |
| [src/modules/user/user.service.ts](src/modules/user/user.service.ts#L174) | 174 | Добавить обновление имени/email и тд |

### FIXMEs
| Filename | line # | FIXME |
|:------|:------:|:------|
| [src/modules/user/user.service.ts](src/modules/user/user.service.ts#L130) | 130 | Исправить авторизацию - костыль, правильное решение ниже |
