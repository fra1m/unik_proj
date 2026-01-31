### TODOs
| Filename | line # | TODO |
|:------|:------:|:------|
| [src/main.ts](src/main.ts#L1) | 1 | Все function вынести в отдельный файл (сервис) |
| [src/main.ts](src/main.ts#L2) | 2 | Подумать нужен ли RpcToHttpExceptionFilter |
| [src/contracts/patterns.ts](src/contracts/patterns.ts#L1) | 1 | под каждый сервис сделать отдельные патерны - вынести auth и тд |
| [src/modules/auth/auth.service.ts](src/modules/auth/auth.service.ts#L1) | 1 | Проверить название патернов. Провить их логику в микросервисе auth |
| [src/modules/users/users.controller.ts](src/modules/users/users.controller.ts#L92) | 92 | доделать авторизацию |
| [src/modules/users/users.service.ts](src/modules/users/users.service.ts#L1) | 1 | Проверить название патернов. Провить их логику в микросервисе users |

### FIXMEs
| Filename | line # | FIXME |
|:------|:------:|:------|
| [src/needToTest/auth/auth.service.spec.ts](src/needToTest/auth/auth.service.spec.ts#L1) | 1 | Написать тесты к auth заново, сейчас не работают |
| [src/needToTest/common/registerRmqClient.spec.ts](src/needToTest/common/registerRmqClient.spec.ts#L1) | 1 | Написать тесты к registerRmqClient заново, сейчас не работают |
| [src/needToTest/common/rpc.spec.ts](src/needToTest/common/rpc.spec.ts#L1) | 1 | Написать тесты к rpc заново, сейчас не работают |
| [src/needToTest/users/users.service.spec.ts](src/needToTest/users/users.service.spec.ts#L1) | 1 | Написать тесты к users заново, сейчас не работают |
