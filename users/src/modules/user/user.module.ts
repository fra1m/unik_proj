import { Module } from '@nestjs/common';
import { UserService } from './user.service';
import { UserController } from './user.controller';
import { TypeOrmModule } from '@nestjs/typeorm';
import { UserEntity } from './entities/user.entity';
import { UserStatsEntity } from './entities/user-stats.entity';
import { UserHistoryEntity } from './entities/user-history.entity';

@Module({
  imports: [
    TypeOrmModule.forFeature([UserEntity, UserStatsEntity, UserHistoryEntity]),
  ],
  controllers: [UserController],
  providers: [UserService],
})
export class UserModule {}
