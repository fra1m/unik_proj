import { Module } from '@nestjs/common';
import { UsersService } from './users.service';
import { UsersController } from './users.controller';
import { RmqModule } from 'src/common/rmq/rmq.module';

@Module({
  imports: [RmqModule.forUsers()],
  controllers: [UsersController],
  providers: [UsersService],
  exports: [UsersService],
})
export class UsersModule {}
