import { Module } from '@nestjs/common';
import { AuthService } from './auth.service';
import { RmqModule } from 'src/common/rmq/rmq.module';
import { UsersModule } from '../users/users.module';
import { FaceAuthController } from './face-auth.controller';
import { FaceCoreService } from './face-core.service';

@Module({
  imports: [RmqModule.forAuth(), UsersModule],
  controllers: [FaceAuthController],
  providers: [AuthService, FaceCoreService],
  exports: [AuthService],
})
export class AuthModule {}
