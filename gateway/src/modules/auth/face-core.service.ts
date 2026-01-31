import { BadGatewayException, Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

type FaceCoreResponse = {
  embedding?: number[];
  length?: number;
  updatedAt?: string;
  dataPath?: string;
  pose?: {
    valid?: boolean;
    yawNorm?: number;
    pitchNorm?: number;
    magnitude?: number;
    slot?: number;
  };
  error?: string;
  message?: string;
};

type FaceCoreAppResponse = {
  running?: boolean;
  pid?: number;
  started?: boolean;
  stopped?: boolean;
  appPath?: string;
  appCwd?: string;
  error?: string;
  message?: string;
};

@Injectable()
export class FaceCoreService {
  private readonly baseUrl: string;

  constructor(private readonly cfg: ConfigService) {
    this.baseUrl =
      this.cfg.get<string>('FACE_CORE_URL', { infer: true }) ??
      'http://localhost:3010';
  }

  async getLatestEmbedding(): Promise<number[]> {
    const url = `${this.baseUrl}/embedding/latest`;
    let response: Response;
    try {
      response = await fetch(url);
    } catch (err) {
      throw new BadGatewayException(`Face core is unavailable: ${String(err)}`);
    }

    let payload: FaceCoreResponse;
    try {
      payload = (await response.json()) as FaceCoreResponse;
    } catch (err) {
      throw new BadGatewayException('Face core returned invalid JSON');
    }

    if (!response.ok) {
      throw new BadGatewayException(
        payload?.message || payload?.error || 'Face core error',
      );
    }

    const embedding = Array.isArray(payload.embedding)
      ? payload.embedding.map((value) => Number(value)).filter(Number.isFinite)
      : [];

    if (embedding.length < 8) {
      throw new BadGatewayException('Face core returned empty embedding');
    }

    return embedding;
  }

  async getEmbeddingFromImage(imageBase64: string): Promise<{
    embedding: number[];
    pose?: {
      valid: boolean;
      yawNorm: number;
      pitchNorm: number;
      magnitude: number;
      slot?: number;
    };
  }> {
    const url = `${this.baseUrl}/embedding/from-image`;
    let response: Response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageBase64 }),
      });
    } catch (err) {
      throw new BadGatewayException(`Face core is unavailable: ${String(err)}`);
    }

    let payload: FaceCoreResponse;
    try {
      payload = (await response.json()) as FaceCoreResponse;
    } catch {
      throw new BadGatewayException('Face core returned invalid JSON');
    }

    if (!response.ok) {
      throw new BadGatewayException(
        payload?.message || payload?.error || 'Face core error',
      );
    }

    const embedding = Array.isArray(payload.embedding)
      ? payload.embedding.map((value) => Number(value)).filter(Number.isFinite)
      : [];

    if (embedding.length < 8) {
      throw new BadGatewayException('Face core returned empty embedding');
    }

    const poseRaw = payload.pose;
    const pose =
      poseRaw && typeof poseRaw === 'object'
        ? {
            valid: !!poseRaw.valid,
            yawNorm: Number(poseRaw.yawNorm),
            pitchNorm: Number(poseRaw.pitchNorm),
            magnitude: Number(poseRaw.magnitude),
            slot: Number.isFinite(Number(poseRaw.slot))
              ? Number(poseRaw.slot)
              : undefined,
          }
        : undefined;

    return { embedding, pose };
  }

  async getLatestEmbeddingInfo(): Promise<{
    length: number;
    updatedAt?: string;
    dataPath?: string;
  }> {
    const url = `${this.baseUrl}/embedding/latest`;
    let response: Response;
    try {
      response = await fetch(url);
    } catch (err) {
      throw new BadGatewayException(`Face core is unavailable: ${String(err)}`);
    }

    let payload: FaceCoreResponse;
    try {
      payload = (await response.json()) as FaceCoreResponse;
    } catch {
      throw new BadGatewayException('Face core returned invalid JSON');
    }

    if (!response.ok) {
      throw new BadGatewayException(
        payload?.message || payload?.error || 'Face core error',
      );
    }

    const length = Number.isFinite(payload.length)
      ? Number(payload.length)
      : Array.isArray(payload.embedding)
        ? payload.embedding.length
        : 0;

    return {
      length,
      updatedAt: payload.updatedAt,
      dataPath: payload.dataPath,
    };
  }

  async getAppStatus(): Promise<{
    running: boolean;
    pid?: number;
    appPath?: string;
    appCwd?: string;
  }> {
    const url = `${this.baseUrl}/app/status`;
    let response: Response;
    try {
      response = await fetch(url);
    } catch (err) {
      throw new BadGatewayException(`Face core is unavailable: ${String(err)}`);
    }

    let payload: FaceCoreAppResponse;
    try {
      payload = (await response.json()) as FaceCoreAppResponse;
    } catch {
      throw new BadGatewayException('Face core returned invalid JSON');
    }

    if (!response.ok) {
      throw new BadGatewayException(
        payload?.message || payload?.error || 'Face core error',
      );
    }

    return {
      running: !!payload.running,
      pid: payload.pid,
      appPath: payload.appPath,
      appCwd: payload.appCwd,
    };
  }

  async startApp(): Promise<{
    started: boolean;
    running: boolean;
    pid?: number;
    appPath?: string;
    appCwd?: string;
  }> {
    const url = `${this.baseUrl}/app/start`;
    let response: Response;
    try {
      response = await fetch(url, { method: 'POST' });
    } catch (err) {
      throw new BadGatewayException(`Face core is unavailable: ${String(err)}`);
    }

    let payload: FaceCoreAppResponse;
    try {
      payload = (await response.json()) as FaceCoreAppResponse;
    } catch {
      throw new BadGatewayException('Face core returned invalid JSON');
    }

    if (!response.ok) {
      throw new BadGatewayException(
        payload?.message || payload?.error || 'Face core error',
      );
    }

    return {
      started: !!payload.started,
      running: !!payload.running,
      pid: payload.pid,
      appPath: payload.appPath,
      appCwd: payload.appCwd,
    };
  }

  async stopApp(): Promise<{
    stopped: boolean;
    running: boolean;
    pid?: number;
    appPath?: string;
    appCwd?: string;
  }> {
    const url = `${this.baseUrl}/app/stop`;
    let response: Response;
    try {
      response = await fetch(url, { method: 'POST' });
    } catch (err) {
      throw new BadGatewayException(`Face core is unavailable: ${String(err)}`);
    }

    let payload: FaceCoreAppResponse;
    try {
      payload = (await response.json()) as FaceCoreAppResponse;
    } catch {
      throw new BadGatewayException('Face core returned invalid JSON');
    }

    if (!response.ok) {
      throw new BadGatewayException(
        payload?.message || payload?.error || 'Face core error',
      );
    }

    return {
      stopped: !!payload.stopped,
      running: !!payload.running,
      pid: payload.pid,
      appPath: payload.appPath,
      appCwd: payload.appCwd,
    };
  }
}
