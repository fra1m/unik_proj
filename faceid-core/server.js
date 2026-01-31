const http = require('http')
const fs = require('fs')
const path = require('path')
const os = require('os')
const { execFile, spawn } = require('child_process')

const port = Number(process.env.PORT || 3010)
const dataPath = process.env.DATA_PATH || path.join(__dirname, 'data', 'last_embedding.json')
const faceidCliPath = process.env.FACEID_CLI_PATH || '/app/faceid-bin/FaceIDCli'
const faceidCliCwd = process.env.FACEID_CLI_CWD || '/app/faceid-bin'
const rawFaceidAppPath = process.env.FACEID_APP_PATH || ''
const rawFaceidAppCwd = process.env.FACEID_APP_CWD || ''
const faceidAppCwd =
  rawFaceidAppCwd || (rawFaceidAppPath ? path.dirname(rawFaceidAppPath) : '')
const resolvedFaceidAppCwd = faceidAppCwd
  ? path.isAbsolute(faceidAppCwd)
    ? faceidAppCwd
    : path.resolve(process.cwd(), faceidAppCwd)
  : ''
const faceidAppPath = rawFaceidAppPath
  ? path.isAbsolute(rawFaceidAppPath)
    ? rawFaceidAppPath
    : rawFaceidAppCwd
      ? path.resolve(rawFaceidAppCwd, rawFaceidAppPath)
      : path.resolve(process.cwd(), rawFaceidAppPath)
  : ''
const faceidAppArgs = (process.env.FACEID_APP_ARGS || '')
  .split(' ')
  .map((item) => item.trim())
  .filter(Boolean)

let faceidAppProcess = null

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload)
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  })
  res.end(body)
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []
    req.on('data', (chunk) => chunks.push(chunk))
    req.on('end', () => {
      try {
        const raw = Buffer.concat(chunks).toString('utf8')
        const parsed = raw ? JSON.parse(raw) : {}
        resolve(parsed)
      } catch (err) {
        reject(err)
      }
    })
    req.on('error', reject)
  })
}

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true })
}

function decodeImageBase64(input) {
  if (typeof input !== 'string' || !input.trim()) {
    throw new Error('imageBase64 is required')
  }
  const trimmed = input.trim()
  const dataUrlMatch = trimmed.match(/^data:(.+);base64,(.*)$/)
  const base64 = dataUrlMatch ? dataUrlMatch[2] : trimmed
  const buffer = Buffer.from(base64, 'base64')
  if (!buffer.length) {
    throw new Error('Failed to decode base64 image')
  }
  return buffer
}

function runFaceIdCli(imagePath) {
  return new Promise((resolve, reject) => {
    execFile(faceidCliPath, [imagePath], { cwd: faceidCliCwd }, (err, stdout, stderr) => {
      if (err) {
        const message = stderr || stdout || err.message
        reject(new Error(message))
        return
      }
      try {
        const parsed = JSON.parse(stdout)
        if (!parsed || !Array.isArray(parsed.embedding)) {
          reject(new Error('FaceID CLI returned invalid JSON'))
          return
        }
        const embedding = parsed.embedding.map((v) => Number(v)).filter(Number.isFinite)
        if (!embedding.length) {
          reject(new Error('Empty embedding from FaceID CLI'))
          return
        }
        const pose = parsed.pose && typeof parsed.pose === 'object'
          ? {
              valid: !!parsed.pose.valid,
              yawNorm: Number(parsed.pose.yawNorm),
              pitchNorm: Number(parsed.pose.pitchNorm),
              magnitude: Number(parsed.pose.magnitude),
              slot: Number(parsed.pose.slot),
            }
          : undefined
        resolve({ embedding, pose })
      } catch (parseErr) {
        reject(new Error(`Failed to parse CLI output: ${String(parseErr)}`))
      }
    })
  })
}

function writeLatestEmbedding(embedding) {
  ensureDir(dataPath)
  const updatedAt = new Date().toISOString()
  const payload = {
    embedding,
    length: embedding.length,
    updatedAt,
    timestamp: updatedAt,
  }
  fs.writeFileSync(dataPath, JSON.stringify(payload, null, 2), 'utf8')
}

function readLatestEmbeddingPayload() {
  const raw = fs.readFileSync(dataPath, 'utf8')
  const parsed = JSON.parse(raw)
  if (!parsed || !Array.isArray(parsed.embedding)) {
    throw new Error('Invalid embedding payload')
  }
  const embedding = parsed.embedding.map((v) => Number(v)).filter(Number.isFinite)
  if (!embedding.length) {
    throw new Error('Empty embedding')
  }
  let updatedAt = null
  if (typeof parsed.updatedAt === 'string' && parsed.updatedAt.trim()) {
    updatedAt = parsed.updatedAt
  } else if (typeof parsed.timestamp === 'string' && parsed.timestamp.trim()) {
    updatedAt = parsed.timestamp
  } else {
    const stat = fs.statSync(dataPath)
    updatedAt = stat.mtime.toISOString()
  }
  return { embedding, length: embedding.length, updatedAt }
}

function getAppStatus() {
  const running = !!(faceidAppProcess && !faceidAppProcess.killed)
  return {
    running,
    pid: running ? faceidAppProcess.pid : undefined,
    appPath: faceidAppPath || undefined,
    appCwd: resolvedFaceidAppCwd || undefined,
    appArgs: faceidAppArgs.length ? faceidAppArgs : undefined,
  }
}

function ensureAppConfig() {
  if (!faceidAppPath) {
    throw new Error('FACEID_APP_PATH is not configured')
  }
  if (!fs.existsSync(faceidAppPath)) {
    throw new Error(`FaceID app not found: ${faceidAppPath}`)
  }
}

function startFaceIdApp() {
  ensureAppConfig()
  if (faceidAppProcess && !faceidAppProcess.killed) {
    return { started: false, ...getAppStatus() }
  }

  return new Promise((resolve, reject) => {
    let child
    try {
      child = spawn(faceidAppPath, faceidAppArgs, {
        cwd: resolvedFaceidAppCwd || process.cwd(),
        stdio: 'ignore',
      })
    } catch (err) {
      reject(err)
      return
    }

    const onError = (err) => {
      child.removeListener('spawn', onSpawn)
      reject(err)
    }
    const onSpawn = () => {
      child.removeListener('error', onError)
      faceidAppProcess = child
      child.on('exit', () => {
        faceidAppProcess = null
      })
      resolve({ started: true, ...getAppStatus() })
    }

    child.once('error', onError)
    child.once('spawn', onSpawn)
  })
}

function stopFaceIdApp() {
  if (!faceidAppProcess || faceidAppProcess.killed) {
    return { stopped: false, ...getAppStatus() }
  }
  faceidAppProcess.kill('SIGTERM')
  return { stopped: true, ...getAppStatus() }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`)

  if (req.method === 'GET' && url.pathname === '/health') {
    return sendJson(res, 200, {
      status: 'ok',
      service: 'faceid-core',
      dataPath,
      faceidCliPath,
      faceidCliCwd,
      faceidAppPath: faceidAppPath || undefined,
      faceidAppCwd: faceidAppCwd || undefined,
      timestamp: new Date().toISOString(),
    })
  }

  if (req.method === 'GET' && url.pathname === '/app/status') {
    return sendJson(res, 200, getAppStatus())
  }

  if (req.method === 'POST' && url.pathname === '/app/start') {
    try {
      const status = await startFaceIdApp()
      return sendJson(res, 200, status)
    } catch (err) {
      return sendJson(res, 500, {
        error: 'Failed to start FaceID app',
        message: String(err && err.message ? err.message : err),
      })
    }
  }

  if (req.method === 'POST' && url.pathname === '/app/stop') {
    try {
      const status = stopFaceIdApp()
      return sendJson(res, 200, status)
    } catch (err) {
      return sendJson(res, 500, {
        error: 'Failed to stop FaceID app',
        message: String(err && err.message ? err.message : err),
      })
    }
  }

  if (req.method === 'GET' && url.pathname === '/embedding/latest') {
    try {
      const payload = readLatestEmbeddingPayload()
      return sendJson(res, 200, {
        embedding: payload.embedding,
        length: payload.length,
        updatedAt: payload.updatedAt,
        dataPath,
      })
    } catch (err) {
      return sendJson(res, 500, {
        error: 'Failed to read embedding',
        message: String(err && err.message ? err.message : err),
        dataPath,
      })
    }
  }

  if (req.method === 'POST' && url.pathname === '/embedding/from-image') {
    try {
      const body = await readJsonBody(req)
      const buffer = decodeImageBase64(body.imageBase64)
      const tmpPath = path.join(os.tmpdir(), `faceid-${Date.now()}.jpg`)
      fs.writeFileSync(tmpPath, buffer)

      const { embedding, pose } = await runFaceIdCli(tmpPath)
      writeLatestEmbedding(embedding)

      return sendJson(res, 200, {
        embedding,
        length: embedding.length,
        pose,
        updatedAt: new Date().toISOString(),
        dataPath,
      })
    } catch (err) {
      return sendJson(res, 500, {
        error: 'Failed to compute embedding',
        message: String(err && err.message ? err.message : err),
      })
    }
  }

  return sendJson(res, 404, { error: 'Not found' })
})

server.listen(port, '0.0.0.0', () => {
  // eslint-disable-next-line no-console
  console.log(`[faceid-core] listening on http://0.0.0.0:${port}`)
  // eslint-disable-next-line no-console
  console.log(`[faceid-core] data path: ${dataPath}`)
  // eslint-disable-next-line no-console
  console.log(`[faceid-core] cli: ${faceidCliPath} (cwd=${faceidCliCwd})`)
})
