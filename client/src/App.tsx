import { useEffect, useMemo, useRef, useState } from 'react'

type Tokens = {
  accessToken: string
  refreshToken: string
}

type User = {
  sub: number
  email: string
  name: string
  role: string
}

type RegisterResponse = {
  user: User
  tokens: Tokens
  face: { samples: number }
}

type LoginResponse = {
  user: User
  tokens: Tokens
  verify: { matched: boolean; score: number; threshold: number }
}

type CaptureResponse = {
  embedding: number[]
  length: number
  pose?: {
    valid: boolean
    yawNorm: number
    pitchNorm: number
    magnitude: number
    slot?: number
  }
}

type Mode = 'register' | 'login'

type Stage = 'idle' | 'capturing' | 'submitting' | 'success' | 'error'

type ApiError = { status?: number; message: string }

type QualityMode = 'standard' | 'low'

type QualityPreset = { maxWidth: number; jpegQuality: number }

const AUTO_INTERVAL_MS = 900
const RING_GAP_DEG = 10
const RING_SIZE = 320
const RING_RADIUS = 126

const QUALITY_PRESETS: Record<QualityMode, QualityPreset> = {
  standard: { maxWidth: 720, jpegQuality: 0.86 },
  low: { maxWidth: 480, jpegQuality: 0.7 },
}

const REGISTER_SAMPLES = 8
const LOGIN_SAMPLES = 3

const polarToCartesian = (cx: number, cy: number, r: number, angle: number) => {
  const rad = ((angle - 90) * Math.PI) / 180
  return {
    x: cx + r * Math.cos(rad),
    y: cy + r * Math.sin(rad),
  }
}

const describeArc = (
  cx: number,
  cy: number,
  r: number,
  startAngle: number,
  endAngle: number,
) => {
  const start = polarToCartesian(cx, cy, r, endAngle)
  const end = polarToCartesian(cx, cy, r, startAngle)
  const largeArcFlag = endAngle - startAngle <= 180 ? '0' : '1'
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`
}

const buildRingSegments = (segments: number, gapDeg: number) => {
  const cx = RING_SIZE / 2
  const cy = RING_SIZE / 2
  const step = 360 / segments
  const gap = Math.min(gapDeg, step * 0.6)
  return Array.from({ length: segments }, (_, index) => {
    const start = index * step + gap / 2
    const end = (index + 1) * step - gap / 2
    return describeArc(cx, cy, RING_RADIUS, start, end)
  })
}

// no pose gating in auto mode

function App() {
  const apiBaseUrl = useMemo(() => {
    const fromEnv = import.meta.env.VITE_API_URL as string | undefined
    return fromEnv ?? 'http://localhost:3000/api/v1'
  }, [])

  const [mode, setMode] = useState<Mode>('register')
  const [stage, setStage] = useState<Stage>('idle')
  const [email, setEmail] = useState('vasili@example.com')
  const [name, setName] = useState('Vasili')
  const [threshold, setThreshold] = useState('0.55')

  const [result, setResult] = useState('')
  const [cameraOn, setCameraOn] = useState(false)
  const [cameraReady, setCameraReady] = useState(false)
  const [cameraBusy, setCameraBusy] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [captureError, setCaptureError] = useState('')
  const [samples, setSamples] = useState<number[][]>([])
  const [qualityMode, setQualityMode] = useState<QualityMode>('standard')
  const [autoCollect, setAutoCollect] = useState(false)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const autoTimerRef = useRef<number | null>(null)
  const captureInFlightRef = useRef(false)
  const submitInFlightRef = useRef(false)
  const ringSegments = mode === 'register' ? REGISTER_SAMPLES : LOGIN_SAMPLES
  const ringPaths = useMemo(() => buildRingSegments(ringSegments, RING_GAP_DEG), [ringSegments])

  const minSamples = mode === 'register' ? REGISTER_SAMPLES : LOGIN_SAMPLES
  const maxSamples = minSamples
  const progress = Math.min(samples.length / maxSamples, 1) * 100
  const filledSegments = Math.min(
    ringSegments,
    samples.length,
  )
  const activeSegment = -1

  const request = async <T,>(path: string, payload: unknown): Promise<T> => {
    const response = await fetch(`${apiBaseUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(payload),
    })

    const text = await response.text()
    if (!response.ok) {
      throw {
        status: response.status,
        message: text || response.statusText || `HTTP ${response.status}`,
      } as ApiError
    }

    if (!text) return {} as T
    return JSON.parse(text) as T
  }

  const mapError = (err: unknown, context: 'capture' | 'submit'): string => {
    const fallback = context === 'capture'
      ? 'Не удалось получить эмбеддинг. Проверь камеру и попробуй ещё раз.'
      : 'Не удалось выполнить запрос. Попробуй ещё раз.'

    if (!err || typeof err !== 'object') return fallback
    const status = (err as ApiError).status
    let message = (err as ApiError).message || ''

    if (message.trim().startsWith('{')) {
      try {
        const parsed = JSON.parse(message)
        if (typeof parsed?.message === 'string') {
          message = parsed.message
        } else if (Array.isArray(parsed?.message)) {
          message = parsed.message.join(', ')
        }
      } catch {
        // ignore JSON parse errors
      }
    }

    if (status === 413) {
      return context === 'capture'
        ? 'Сервер отклонил кадр: слишком большой размер (HTTP 413). Мы снизили качество. Попробуй ещё раз или подойди ближе к камере.'
        : 'Сервер отклонил запрос из‑за размера (HTTP 413). Попробуй повторить или уменьшить качество.'
    }

    if (message.trim().startsWith('<!DOCTYPE') || message.trim().startsWith('<html')) {
      return fallback
    }

    const lowered = message.toLowerCase()
    if (lowered.includes('face core') || lowered.includes('faceid-core')) {
      return 'Сервис FaceID недоступен. Проверь, что faceid-core запущен.'
    }
    if (lowered.includes('no face detected')) {
      return 'Лицо не обнаружено. Посмотри прямо в камеру и попробуй ещё раз.'
    }
    if (lowered.includes('empty embedding')) {
      return 'Эмбеддинг не получен. Попробуй ещё раз, держи лицо ровно.'
    }
    if (lowered.includes('failed to compute embedding')) {
      return 'Не удалось посчитать эмбеддинг. Попробуй улучшить освещение.'
    }

    return message || fallback
  }

  const stopCamera = () => {
    const stream = streamRef.current
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
    }
    streamRef.current = null
    setCameraOn(false)
    setCameraReady(false)
    setAutoCollect(false)
  }

  const startCamera = async (): Promise<boolean> => {
    setCameraError('')
    setCameraBusy(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      streamRef.current = stream
      setCameraOn(true)
      return true
    } catch (err) {
      setCameraError(err instanceof Error ? err.message : String(err))
      stopCamera()
      return false
    } finally {
      setCameraBusy(false)
    }
  }

  useEffect(() => {
    if (!cameraOn) {
      setCameraReady(false)
      return
    }

    const video = videoRef.current
    const stream = streamRef.current
    if (!video || !stream) {
      return
    }

    video.srcObject = stream
    const handleReady = () => setCameraReady(true)
    video.addEventListener('loadedmetadata', handleReady)
    video.addEventListener('canplay', handleReady)
    void video.play().catch(() => undefined)

    return () => {
      video.removeEventListener('loadedmetadata', handleReady)
      video.removeEventListener('canplay', handleReady)
    }
  }, [cameraOn])

  useEffect(() => {
    return () => stopCamera()
  }, [])

  useEffect(() => {
    setSamples([])
    setCaptureError('')
    setResult('')
    setStage('idle')
    setAutoCollect(false)
  }, [mode])

  const captureFrameBase64 = (): string => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) {
      throw new Error('Камера ещё не готова')
    }

    const preset = QUALITY_PRESETS[qualityMode]
    const rawWidth = video.videoWidth || 640
    const rawHeight = video.videoHeight || 480
    const scale = Math.min(1, preset.maxWidth / rawWidth)
    const width = Math.max(1, Math.round(rawWidth * scale))
    const height = Math.max(1, Math.round(rawHeight * scale))

    canvas.width = width
    canvas.height = height

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      throw new Error('Не удалось подготовить кадр')
    }
    ctx.drawImage(video, 0, 0, width, height)
    return canvas.toDataURL('image/jpeg', preset.jpegQuality)
  }

  const captureSample = async () => {
    if (!cameraOn || !cameraReady || captureInFlightRef.current || stage !== 'capturing') {
      return
    }
    if (samples.length >= maxSamples) {
      return
    }

    captureInFlightRef.current = true
    try {
      const imageBase64 = captureFrameBase64()
      const data = await request<CaptureResponse>('/auth/face/capture', {
        imageBase64,
      })
      setSamples((prev) => {
        if (prev.length >= maxSamples) return prev
        return [...prev, data.embedding]
      })
      setCaptureError('')
    } catch (error) {
      const msg = mapError(error, 'capture')
      setCaptureError(msg)
      if ((error as ApiError)?.status === 413) {
        setQualityMode('low')
      }
    } finally {
      captureInFlightRef.current = false
    }
  }

  useEffect(() => {
    if (!autoCollect || stage !== 'capturing' || !cameraReady) {
      if (autoTimerRef.current) {
        window.clearInterval(autoTimerRef.current)
        autoTimerRef.current = null
      }
      return
    }
    autoTimerRef.current = window.setInterval(() => {
      void captureSample()
    }, AUTO_INTERVAL_MS)
    return () => {
      if (autoTimerRef.current) {
        window.clearInterval(autoTimerRef.current)
        autoTimerRef.current = null
      }
    }
  }, [autoCollect, stage, cameraReady, qualityMode])

  useEffect(() => {
    if (samples.length >= maxSamples && autoCollect) {
      setAutoCollect(false)
    }
  }, [samples.length, maxSamples, autoCollect])

  const startFlow = async () => {
    setResult('')
    setCaptureError('')
    setCameraError('')
    setSamples([])
    setAutoCollect(false)

    const ready = cameraOn ? true : await startCamera()
    if (ready) {
      setStage('capturing')
      setAutoCollect(true)
    } else {
      setStage('error')
    }
  }

  const submit = async () => {
    if (submitInFlightRef.current) return
    submitInFlightRef.current = true
    setStage('submitting')
    setResult('')

    try {
      if (mode === 'register') {
        const data = await request<RegisterResponse>('/auth/face/register-with-embeddings', {
          email,
          name,
          embeddings: samples,
        })
        setResult(
          `РЕГИСТРАЦИЯ УСПЕШНА\nuser=${data.user.email} id=${data.user.sub}\nfaceSamples=${data.face.samples}`,
        )
      } else {
        const parsedThreshold = Number(threshold)
        const data = await request<LoginResponse>('/auth/face/login-with-embeddings', {
          email,
          embeddings: samples,
          threshold: Number.isFinite(parsedThreshold) ? parsedThreshold : undefined,
        })
        setResult(
          `ВХОД УСПЕШЕН\nuser=${data.user.email} id=${data.user.sub}\nscore=${data.verify.score.toFixed(3)} threshold=${data.verify.threshold.toFixed(3)}`,
        )
      }
      setStage('success')
    } catch (error) {
      setStage('error')
      setResult(mapError(error, 'submit'))
    } finally {
      submitInFlightRef.current = false
    }
  }

  useEffect(() => {
    if (stage === 'capturing' && samples.length >= minSamples) {
      void submit()
    }
  }, [stage, samples.length, minSamples])

  const reset = () => {
    setStage('idle')
    setSamples([])
    setResult('')
    setCaptureError('')
    setCameraError('')
    setAutoCollect(false)
  }

  const stageLabel = (() => {
    switch (stage) {
      case 'capturing':
        return 'Сканирование лица…'
      case 'submitting':
        return 'Отправляем данные…'
      case 'success':
        return 'Готово'
      case 'error':
        return 'Ошибка'
      default:
        return 'Готов к старту'
    }
  })()

  const primaryLabel = (() => {
    if (stage === 'capturing') return 'Сканирование…'
    if (stage === 'submitting') return 'Отправка…'
    if (stage === 'success') return 'Повторить'
    if (stage === 'error') return 'Повторить'
    return mode === 'register' ? 'Начать регистрацию' : 'Начать вход'
  })()

  const primaryAction = () => {
    if (stage === 'success' || stage === 'error') return reset()
    return startFlow()
  }

  const accent = mode === 'register' ? 'from-emerald-500' : 'from-indigo-500'

  return (
    <div className="min-h-screen bg-[#f6f2ea] text-slate-900">
      <div className="relative overflow-hidden">
        <div className="absolute -top-32 right-10 h-72 w-72 rounded-full bg-sky-200/70 blur-[120px]" />
        <div className="absolute -bottom-40 left-0 h-80 w-80 rounded-full bg-emerald-200/60 blur-[140px]" />
        <div className="mx-auto flex max-w-6xl flex-col gap-8 px-6 py-10 lg:py-14">
          <header className="flex flex-col gap-4">
            <p className="text-xs font-semibold uppercase tracking-[0.4em] text-slate-500">FaceID</p>
            <h1 className="text-3xl font-semibold lg:text-4xl">Вход и регистрация по лицу</h1>
            <p className="max-w-2xl text-sm text-slate-600">
              Одна кнопка запускает процесс: включаем камеру и автоматически собираем нужные кадры.
            </p>
          </header>

          <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="space-y-6">
              <section className="rounded-3xl border border-white/70 bg-white/80 p-6 shadow-sm backdrop-blur">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="text-lg font-semibold">Режим</h2>
                    <p className="text-xs text-slate-500">Выбери сценарий и заполни данные.</p>
                  </div>
                  <div className="inline-flex rounded-full bg-slate-100 p-1 text-xs font-semibold">
                    <button
                      className={`rounded-full px-4 py-2 transition ${
                        mode === 'register'
                          ? 'bg-slate-900 text-white'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                      onClick={() => setMode('register')}
                      disabled={mode === 'register'}
                    >
                      Регистрация
                    </button>
                    <button
                      className={`rounded-full px-4 py-2 transition ${
                        mode === 'login'
                          ? 'bg-slate-900 text-white'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                      onClick={() => setMode('login')}
                      disabled={mode === 'login'}
                    >
                      Вход
                    </button>
                  </div>
                </div>

                <div className="mt-5 grid gap-4">
                  <label className="grid gap-2 text-sm font-medium text-slate-600">
                    Эл. почта
                    <input
                      className="h-11 rounded-2xl border border-slate-200 bg-white px-4 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none"
                      value={email}
                      onChange={(event) => setEmail(event.target.value)}
                      placeholder="your@email.com"
                    />
                  </label>

                  {mode === 'register' && (
                    <label className="grid gap-2 text-sm font-medium text-slate-600">
                      Имя
                      <input
                        className="h-11 rounded-2xl border border-slate-200 bg-white px-4 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none"
                        value={name}
                        onChange={(event) => setName(event.target.value)}
                        placeholder="Ваше имя"
                      />
                    </label>
                  )}

                  {mode === 'login' && (
                    <label className="grid gap-2 text-sm font-medium text-slate-600">
                      Порог совпадения
                      <input
                        className="h-11 rounded-2xl border border-slate-200 bg-white px-4 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none"
                        value={threshold}
                        onChange={(event) => setThreshold(event.target.value)}
                        placeholder="0.55"
                      />
                    </label>
                  )}
                </div>
              </section>

              <section className="rounded-3xl border border-white/70 bg-white/80 p-6 shadow-sm backdrop-blur">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold">Сканирование</h2>
                    <p className="text-xs text-slate-500">Минимум семплов: {minSamples}</p>
                  </div>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                    {stageLabel}
                  </span>
                </div>

                <div className="mt-5 grid gap-4">
                  <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.25em] text-slate-400">
                      Автосканирование
                    </div>
                    <div className="mt-1 text-base font-semibold text-slate-900">
                      Держи лицо в центре круга — мы сами соберём нужные кадры
                    </div>
                  </div>
                  <button
                    className={`h-12 rounded-full bg-gradient-to-r ${accent} to-slate-900 px-6 text-sm font-semibold text-white shadow-md transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60`}
                    onClick={primaryAction}
                    disabled={cameraBusy || stage === 'capturing' || stage === 'submitting'}
                  >
                    {primaryLabel}
                  </button>

                  <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
                    <div className="flex items-center justify-between text-xs font-semibold text-slate-500">
                      <span>{cameraOn ? (cameraReady ? 'Камера готова' : 'Запуск камеры…') : 'Камера выключена'}</span>
                      <span>
                        {samples.length} / {maxSamples}
                      </span>
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-slate-200">
                      <div
                        className="h-2 rounded-full bg-emerald-500 transition-all"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <div className="mt-2 text-[11px] text-slate-500">
                      Автосбор работает сам, поворачивать голову не нужно.
                      Качество: {qualityMode === 'standard' ? 'обычное' : 'сниженное'}.
                    </div>
                  </div>

                  {(cameraError || captureError) && (
                    <div className="rounded-2xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
                      {cameraError || captureError}
                    </div>
                  )}
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-slate-500">
                  {cameraOn && (
                    <button
                      className="rounded-full border border-transparent px-2 py-2 font-semibold text-slate-400 hover:text-slate-600"
                      onClick={stopCamera}
                      disabled={stage === 'submitting'}
                    >
                      Остановить камеру
                    </button>
                  )}
                </div>
              </section>

              <section className="rounded-3xl border border-white/70 bg-white/80 p-6 shadow-sm backdrop-blur">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold">Результат</h2>
                  <span className="text-xs font-semibold text-slate-500">
                    {stage === 'success' ? 'Успешно' : stage === 'error' ? 'Ошибка' : 'Ожидание'}
                  </span>
                </div>

                <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-600">
                  {result ? (
                    <pre className="whitespace-pre-wrap font-mono text-[11px] leading-5 text-slate-900">
                      {result}
                    </pre>
                  ) : (
                    <p>
                      Результат появится автоматически после завершения сканирования.
                    </p>
                  )}
                </div>
              </section>
            </div>

            <div className="space-y-6">
              <section className="rounded-3xl border border-white/70 bg-white/80 p-6 shadow-sm backdrop-blur">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold">Превью камеры</h2>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                    {cameraOn ? (cameraReady ? 'Включена' : 'Запуск') : 'Выключена'}
                  </span>
                </div>
                <div className="relative mt-4 overflow-hidden rounded-2xl border border-slate-200 bg-slate-900">
                  <video
                    ref={videoRef}
                    className="aspect-video w-full object-cover"
                    style={{ transform: 'scaleX(-1)' }}
                    playsInline
                    muted
                  />
                  <canvas ref={canvasRef} className="hidden" />
                  <div
                    className="pointer-events-none absolute inset-0 flex items-center justify-center"
                    style={{ transform: 'scaleX(-1)' }}
                  >
                    <svg
                      className="h-[86%] w-[86%]"
                      viewBox={`0 0 ${RING_SIZE} ${RING_SIZE}`}
                      fill="none"
                    >
                      <circle
                        cx={RING_SIZE / 2}
                        cy={RING_SIZE / 2}
                        r={RING_RADIUS - 20}
                        stroke="rgba(255,255,255,0.18)"
                        strokeWidth={2}
                      />
                      {ringPaths.map((path, index) => {
                        const isFilled = index < filledSegments
                        const isActive = index === activeSegment
                        const stroke = isFilled
                          ? '#34d399'
                          : isActive
                            ? '#facc15'
                            : 'rgba(255,255,255,0.25)'
                        const strokeWidth = isFilled || isActive ? 7 : 4
                        return (
                          <path
                            key={`${path}-${index}`}
                            d={path}
                            stroke={stroke}
                            strokeWidth={strokeWidth}
                            strokeLinecap="round"
                          />
                        )
                      })}
                      <circle
                        cx={RING_SIZE / 2}
                        cy={RING_SIZE / 2}
                        r={52}
                        stroke="rgba(255,255,255,0.35)"
                        strokeWidth={2}
                      />
                      <circle cx={RING_SIZE / 2 - 18} cy={RING_SIZE / 2 - 6} r={3} fill="rgba(255,255,255,0.6)" />
                      <circle cx={RING_SIZE / 2 + 18} cy={RING_SIZE / 2 - 6} r={3} fill="rgba(255,255,255,0.6)" />
                      <path
                        d={`M ${RING_SIZE / 2 - 14} ${RING_SIZE / 2 + 12} Q ${RING_SIZE / 2} ${
                          RING_SIZE / 2 + 24
                        } ${RING_SIZE / 2 + 14} ${RING_SIZE / 2 + 12}`}
                        stroke="rgba(255,255,255,0.6)"
                        strokeWidth={3}
                        strokeLinecap="round"
                        fill="none"
                      />
                    </svg>
                  </div>
                </div>
              </section>

              <section className="rounded-3xl border border-white/70 bg-white/80 p-6 text-sm text-slate-600 shadow-sm backdrop-blur">
                <h3 className="text-base font-semibold text-slate-900">Подсказки</h3>
                <ul className="mt-3 list-inside list-disc space-y-2 text-sm">
                  <li>Держи лицо в центре круга и не двигайся резко.</li>
                  <li>Система сама собирает кадры, поворачивать голову не нужно.</li>
                  <li>Смотри в камеру, убери маску/очки для лучшего результата.</li>
                  <li>Если видишь ошибку 413 — кадр слишком большой, качество уже снижено.</li>
                  <li>Для входа нужно {LOGIN_SAMPLES} кадра, для регистрации — {REGISTER_SAMPLES}.</li>
                </ul>
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
