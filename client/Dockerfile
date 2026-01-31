# syntax=docker/dockerfile:1.6

# -------- deps --------
FROM node:24-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json .npmrc* ./
RUN --mount=type=cache,target=/root/.npm npm ci

# -------- build (prod) --------
FROM node:24-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Важно: Vite читает только VITE_* на этапе билда
ARG VITE_API_BASE=/course_api
ENV VITE_API_BASE=${VITE_API_BASE}

RUN npm run build

# -------- dev (vite) --------
FROM node:24-alpine AS dev
WORKDIR /app
COPY package.json package-lock.json .npmrc* ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY . .
ENV HOST=0.0.0.0
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# -------- nginx runner (prod) --------
FROM nginx:1.27-alpine AS runner
# наш vhost
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
# статика фронта
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null 2>&1 || exit 1
CMD ["nginx", "-g", "daemon off;"]