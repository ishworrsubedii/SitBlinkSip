# Frontend (Next.js)
FROM node:20-alpine AS builder
WORKDIR /app

# NEXT_PUBLIC_* vars are inlined into the client bundle at build time, so they
# must be set here (not just at container run time). Defaults match the
# repo's .env.local — override via docker-compose build args if the backend
# is reachable at a different host/port for the browser.
ARG NEXT_PUBLIC_API_BASE_URL=localhost:8000
ARG NEXT_PUBLIC_WS_PREFIX=ws://
ARG NEXT_PUBLIC_HTTP_PREFIX=http://
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_WS_PREFIX=$NEXT_PUBLIC_WS_PREFIX
ENV NEXT_PUBLIC_HTTP_PREFIX=$NEXT_PUBLIC_HTTP_PREFIX

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---- runtime ----
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000
CMD ["npm", "start"]
