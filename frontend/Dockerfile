# ── Frontend Dockerfile ──
# Multi-stage: build → serve with nginx

# -- Build stage --
FROM node:20-alpine AS build

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

COPY . .
RUN npm run build

# -- Production stage --
FROM nginx:1.27-alpine

# Install openssl for self-signed cert
RUN apk add --no-cache openssl

# Generate self-signed certificate (valid 2 years)
RUN mkdir -p /etc/nginx/ssl && \
    openssl req -x509 -nodes -days 730 \
      -newkey rsa:2048 \
      -keyout /etc/nginx/ssl/selfsigned.key \
      -out /etc/nginx/ssl/selfsigned.crt \
      -subj "/CN=136.110.86.112" \
      -addext "subjectAltName=IP:136.110.86.112"

# Copy built frontend
COPY --from=build /app/dist /usr/share/nginx/html

# Nginx config: HTTP (80) redirects to HTTPS (443)
RUN printf 'client_max_body_size 2G;\n\
server {\n\
  listen 80;\n\
  return 301 https://$host$request_uri;\n\
}\n\
server {\n\
  listen 443 ssl;\n\
  ssl_certificate /etc/nginx/ssl/selfsigned.crt;\n\
  ssl_certificate_key /etc/nginx/ssl/selfsigned.key;\n\
  ssl_protocols TLSv1.2 TLSv1.3;\n\
  root /usr/share/nginx/html;\n\
  index index.html;\n\
  location /api/ {\n\
    proxy_pass http://host.docker.internal:8000;\n\
    proxy_http_version 1.1;\n\
    proxy_set_header Upgrade $http_upgrade;\n\
    proxy_set_header Connection "upgrade";\n\
    proxy_set_header Host $host;\n\
    proxy_read_timeout 600s;\n\
    proxy_send_timeout 600s;\n\
  }\n\
  location /pipeline/ {\n\
    proxy_pass http://host.docker.internal:8100;\n\
    proxy_http_version 1.1;\n\
    proxy_set_header Upgrade $http_upgrade;\n\
    proxy_set_header Connection "upgrade";\n\
    proxy_set_header Host $host;\n\
    proxy_read_timeout 600s;\n\
    proxy_send_timeout 600s;\n\
  }\n\
  location / {\n\
    try_files $uri $uri/ /index.html;\n\
  }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80 443
CMD ["nginx", "-g", "daemon off;"]
