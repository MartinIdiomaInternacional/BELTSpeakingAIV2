FROM node:20-alpine as build
WORKDIR /app

# 1) Copy package manifests first (best caching + ensures deps exist)
COPY frontend/package*.json ./
RUN npm install

# 2) Copy the rest of the frontend and build
COPY frontend/ ./
RUN npm run build


FROM nginx:alpine
RUN apk add --no-cache bash gettext

COPY --from=build /app/dist /usr/share/nginx/html
COPY frontend/nginx/default.conf.template /etc/nginx/conf.d/default.conf.template
COPY frontend/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV API_URL=""
EXPOSE 80
CMD ["/entrypoint.sh"]

