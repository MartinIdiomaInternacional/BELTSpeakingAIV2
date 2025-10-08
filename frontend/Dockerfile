
FROM node:20-alpine as build
WORKDIR /app
COPY frontend/package.json frontend/vite.config.js /app/
RUN npm install
COPY frontend /app
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
