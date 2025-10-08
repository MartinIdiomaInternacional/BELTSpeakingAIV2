
#!/bin/sh
set -e
: "${API_URL:=http://localhost:8000}"
envsubst '${API_URL}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf
nginx -g 'daemon off;'
