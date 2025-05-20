#!/bin/bash

echo "Остановка всех сервисов..."

if command -v ray >/dev/null 2>&1; then
    echo "Остановка Ray..."
    ray stop || true
fi

if command -v pm2 >/dev/null 2>&1; then
    echo "Остановка всех PM2 процессов..."
    pm2 stop all || true
    pm2 delete all || true
fi

echo "Остановка Prometheus и Grafana..."
pkill -f prometheus || true
pkill -f grafana || true

echo "Остановка Triton Server..."
pkill -f tritonserver || true

echo "Остановка связанных Docker контейнеров..."
running_containers=$(docker ps --format "{{.ID}} {{.Image}}" | grep -E "tritonserver|ray" | awk '{print $1}')
if [ ! -z "$running_containers" ]; then
    echo $running_containers | xargs docker stop
fi

echo "Очистка временных файлов..."
rm -rf /tmp/ray
rm -rf /tmp/rayserve-demo

echo "Все сервисы остановлены и очищены."