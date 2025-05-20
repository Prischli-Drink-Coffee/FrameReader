#!/bin/bash

TAG=
RUN_PREFIX=

declare -A FRAMEWORKS
FRAMEWORKS["OCR"]=1

DEFAULT_FRAMEWORK="OCR"

SOURCE_DIR=$(dirname "$(readlink -f "$0")")

IMAGE="tritonserver:r24.08"
IMAGE_TAG_OCR="-ocr"

get_options() {
    while :; do
        case $1 in
        -h | -\? | --help)
            show_help
            exit
            ;;
        --framework)
            if [ "$2" ]; then
                FRAMEWORK=$2
                shift
            else
                error 'ERROR: "--framework" requires an argument.'
            fi
            ;;
        --image)
            if [ "$2" ]; then
                IMAGE=$2
                shift
            else
                error 'ERROR: "--image" requires an argument.'
            fi
            ;;
        --dry-run)
            RUN_PREFIX="echo"
            echo ""
            echo "=============================="
            echo "DRY RUN: COMMANDS PRINTED ONLY"
            echo "=============================="
            echo ""
            ;;
        --)
            shift
            break
            ;;
        -?*)
            error 'ERROR: Unknown option: ' $1
            ;;
        ?*)
            error 'ERROR: Unknown option: ' $1
            ;;
        *)
            break
            ;;
        esac

        shift
    done

    if [ -z "$FRAMEWORK" ]; then
        FRAMEWORK=$DEFAULT_FRAMEWORK
    fi

    if [ ! -z "$FRAMEWORK" ]; then
        FRAMEWORK=${FRAMEWORK^^}
        if [[ ! -n "${FRAMEWORKS[$FRAMEWORK]}" ]]; then
            error 'ERROR: Unknown framework: ' $FRAMEWORK
        fi
    fi

    if [[ $FRAMEWORK == "OCR" ]]; then
        IMAGE="${IMAGE}${IMAGE_TAG_OCR}"
    fi
}

show_help() {
    echo "usage: run.sh"
    echo "  [--image image]"
    echo "  [--framework framework one of OCR]"
    echo "  [--dry-run print docker commands without running]"
    exit 0
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

get_options "$@"

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

if command -v pm2 >/dev/null 2>&1; then
    echo "Остановка всех PM2 процессов..."
    pm2 stop all || true
    pm2 delete all || true
fi

echo "Очистка временных директорий..."
sudo rm -rf /tmp/ray
sudo rm -rf /tmp/rayserve-demo
sudo mkdir -p /tmp/rayserve-demo

ip_address=$(hostname -I | awk '{print $1}')
echo "IP адрес: $ip_address"

echo "Запуск TritonServer с Ray Serve..."

if command -v lsof >/dev/null 2>&1; then
    if lsof -Pi :6666 -sTCP:LISTEN -t >/dev/null ; then
        echo "Порт 6666 занят. Остановка процессов..."
        ray stop || true
        lsof -t -i:6666 | xargs -r kill -9
        echo "Порт 6666 освобожден."
    fi
else
    echo "Команда lsof не найдена. Устанавливаем..."
    apt-get update && apt-get install -y lsof
fi

$RUN_PREFIX docker run --gpus all -d \
    --name tritonserver \
    --network host \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -eHF_TOKEN -eGITHUB_TOKEN \
    -eAWS_DEFAULT_REGION -eAWS_ACCESS_KEY_ID -eAWS_SECRET_ACCESS_KEY -eS3_BUCKET_URL \
    -v/tmp:/tmp \
    -v ${SOURCE_DIR}:/workspace \
    -v${SOURCE_DIR}/.cache/huggingface:/root/.cache/huggingface \
    -v${SOURCE_DIR}/backend:/opt/tritonserver/backends/ \
    -w /workspace $IMAGE /bin/bash -c "
    source /opt/.venv/bin/activate &&
    cd /workspace &&
    pm2 start 'serve run tritonserver_deployment:deployment' --name triton && # Ensure Serve listens on all interfaces
    echo 'Сервисы запущены. Используйте команду \"pm2 logs\" для просмотра логов.' &&
    echo 'Для подключения к контейнеру используйте \"docker exec -it tritonserver /bin/bash\".' &&
    tail -f /dev/null
    "

{ set +x; } 2>/dev/null
