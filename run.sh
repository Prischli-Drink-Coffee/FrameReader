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

if nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU обнаружен. Запуск с поддержкой GPU."
    GPU_FLAG="--gpus all"
else
    echo "ВНИМАНИЕ: NVIDIA GPU не обнаружен. Запуск без поддержки GPU."
    GPU_FLAG=""
fi

$RUN_PREFIX docker run $GPU_FLAG -it --rm \
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
    -w /workspace $IMAGE /bin/bash

{ set +x; } 2>/dev/null