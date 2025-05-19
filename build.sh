TAG=
RUN_PREFIX=
BUILD_MODELS=()

declare -A FRAMEWORKS=(["OCR"]=1)
DEFAULT_FRAMEWORK="OCR"

SOURCE_DIR=$(dirname "$(readlink -f "$0")")
DOCKERFILE=${SOURCE_DIR}/docker/Dockerfile

BASE_IMAGE=nvcr.io/nvidia/tritonserver
BASE_IMAGE_TAG_OCR=24.08-py3

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
    --build-models)
        if [ "$2" ]; then
                BUILD_MODELS+=("$2")
                shift
            else
        BUILD_MODELS+=("all")
            fi
            ;;
        --base)
            if [ "$2" ]; then
                BASE_IMAGE=$2
                shift
            else
                error 'ERROR: "--base" requires an argument.'
            fi
            ;;
    --base-image-tag)
            if [ "$2" ]; then
                BASE_IMAGE_TAG=$2
                shift
            else
                error 'ERROR: "--base" requires an argument.'
            fi
            ;;
        --build-arg)
            if [ "$2" ]; then
                BUILD_ARGS+="--build-arg $2 "
                shift
            else
                error 'ERROR: "--build-arg" requires an argument.'
            fi
            ;;
        --tag)
            if [ "$2" ]; then
                TAG=$2
                shift
            else
                error 'ERROR: "--tag" requires an argument.'
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
    --no-cache)
        NO_CACHE=" --no-cache"
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
    if [ -z $BASE_IMAGE_TAG ]; then
        BASE_IMAGE_TAG=BASE_IMAGE_TAG_${FRAMEWORK}
        BASE_IMAGE_TAG=${!BASE_IMAGE_TAG}
    fi
    fi

    if [ -z "$TAG" ]; then
        TAG="tritonserver:r24.08"

    if [[ $FRAMEWORK == "OCR" ]]; then
        TAG+="-ocr"
    fi

    fi

}


show_image_options() {
    echo ""
    echo "Building Triton Inference Server Image: '${TAG}'"
    echo ""
    echo "   Base: '${BASE_IMAGE}'"
    echo "   Base_Image_Tag: '${BASE_IMAGE_TAG}'"
    echo "   Build Context: '${SOURCE_DIR}'"
    echo "   Build Options: '${BUILD_OPTIONS}'"
    echo "   Build Arguments: '${BUILD_ARGS}'"
    echo "   Framework: '${FRAMEWORK}'"
    echo ""
}

show_help() {
    echo "usage: build.sh"
    echo "  [--base base image]"
    echo "  [--base-imge-tag base image tag]"
    echo "  [--framework framework one of ${!FRAMEWORKS[@]}]"
    echo "  [--build-arg additional build args to pass to docker build]"
    echo "  [--tag tag for image]"
    echo "  [--dry-run print docker commands without running]"
    exit 0
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

get_options "$@"

BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG --build-arg FRAMEWORK=$FRAMEWORK "

if [ ! -z ${GITHUB_TOKEN} ]; then
    BUILD_ARGS+=" --build-arg GITHUB_TOKEN=${GITHUB_TOKEN} "
fi

if [ ! -z ${HF_TOKEN} ]; then
    BUILD_ARGS+=" --build-arg HF_TOKEN=${HF_TOKEN} "
fi

show_image_options

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

$RUN_PREFIX docker build -f $DOCKERFILE $BUILD_OPTIONS $BUILD_ARGS -t $TAG $SOURCE_DIR $NO_CACHE

{ set +x; } 2>/dev/null

if [[ $FRAMEWORK == OCR ]]; then
    if [ -z "$RUN_PREFIX" ]; then
    set -x
    fi
    
    $RUN_PREFIX mkdir -p $PWD/backend/python
    $RUN_PREFIX mkdir -p $PWD/backend/donut
    $RUN_PREFIX mkdir -p $PWD/backend/yolo

    $RUN_PREFIX docker run --rm -it -v ${SOURCE_DIR}:/workspace $TAG /bin/bash -c "cp -r /opt/tritonserver/backends/python/* /workspace/backend/python/"

    { set +x; } 2>/dev/null

    for model in "${BUILD_MODELS[@]}"
    do
    if [ -z "$RUN_PREFIX" ]; then
        set -x
    fi

    $RUN_PREFIX docker run --rm -it -v ${SOURCE_DIR}:/workspace $TAG /bin/bash -c "/workspace/scripts/build_models.sh"

    { set +x; } 2>/dev/null
    done
fi