import argparse
import sys
import time

import tritonserver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument(
        "--model-repository", type=str, default="/workspace/models"
    )
    parser.add_argument("--timeout", type=int, default=60 * 20)

    args = parser.parse_args()

    server = tritonserver.Server(
        model_repository=args.model_repository,
        model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
    )

    server.start(wait_until_ready=True)
    models = server.models()

    if args.model == "all":
        models = models.keys()
    else:
        args.model = (args.model, -1)
        if not args.model in models:
            print(f"Model: {args.model} not known")
            sys.exit(1)
        models = [args.model]

    for model in models:
        if model[1] != -1:
            continue
        print(f"Loading Model: {model}")
        model = server.load(model[0])
        start = time.time()
        while not model.ready() and ((time.time() - start) <= args.timeout):
            time.sleep(10)

        if model.ready():
            print(f"Model: {model} Loaded")
        else:
            print(f"Error loading: {model}")
            sys.exit(1)

        server.unload(model, wait_until_unloaded=True)

    server.stop()
