
source $(dirname "$0")/../start_ray.sh

cd $(dirname "$0")/..

echo "Starting Ray Serve with Triton Server..."
python -m ray.serve run tritonserver_deployment:deployment

echo "Ray Serve with Triton Server is running. You can access the API at http://localhost:8000/"