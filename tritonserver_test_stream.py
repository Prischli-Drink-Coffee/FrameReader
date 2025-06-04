import aiohtt
import json
import websockets
from pathlib import Path


class OCRStreamingTestClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")

    async def test_sse_streaming(self, image_paths: List[str], model: str = "yolo", chunk_size: int = 1):
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            
            for path in image_paths:
                with open(path, 'rb') as f:
                    data.add_field('images', f, filename=Path(path).name)
            
            async with session.post(
                f"{self.base_url}/stream/{model}",
                data=data,
                params={'chunk_size': chunk_size}
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            print(f"SSE Event: {data}")
                        except json.JSONDecodeError:
                            pass

    async def test_websocket(self, model: str = "yolo"):
        uri = f"{self.ws_url}/ws/inference/{model}"
        
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({"type": "ping"}))

            async for message in websocket:
                data = json.loads(message)
                print(f"WebSocket Message: {data}")
                
                if data.get("type") == "pong":
                    break

    async def test_health_check(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                data = await response.json()
                print(f"Health Check: {data}")
                return data


async def main():
    client = OCRStreamingTestClient()
    
    await client.test_health_check()
    await client.test_websocket("yolo")

    await client.test_sse_streaming(["./docs/test.jpg"], "yolo", 1)

if __name__ == "__main__":
    asyncio.run(main())