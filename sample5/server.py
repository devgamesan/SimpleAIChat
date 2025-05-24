"""
WebSocketサーバー + Whisper API（Int16 PCM → WAV ヘッダー付与版）
"""

import asyncio
import websockets
from openai import AsyncOpenAI
import json
import io
import struct

# OpenAI APIキー（環境変数管理推奨）
OPENAI_API_KEY = "your api key"
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# AudioContext の標準サンプルレート（48kHz）
AUDIO_SAMPLE_RATE = 48000

def create_wav_header(pcm_data_length):
    """WAVヘッダを作成するヘルパー関数"""
    return (
        b"RIFF"
        + struct.pack("<I", 36 + pcm_data_length)  # ChunkSize
        + b"WAVEfmt "  # Subchunk1ID
        + struct.pack(
            "<IHHIIHH",
            16,
            1,
            1,
            AUDIO_SAMPLE_RATE,
            AUDIO_SAMPLE_RATE * 2,
            2,
            16,
        )
        + b"data"
        + struct.pack("<I", pcm_data_length)  # Subchunk2Size
    )

async def send_json_response(websocket, data):
    """JSON形式の応答を送信するヘルパー関数"""
    await websocket.send(json.dumps(data))

async def handle_connection(websocket):
    async for message in websocket:
        try:
            # バイナリなら生PCM(Int16)として WAV ヘッダーを作成
            if isinstance(message, (bytes, bytearray)):
                print(f"Received PCM data size: {len(message)} bytes")
                
                wav_header = create_wav_header(len(message))
                wav = wav_header + message
                
                print(f"WAV header: {wav_header.hex()}")
                file_obj = io.BytesIO(wav)

                # Whisper API呼び出し
                resp = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.wav", file_obj, "audio/wav"),
                    language="ja"  # 例: 日本語に指定
                )
                await send_json_response(websocket, {"text": resp.text})
                continue

            # ここ以下は JSON 文字列を受け取る従来処理
            data = json.loads(message)
            # （…もし必要なら他のtype処理…）

        except Exception as e:
            # エラー時は確実にJSON形式で送信
            await send_json_response(websocket, {"error": str(e)})

async def main():
    server = await websockets.serve(handle_connection, "localhost", 8000, max_size=None)
    print("WebSocket server listening on ws://localhost:8000")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())