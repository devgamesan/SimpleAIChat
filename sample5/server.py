"""
WebSocketサーバー + Whisper API（Int16 PCM → WAV ヘッダー付与版）

このファイルでは、WebSocketを通じたリアルタイム音声認識サーバーを実装しています。
主な処理フローは以下の通りです：
1. クライアントから受信したInt16 PCM形式の音声データを受け取ります
2. 受信データにWAVファイルヘッダーを付与して、OpenAI Whisper APIに適した形式に変換します
3. Whisper APIを使用して音声認識を行い、テキスト結果をクライアントに返します

特徴：
- 非同期処理（async/await）を使用した高パフォーマンスなWebSocketサーバー
- PCMデータをWAV形式に変換するためのヘッダ生成処理を含む
- OpenAI APIキーを環境変数で管理するよう推奨（本コードではサンプルとして直接記載）
- エラーハンドリング付きのJSON応答処理を実装
"""
import asyncio
import websockets
from openai import AsyncOpenAI
import json
import io
import struct

# OpenAI APIキー（実際には環境変数で管理することを推奨）
# 本コードではセキュリティのため、"your api key"を実際のキーに置き換えてください
OPENAI_API_KEY = "your api key"
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# AudioContextの標準サンプルレート（48kHz）
# Whisper APIはこのサンプルレートを前提としている
AUDIO_SAMPLE_RATE = 48000

def create_wav_header(pcm_data_length):
    """
    WAVヘッダを作成するヘルパー関数
    PCMデータをWAV形式に変換するためのヘッダを生成
    """
    # RIFF チャンクの構造
    # RIFF + ChunkSize + WAVE で構成
    # ChunkSize = 36（ヘッダサイズ） + pcm_data_length（データサイズ）
    return (
        b"RIFF"  # RIFFチャンクID
        + struct.pack("<I", 36 + pcm_data_length)  # ChunkSize（リトルエンディアン）
        + b"WAVEfmt "  # fmtサブチャンクID
    
        # fmtサブチャンクの構造
        # 16（サブチャンクサイズ）+ 1（フォーマットID）+ 1（チャネル数）+
        # サンプルレート + バイトレート + ビットレート + 2（フォーマットID）+ 16（ビット深度）
        + struct.pack(
            "<IHHIIHH",
            16,  # Subchunk1Size
            1,   # AudioFormat (PCM)
            1,   # NumChannels (モノラル)
            AUDIO_SAMPLE_RATE,  # SampleRate
            AUDIO_SAMPLE_RATE * 2,  # ByteRate = SampleRate * NumChannels * BitsPerSample/8
            2,   # BlockAlign = NumChannels * BitsPerSample/8
            16,  # BitsPerSample
        )
        
        # dataサブチャンクの構造
        + b"data"  # Subchunk2ID
        + struct.pack("<I", pcm_data_length)  # Subchunk2Size
    )

async def send_json_response(websocket, data):
    """
    JSON形式の応答を送信するヘルパー関数
    クライアントに構造化されたデータを送信するための共通処理
    """
    await websocket.send(json.dumps(data))

async def handle_connection(websocket):
    """
    WebSocket接続の処理
    クライアントから送信されたメッセージを処理し、音声認識結果を返す
    """
    async for message in websocket:
        try:
            # メッセージがバイナリデータ（PCM）の場合の処理
            if isinstance(message, (bytes, bytearray)):
                # 受信データのサイズを確認
                print(f"Received PCM data size: {len(message)} bytes")
                
                # WAVヘッダの作成（PCMデータの長さを元に）
                wav_header = create_wav_header(len(message))
                # WAVファイル作成（ヘッダ + PCMデータ）
                wav = wav_header + message
                
                # ヘッダの内容を確認（デバッグ用）
                print(f"WAV header: {wav_header.hex()}")
                
                # BytesIOを使ってメモリ上のファイルオブジェクトを作成
                file_obj = io.BytesIO(wav)

                # Whisper API呼び出し
                # Whisper APIはWAVファイルを入力として認識処理を行う
                resp = await client.audio.transcriptions.create(
                    model="whisper-1",  # 使用するモデル
                    file=("audio.wav", file_obj, "audio/wav"),  # 入力ファイル
                    language="ja"  # 認識対象言語（例: 日本語）
                )
                # 認識結果をJSON形式でクライアントに送信
                await send_json_response(websocket, {"text": resp.text})
                continue

            # ここ以下はJSON文字列を受け取る従来処理（拡張性のため）
            # 他のメッセージタイプ（テキストなど）の処理が必要な場合に備えておく
            data = json.loads(message)
            # （…もし必要なら他のtype処理…）

        except Exception as e:
            # エラー発生時の処理
            # すべてのエラーをJSON形式でクライアントに通知
            await send_json_response(websocket, {"error": str(e)})

async def main():
    """
    サーバーの起動処理
    WebSocketサーバーをローカルホスト8000ポートで起動
    """
    # WebSocketsサーバーの起動
    # max_size=None: 大きなファイルの受信を許可
    server = await websockets.serve(handle_connection, "localhost", 8000, max_size=None)
    print("WebSocket server listening on ws://localhost:8000")
    # サーバー終了まで待機
    await server.wait_closed()

if __name__ == "__main__":
    # サーバーの実行
    asyncio.run(main())