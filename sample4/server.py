"""
WebSocketサーバーによる音声データ処理の概要

このファイルは、WebSocket経由で音声データを受信し、OpenAIのWhisperモデルを使って音声認識を行うサーバーサイドのコードです。

処理の流れ:
1. WebSocket接続を確立し、クライアントからのメッセージを受信
2. 受信したメッセージが"audio_chunk"タイプであることを確認
3. Base64形式の音声データをデコードし、バイトオブジェクトに変換
4. OpenAIのAsyncOpenAIクライアントを使用して音声をテキストに変換
5. 変換結果をクライアントに送信
6. エラーが発生した場合はエラーメッセージを送信

注意点:
- OpenAI APIキーは環境変数から取得していますが、デバッグ用に直接記述しています
- 本番環境ではAPIキーの管理に.envファイルやクラウドコンソールを使用することを推奨
- WebSocketサーバーはlocalhost:8000で起動し、クライアントからの接続を待ち受けます
"""
# 必要なライブラリのインポート
import os
import asyncio
import websockets
from openai import AsyncOpenAI
import json
import io
import base64


# デバッグ用のAPIキー（開発中にのみ使用し、本番環境では削除してください）
# ここではデバッグ用に直接記述していますが、実際の運用では.envファイルやクラウドコンソールで管理してください
OPENAI_API_KEY = "your api key"

# 非同期用OpenAIクライアントの初期化
# これにより、WebSocket経由で受け取った音声データを非同期に処理できます
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def handle_connection(websocket):
    """WebSocket接続のハンドラ関数"""
    async for message in websocket:
        try:
            # 1. メッセージのJSON解析
            # クライアントから送信されたデータをJSON形式で読み込みます
            data = json.loads(message)
            
            # 2. メッセージタイプのチェック
            # "audio_chunk"タイプのメッセージのみ処理します
            if data.get("type") == "audio_chunk":
                audio_data = data["data"]
                
                # 3. 音声データの検証
                # 空のデータはエラーとして処理します
                if not audio_data:
                    raise ValueError("Empty audio data received")
                
                # 4. Base64データのデコード
                # クライアントから送信された音声データをバイトに変換します
                audio_bytes = base64.b64decode(audio_data)
                file_obj = io.BytesIO(audio_bytes)  # バイトデータをファイルオブジェクトに変換

                # 5. OpenAI API呼び出し
                # 音声データをテキストに変換します
                response = await client.audio.transcriptions.create(
                    model="whisper-1",  # 使用するモデル
                    file=("audio.webm", file_obj),  # ファイル名と形式を指定,
                    language="ja"  # 例: 日本語に指定
                )
                
                # 6. 結果の送信
                # テキスト結果をクライアントに送信します
                await websocket.send(json.dumps({"text": response.text}))
        
        except Exception as e:
            # エラーが発生した場合の処理
            # クライアントにエラーメッセージを送信します
            await websocket.send(json.dumps({"error": str(e)}))

async def main():
    """メイン関数：WebSocketサーバーの起動"""
    # 1. サーバーの起動
    # localhost:8000でWebSocketサーバーを起動します
    server = await websockets.serve(
        handle_connection,
        "localhost",
        8000,
        # 2. オリジン制限（コメントアウト時はすべてのオリジンを許可）
        # 本番環境では必ず指定してください：origin="http://localhost:8080"
    )

    print("WebSocket server listening on ws://localhost:8000")
    # 3. サーバーが終了するまで待機
    await server.wait_closed()

# 4. イベントループの開始
# 非同期処理を実行します
asyncio.run(main())