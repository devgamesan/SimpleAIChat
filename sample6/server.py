"""
WebSocketサーバー実装（音声認識機能付き）

このファイルは、WebSocketを通じてクライアントからPCM音声データを受け取り、
OpenAIのWhisperモデルを使用して音声をテキストに変換するサーバーを実装しています。

主な処理フロー:
1. クライアントからWebSocket接続を受け付ける
2. PCM音声データを受信
3. WAV形式に変換し、一時ファイルとして保存
4. Whisperモデルを使用して音声認識を実施
5. 認識結果をクライアントにJSON形式で返送
6. 一時ファイルを削除

使用技術:
- asyncio: 非同期処理
- websockets: WebSocket通信
- whisper: 音声認識モデル
- tempfile: 一時ファイル管理
- struct: バイナリデータ処理
"""
import asyncio
import websockets
import json
import io
import struct
import whisper  # OpenAIの音声認識モデルWhisperを使用
import tempfile  # 一時ファイルの作成と管理用
import os  # ファイル操作（削除など）用

# AudioContextの標準サンプルレート（48kHz）
# Whisperモデルは48kHzの入力音声を前提としているため、この値は変更しない
AUDIO_SAMPLE_RATE = 48000


def create_wav_header(pcm_data_length):
    """
    PCMデータをWAV形式に変換するためのヘッダを生成する関数

    WAVフォーマットはRIFFチャンクベースの構造を持ち、以下の要素で構成される:
    1. RIFFチャンク（ファイル全体のコンテナ）
    2. fmtサブチャンク（音声フォーマット情報）
    3. dataサブチャンク（実際の音声データ）

    引数:
        pcm_data_length (int): PCMデータのバイト長

    戻り値:
        bytes: WAVヘッダバイナリデータ
    """
    # RIFFチャンクの構築（12バイト）
    riff_chunk = (
        b"RIFF"  # チャンクID（4バイト）
        + struct.pack("<I", 36 + pcm_data_length)  # ファイルサイズ（4バイト, リトルエンディアン）
        + b"WAVE"  # フォーマットタイプ（4バイト）
    )

    # fmtサブチャンクの構築（24バイト）
    fmt_chunk = (
        b"fmt "  # サブチャンクID（4バイト）
        + struct.pack(
            "<IHHIIHH",
            16,  # サブチャンクサイズ（4バイト, PCMの場合は16）
            1,  # オーディオフォーマット（2バイト, 1=PCM）
            1,  # チャンネル数（2バイト, 1=モノラル）
            AUDIO_SAMPLE_RATE,  # サンプルレート（4バイト）
            AUDIO_SAMPLE_RATE * 2,  # バイトレート（4バイト, SampleRate * NumChannels * BitsPerSample/8）
            2,  # ブロックアラインメント（2バイト, NumChannels * BitsPerSample/8）
            16,  # ビット深度（2バイト, 16=16bit）
        )
    )

    # dataサブチャンクの構築（8バイト + 音声データ）
    data_chunk = (
        b"data"  # サブチャンクID（4バイト）
        + struct.pack("<I", pcm_data_length)  # データサイズ（4バイト）
    )

    return riff_chunk + fmt_chunk + data_chunk


async def send_json_response(websocket, data):
    """
    WebSocket経由でJSON形式のレスポンスを送信するヘルパー関数

    引数:
        websocket (WebSocket): 接続中のWebSocketオブジェクト
        data (dict): 送信するデータ（辞書形式）

    処理内容:
        1. 辞書をJSON文字列にシリアライズ
        2. UTF-8エンコードしてクライアントに送信
    """
    await websocket.send(json.dumps(data))


async def handle_connection(websocket, whisper_model):
    """
    クライアントからのWebSocket接続を処理するメイン関数

    処理フロー:
        1. クライアントからメッセージ（音声データ/テキスト）を受信
        2. バイナリデータ（PCM）の場合:
            a. WAVヘッダを生成
            b. 一時ファイルとして保存
            c. Whisperで音声認識
            d. 結果をクライアントに返信
            e. 一時ファイルを削除
        3. エラー発生時は適切なエラーレスポンスを返信

    引数:
        websocket (WebSocket): クライアントとの接続オブジェクト
        whisper_model (Whisper): ロード済みのWhisperモデルインスタンス
    """
    async for message in websocket:
        try:
            # バイナリデータ（PCM音声）の処理
            if isinstance(message, (bytes, bytearray)):
                print(f"受信PCMデータサイズ: {len(message)} bytes")

                # WAVヘッダ生成（PCMデータ長を元に）
                wav_header = create_wav_header(len(message))
                print(f"生成WAVヘッダ: {wav_header.hex()}")

                # WAVファイル構築（ヘッダ + PCMデータ）
                wav_data = wav_header + message

                # メモリ上でファイルオブジェクトとして扱う
                file_obj = io.BytesIO(wav_data)

                # 一時ファイル作成（自動削除のためwith文を使用）
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    tmpfile.write(file_obj.getvalue())
                    tmp_path = tmpfile.name

                try:
                    # Whisperによる音声認識実行
                    result = whisper_model.transcribe(
                        tmp_path,
                        language="ja",  # 日本語指定
                        temperature=0.2,  # 低いほど確定的な出力
                        beam_size=5,      # ビームサーチ幅
                        best_of=5,        # 生成候補数
                        no_speech_threshold=0.5,  # 無音判定閾値
                        compression_ratio_threshold=2.4,  # 圧縮率閾値
                    )

                    # 認識結果をクライアントに返信
                    await send_json_response(
                        websocket,
                        {"text": result["text"].strip()}  # 前後空白を除去
                    )
                finally:
                    # 一時ファイルを確実に削除
                    try:
                        os.remove(tmp_path)
                    except OSError as e:
                        print(f"一時ファイル削除エラー: {e}")

                continue

            # テキストメッセージの処理（将来の拡張用）
            data = json.loads(message)
            # 必要に応じて他のメッセージタイプを処理...

        except Exception as e:
            # エラーレスポンスの生成
            error_info = {
                "error": str(e),
                "type": type(e).__name__,
            }
            await send_json_response(websocket, error_info)
            print(f"処理エラー: {error_info}")


async def main():
    """
    WebSocketサーバーのメイン起動関数

    処理フロー:
        1. Whisperモデルをロード（初回はダウンロードが発生）
        2. WebSocketサーバーを起動
        3. クライアント接続を待機
    """
    print("Whisperモデルをロード中...")
    # モデル選択ガイド:
    # - CPU環境: "base" または "small" が現実的
    # - GPU環境: "medium" または "large" が可能
    whisper_model = whisper.load_model(
        "large",    # モデルサイズ
        device="cuda:1"  # 使用デバイス（筆者環境はマルチGPU環境のため、GPU指定）
    )
    print("Whisperモデルのロード完了")

    # WebSocketサーバー設定
    server = await websockets.serve(
        lambda ws: handle_connection(ws, whisper_model),  # 接続ハンドラ
        "localhost",  # ホストアドレス
        8000,         # ポート番号
        max_size=None,  # 受信データサイズ制限なし
    )

    print("WebSocketサーバー起動: ws://localhost:8000")
    await server.wait_closed()  # サーバー終了まで待機


if __name__ == "__main__":
    # 非同期サーバーを起動
    asyncio.run(main())