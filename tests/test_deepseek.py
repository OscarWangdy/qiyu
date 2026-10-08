import base64
import unittest

from qiyu.agent import QiYuAgent
from qiyu.deepseek import (
    DeepSeekClient,
    DeepSeekConfig,
    board_to_fen,
    parse_data_url,
    parse_xiangqi_fen,
)
from qiyu.rules import RED, initial_board
from qiyu.server import Game


class FenTests(unittest.TestCase):
    def test_round_trip_initial_position(self):
        fen = board_to_fen(initial_board(), RED)
        board, side = parse_xiangqi_fen(fen)
        self.assertEqual(board, initial_board())
        self.assertEqual(side, RED)

    def test_common_knight_and_bishop_aliases_are_supported(self):
        board, side = parse_xiangqi_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w")
        self.assertEqual(board[1], "h")
        self.assertEqual(board[2], "e")
        self.assertEqual(side, RED)

    def test_position_requires_both_kings(self):
        with self.assertRaisesRegex(ValueError, "红帅和黑将"):
            parse_xiangqi_fen("9/9/9/9/9/9/9/9/9/4K4 w")

    def test_data_url_validation(self):
        raw = b"not-a-real-image-but-valid-base64"
        decoded, mime = parse_data_url("data:image/png;base64," + base64.b64encode(raw).decode())
        self.assertEqual(decoded, raw)
        self.assertEqual(mime, "image/png")


class DeepSeekClientTests(unittest.TestCase):
    def test_vision_response_is_normalized_and_validated(self):
        def transport(url, payload, headers, timeout):
            self.assertTrue(url.endswith("/chat/completions"))
            self.assertEqual(payload["model"], "deepseek-flash")
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertEqual(headers["Authorization"], "Bearer test-key-for-unit-tests")
            return {
                "choices": [{"message": {"content": (
                    '{"fen":"4k4/9/9/9/9/9/9/9/9/4K4 w",'
                    '"side_to_move":"red","confidence":0.87,'
                    '"orientation":"黑上红下","notes":[]}'
                )}}],
                "usage": {"total_tokens": 42},
            }

        client = DeepSeekClient(DeepSeekConfig(api_key="test-key-for-unit-tests"), transport)
        result = client.recognize_position(b"image", "image/png")
        self.assertEqual(result["side"], RED)
        self.assertEqual(result["confidence"], 0.87)
        self.assertEqual(len(result["board"]), 90)

    def test_vision_draft_does_not_invent_a_missing_king(self):
        def transport(url, payload, headers, timeout):
            return {
                "choices": [{"message": {"content": (
                    '{"fen":"4k4/5R3/9/9/9/9/9/9/9/9 w",'
                    '"pieces":[{"row":0,"col":4,"color":"black","type":"king","glyph":"将"},'
                    '{"row":1,"col":5,"color":"red","type":"rook","glyph":"车"}],'
                    '"side_to_move":"red","confidence":0.98,'
                    '"orientation":"黑上红下","notes":[]}'
                )}}],
                "usage": {},
            }

        client = DeepSeekClient(DeepSeekConfig(api_key="test-key-for-unit-tests"), transport)
        result = client.recognize_position(b"image", "image/png")
        self.assertEqual(result["board"].count("k"), 1)
        self.assertEqual(result["board"].count("K"), 0)
        self.assertLessEqual(result["confidence"], 0.65)
        self.assertTrue(any("红帅" in note for note in result["notes"]))
        with self.assertRaisesRegex(ValueError, "红帅和黑将"):
            parse_xiangqi_fen(result["fen"])

    def test_vision_draft_keeps_higher_confidence_on_coordinate_collision(self):
        def transport(url, payload, headers, timeout):
            return {
                "choices": [{"message": {"content": (
                    '{"pieces":['
                    '{"row":0,"col":4,"color":"black","type":"king","confidence":0.99},'
                    '{"row":1,"col":5,"color":"red","type":"rook","confidence":0.3},'
                    '{"row":1,"col":5,"color":"red","type":"horse","confidence":0.9}],'
                    '"side_to_move":"red","confidence":0.8,"orientation":"黑上红下","notes":[]}'
                )}}],
                "usage": {},
            }

        client = DeepSeekClient(DeepSeekConfig(api_key="test-key-for-unit-tests"), transport)
        result = client.recognize_position(b"image", "image/png")
        self.assertEqual(result["board"][14], "H")
        self.assertTrue(any("多枚候选" in note for note in result["notes"]))


class _FakeLanguageClient:
    def __init__(self):
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return {"content": "当前是红方行棋。", "model": "fake", "usage": {}}


class GameLanguageTests(unittest.TestCase):
    def test_chat_receives_verified_board_and_legal_moves(self):
        game = Game(QiYuAgent())
        client = _FakeLanguageClient()
        result = game.chat("该谁走？", client)
        self.assertEqual(result["reply"], "当前是红方行棋。")
        prompt = client.messages[-1]["content"]
        self.assertIn('"side_to_move": "red"', prompt)
        self.assertIn('"legal_moves":', prompt)

    def test_imported_position_replaces_board(self):
        game = Game(QiYuAgent())
        payload = game.load_position("4k4/9/9/9/9/9/9/9/9/4K4 w")
        self.assertEqual(payload["state"]["board"].count("K"), 1)
        self.assertEqual(payload["state"]["side"], RED)


if __name__ == "__main__":
    unittest.main()
