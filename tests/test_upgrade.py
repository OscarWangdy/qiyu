import unittest
from unittest.mock import patch
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import torch

from qiyu.agent import Decision
from qiyu.model import joint_move_scores
from qiyu.retrain import PositionDataset
from qiyu.rules import BLACK, RED, Move, initial_board, legal_moves, move_notation, parse_move
from qiyu.server import Game, Handler
import qiyu.server as server_module


class FakeAgent:
    ready = True
    device = torch.device("cpu")

    def decide(self, board, side):
        move = legal_moves(board, side)[0]
        return Decision(
            move=move,
            confidence=0.75,
            candidates=[{"move": move.key, "notation": move_notation(board, move), "score": 0.75}],
            attention=[1 / 90] * 90,
            explanation="这是一步通过规则校验的演示着。",
            engine="test-agent",
        )


class ChineseNotationTests(unittest.TestCase):
    def test_red_central_cannon_uses_xiangqi_notation(self):
        board = initial_board()
        source, destination = parse_move("h7-e7")
        self.assertEqual(move_notation(board, Move(source, destination)), "炮二平五")

    def test_black_central_cannon_uses_own_file_numbers(self):
        board = initial_board()
        source, destination = parse_move("b2-e2")
        self.assertEqual(move_notation(board, Move(source, destination)), "炮二平五")

    def test_red_horse_advance_names_destination_file(self):
        board = initial_board()
        source, destination = parse_move("b9-c7")
        self.assertEqual(move_notation(board, Move(source, destination)), "马八进七")


class TrainingConsistencyTests(unittest.TestCase):
    def test_inference_scores_match_masked_training_objective(self):
        source = torch.tensor([[2.0, 0.0]])
        destination = torch.tensor([[[1.0, -3.0], [0.0, 4.0]]])
        scores = joint_move_scores(source, destination)
        self.assertEqual(scores[0, 0, 0].item(), 3.0)
        self.assertEqual(scores[0, 1, 1].item(), 4.0)

    def test_color_rotation_preserves_legal_move_and_value(self):
        board = "".join(initial_board())
        source, destination = parse_move("h7-e7")
        record = {"board": board, "side": RED, "source": source, "destination": destination, "result": "1-0"}
        dataset = PositionDataset([record], mirror_augmentation=False, color_rotation=True)
        with patch("torch.rand", return_value=torch.tensor(0.0)):
            _, rotated_source, rotated_destination, value, rotated_board, rotated_side = dataset[0]
        self.assertEqual(rotated_side, BLACK)
        self.assertEqual(value, 1.0)
        self.assertIn(
            (rotated_source, rotated_destination),
            {(move.src, move.dst) for move in legal_moves(list(rotated_board), rotated_side)},
        )


class GameUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.game = Game(FakeAgent())

    def test_player_can_choose_black_and_ai_opens(self):
        payload = self.game.reset_response(BLACK)
        self.assertEqual(payload["state"]["human_side"], BLACK)
        self.assertEqual(payload["state"]["side"], BLACK)
        self.assertIsNotNone(payload["transition"])
        self.assertEqual(len(payload["state"]["history"]), 1)

    def test_player_can_choose_red(self):
        payload = self.game.reset_response(RED)
        self.assertEqual(payload["state"]["human_side"], RED)
        self.assertEqual(payload["state"]["side"], RED)
        self.assertIsNone(payload["transition"])

    def test_chat_hint_updates_explainability_without_moving(self):
        before = list(self.game.board)
        payload = self.game.chat("这步怎么走？")
        self.assertEqual(before, payload["state"]["board"])
        self.assertIn("建议走", payload["reply"])
        self.assertTrue(payload["state"]["candidates"])
        self.assertIn("通过规则校验", payload["state"]["explanation"])
        self.assertEqual(len(payload["state"]["chat"]), 3)

    def test_chat_rejects_empty_question(self):
        with self.assertRaisesRegex(ValueError, "请先输入问题"):
            self.game.chat("   ")

    def test_chat_explains_piece_rule(self):
        payload = self.game.chat("马怎么走？")
        self.assertIn("马走‘日’字", payload["reply"])


class HttpUpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server_module.AGENT = FakeAgent()
        server_module.GAMES.clear()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            method="GET" if payload is None else "POST",
            headers={"Content-Type": "application/json", "X-Qiyu-Session": "upgrade-test-session-01"},
        )
        with urlopen(request, timeout=3) as response:
            return json.load(response)

    def test_chat_endpoint(self):
        result = self.request("/api/chat", {"question": "怎么走？"})
        self.assertIn("建议走", result["reply"])
        self.assertEqual(result["state"]["human_side"], RED)

    def test_health_endpoint_reports_model(self):
        result = self.request("/api/health")
        self.assertTrue(result["ok"])
        self.assertTrue(result["model_ready"])
        self.assertEqual(result["model"], "qiyu-v3")

    def test_reset_endpoint_accepts_black(self):
        result = self.request("/api/reset", {"human_side": BLACK})
        self.assertEqual(result["state"]["human_side"], BLACK)
        self.assertEqual(result["state"]["side"], BLACK)
        self.assertIsNotNone(result["transition"])


if __name__ == "__main__":
    unittest.main()
