from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_13b import DEFAULT_FILE, DEFAULT_REPO, parser


def test_default_13b_model_is_configured():
    assert "13b" in DEFAULT_REPO.lower()
    assert "13b" in DEFAULT_FILE.lower()


def test_cli_parser_supports_chat_and_server():
    args = parser().parse_args(["chat", "--prompt", "hello"])
    assert args.mode == "chat"
    assert args.prompt == "hello"

    args = parser().parse_args(["server", "--port", "9090"])
    assert args.mode == "server"
    assert args.port == 9090
