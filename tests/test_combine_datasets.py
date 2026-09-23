from scripts.combine_datasets import _messages, _parse_dataset


def test_normalizes_instruction_dataset():
    result = _messages({"instruction": "Say hi", "input": "to the user", "output": "Hi!"})
    assert result == [
        {"role": "user", "content": "Say hi\n\nto the user"},
        {"role": "assistant", "content": "Hi!"},
    ]


def test_normalizes_chat_dataset():
    result = _messages({"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]})
    assert result[-1]["content"] == "A"


def test_dataset_config_syntax():
    assert _parse_dataset("org/repo:config") == ("org/repo", "config")
    assert _parse_dataset("org/repo") == ("org/repo", None)
