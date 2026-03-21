import unittest

from app.agents.extract_agent import MenuExtractAgent


class FakeGemma:
    def __init__(self, text):
        self._responses = list(text) if isinstance(text, list) else [text]
        self.calls = []

    def generate_text(self, contents, max_output_tokens=900):
        self.calls.append((contents, max_output_tokens))
        if not self._responses:
            raise RuntimeError("No fake response prepared")
        return self._responses.pop(0)


class ExtractAgentTest(unittest.TestCase):
    def test_extract_items_from_json_response(self):
        fake = FakeGemma('{"items": ["Kimchi Fried Rice", "  Tuna Mayo  ", "Kimchi Fried Rice"]}')
        agent = MenuExtractAgent(fake)

        result = agent.run(image_part=object())

        self.assertEqual(result.items, ["Kimchi Fried Rice", "Tuna Mayo"])
        self.assertEqual(len(fake.calls), 1)

    def test_retries_once_when_first_response_is_not_json(self):
        fake = FakeGemma(
            [
                "not json",
                '{"items": ["Menu A", "Menu B"]}',
            ]
        )
        agent = MenuExtractAgent(fake)

        result = agent.run(image_part=object())

        self.assertEqual(result.items, ["Menu A", "Menu B"])
        self.assertEqual(len(fake.calls), 2)


if __name__ == "__main__":
    unittest.main()
