import unittest

from app.agents._chat_1_avoid_taker import AvoidIntakeAgent
from app.agents._0_contracts import AvoidIntakeInput


class FakeGemma:
    def __init__(self, text):
        self._text = text

    def generate_text(self, contents, max_output_tokens=500):
        return self._text


class AvoidIntakeAgentTest(unittest.TestCase):
    def test_extract_candidates_and_question(self):
        fake = FakeGemma('{"candidates":["egg","bacon","milk"],"confirm_question_ko":"너 egg, bacon, milk 피하는 거 맞지?"}')
        agent = AvoidIntakeAgent(fake)

        out = agent.run(AvoidIntakeInput(user_text="나는 egg, bacon, milk를 못먹어"))

        self.assertEqual(out.candidates, ["egg", "bacon", "milk"])
        self.assertIn("egg", out.confirm_question)
        self.assertIn("egg", out.confirm_question_ko)


if __name__ == "__main__":
    unittest.main()
