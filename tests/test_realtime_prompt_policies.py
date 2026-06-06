import unittest
from unittest.mock import patch

from app import server_realtime


class RealtimePromptPolicyTests(unittest.TestCase):
    def test_tool_descriptions_own_usage_timing(self):
        medical_description = server_realtime.build_medical_qa_tool()["description"]
        web_description = server_realtime.build_search_web_tool()["description"]

        self.assertIn("醫療衛教", medical_description)
        self.assertIn("即時政策", medical_description)
        self.assertIn("即時、近期", web_description)
        self.assertIn("一般聊天與情緒支持請直接回答", web_description)

    @patch.object(server_realtime, "is_medical_qa_enabled", return_value=True)
    def test_prompt_keeps_cross_tool_priority_without_repeating_usage_lists(self, _):
        instructions = server_realtime.build_search_enabled_assistant_instructions(
            server_realtime.REALTIME_MODE_LISTENING
        )

        self.assertIn("醫療衛教優先使用 medical_qa", instructions)
        self.assertIn("只根據 QA 依據回答", instructions)
        self.assertIn("搜尋前把搜尋字串寫成一句清楚的查詢", instructions)
        self.assertNotIn("癌症治療、營養、放療、化療、副作用", instructions)
        self.assertNotIn("一般聊天、情緒支持、延續既有對話", instructions)
        self.assertNotIn("只有使用者詢問即時、近期、外部、指定來源", instructions)


if __name__ == "__main__":
    unittest.main()
