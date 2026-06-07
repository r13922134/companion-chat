import unittest
from unittest.mock import patch

from app import server_feedback, server_realtime


class RealtimePromptPolicyTests(unittest.TestCase):
    def test_tool_descriptions_own_usage_timing(self):
        for server in (server_realtime, server_feedback):
            with self.subTest(server=server.__name__):
                medical_description = server.build_medical_qa_tool()["description"]
                web_description = server.build_search_web_tool(True)["description"]
                web_only_description = server.build_search_web_tool(False)["description"]

                self.assertIn("醫療衛教", medical_description)
                self.assertIn("即時政策", medical_description)
                self.assertIn("即時、近期", web_description)
                self.assertIn("一般醫療衛教優先使用 medical_qa", web_description)
                self.assertIn("一般聊天與情緒支持請直接回答", web_description)
                self.assertNotIn("medical_qa", web_only_description)

    def test_default_prompt_does_not_repeat_tool_policies(self):
        for server in (server_realtime, server_feedback):
            with self.subTest(server=server.__name__):
                instructions = server.build_default_assistant_instructions(
                    server.REALTIME_MODE_LISTENING
                )
                session_instructions = server.build_realtime_client_session_config(
                    server.REALTIME_MODE_LISTENING
                )["instructions"]

                self.assertNotIn("medical_qa", instructions)
                self.assertNotIn("search_web", instructions)
                self.assertNotIn("# Medical QA Policy", instructions)
                self.assertNotIn("# Web Search Policy", instructions)
                self.assertEqual(session_instructions, instructions)

    def test_tool_list_only_references_available_tools(self):
        for server in (server_realtime, server_feedback):
            with self.subTest(server=server.__name__):
                with patch.object(server, "is_medical_qa_enabled", return_value=False):
                    tools = server.build_realtime_tools()

                self.assertEqual([tool["name"] for tool in tools], ["search_web"])
                self.assertNotIn("medical_qa", tools[0]["description"])

    def test_tool_result_prompts_keep_answer_policies(self):
        for server in (server_realtime, server_feedback):
            with self.subTest(server=server.__name__):
                medical_instructions = server.build_medical_qa_assistant_instructions(
                    server.REALTIME_MODE_LISTENING
                )
                web_instructions = server.build_web_search_assistant_instructions(
                    server.REALTIME_MODE_LISTENING
                )

                self.assertIn("只根據剛剛 medical_qa 的工具輸出", medical_instructions)
                self.assertIn("不要加入 QA 沒有支撐的醫療細節", medical_instructions)
                self.assertIn("根據剛剛 search_web 的工具輸出", web_instructions)
                self.assertIn("若工具輸出不足以回答", web_instructions)


if __name__ == "__main__":
    unittest.main()
