
import unittest
from unittest.mock import MagicMock, patch
import json
from hydrocalib.agents.proposal import ProposalAgent
from hydrocalib.agents.types import RoundContext

class TestRetryLogic(unittest.TestCase):
    @patch('hydrocalib.agents.proposal.get_client')
    def test_proposal_retry_success(self, mock_get_client):
        # Setup mock client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock completions.create to fail twice then succeed
        # First call: Exception
        # Second call: Malformed JSON (extract_json_block raises)
        # Third call: Valid JSON
        
        valid_json = '{"candidates": [{"id": 1, "goal": "test", "updates": {}}]}'
        
        # Side effect for chat.completions.create
        # We need to simulate different outcomes
        
        # Mock response object
        success_response = MagicMock()
        success_response.choices[0].message.content = valid_json
        
        malformed_response = MagicMock()
        malformed_response.choices[0].message.content = "Not JSON"
        
        # We want: 
        # 1. Raise Exception -> caught, retry
        # 2. Return malformed -> extract raises -> caught, retry
        # 3. Return valid -> success
        
        # Note: My implementation catches `Exception`. `extract_json_block` raises `ValueError` or `JSONDecodeError` which inherit from Exception.
        
        mock_client.chat.completions.create.side_effect = [
            Exception("Simulated Network Error"),
            malformed_response, # This will cause extract_json_block to raise ValueError/JSONDecodeError if it returns text. Wait, extract_json_block parses the text.
            success_response
        ]
        
        agent = ProposalAgent(physics_information=False)
        
        # Mock RoundContext
        context = MagicMock(spec=RoundContext)
        context.round_index = 1
        context.display_params = {}
        context.params = {}
        context.aggregate_metrics = {}
        context.full_metrics = {}
        context.event_metrics = {}
        context.history_summary = ""
        context.description = ""
        context.physics_information = False
        context.physics_prompt = ""
        context.param_display_names = {}
        context.images = []
        
        print("\n--- Starting Retry Test (Expect 2 failures then success) ---")
        candidates = agent.propose(context, k=1)
        
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['id'], 1)
        print("--- Retry Test Passed ---")
        
        # Verify call count
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)

    @patch('hydrocalib.agents.proposal.get_client')
    def test_proposal_retry_failure(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_client.chat.completions.create.side_effect = Exception("Persistent Error")
        
        agent = ProposalAgent(physics_information=False)
        context = MagicMock(spec=RoundContext)
        context.round_index = 1
        context.images = []
        context.display_params = {} # Add missing attrs to avoid errors in build_prompt
        context.params = {}
        context.aggregate_metrics = {}
        context.full_metrics = {}
        context.event_metrics = {}
        context.history_summary = ""
        context.description = ""
        context.physics_information = False
        context.physics_prompt = ""
        context.param_display_names = {}

        print("\n--- Starting Max Retries Test (Expect Failure) ---")
        with self.assertRaises(RuntimeError) as cm:
            agent.propose(context, k=1)
        
        print(f"Caught expected error: {cm.exception}")
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)

if __name__ == '__main__':
    unittest.main()
