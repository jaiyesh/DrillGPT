import os
import requests
import logging
from colorama import Fore, Style

logger = logging.getLogger("DrillGPT")

class LLMService:
    """Service for interacting with OpenAI's API."""
    
    @staticmethod
    def chat_completion(messages, model="gpt-4o", temperature=0.7, max_tokens=2000):
        """Make a chat completion request to OpenAI API."""
        api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
        
        # In a real application, add proper error handling and retries
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            return response.json()
        except:
            # For demo purposes, return a mock response if API call fails
            return LLMService._generate_mock_response(messages)
    
    @staticmethod
    def _generate_mock_response(messages):
        """Generate a mock response for demonstration purposes."""
        # Extract the last user message
        last_message = next((m for m in reversed(messages) if m["role"] == "user"), None)
        content = last_message.get("content", "") if last_message else ""
        
        # Simple rule-based mock responses
        if "report" in content.lower():
            mock_response = "Here's a drilling performance report based on the data:\n\n"
            mock_response += "- Average ROP: 45.3 ft/hr\n"
            mock_response += "- Maximum ROP achieved: 87.2 ft/hr\n"
            mock_response += "- Current bit wear: Moderate (43%)\n"
            mock_response += "- Risk assessment: Low to moderate risk of bit failure\n\n"
            mock_response += "Recommended actions:\n"
            mock_response += "1. Continue monitoring vibration levels\n"
            mock_response += "2. Consider increasing WOB by 2-3 kips\n"
            mock_response += "3. Maintain current RPM values"
        elif "optimize" in content.lower():
            mock_response = "Based on current drilling conditions, I recommend the following parameter adjustments:\n\n"
            mock_response += "- Increase WOB from 18.5 to 22.0 kips\n"
            mock_response += "- Decrease RPM from 150 to 130\n"
            mock_response += "- Maintain flow rate at 650 GPM\n\n"
            mock_response += "These adjustments should improve ROP by approximately 15-20% while keeping vibration levels within acceptable ranges."
        else:
            mock_response = "I'm your drilling assistant. I can help with generating reports, optimizing drilling parameters, explaining current conditions, or providing recommendations for improving performance."
        
        return {"choices": [{"message": {"content": mock_response}}]} 