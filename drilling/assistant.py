import pandas as pd
import logging
from colorama import Fore, Style

logger = logging.getLogger("DrillGPT")

class DrillingAssistant:
    """LLM-enhanced drilling assistant."""
    
    def __init__(self, llm_service, drilling_models, current_state=None):
        """Initialize the drilling assistant."""
        self.llm_service = llm_service
        self.models = drilling_models
        self.current_state = current_state
        self.conversation_history = []
    
    def add_message(self, role, content):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def format_current_state(self):
        """Format the current drilling state as a text string."""
        if self.current_state is None:
            return "No current drilling state available."
        
        state = self.current_state
        
        result = "--- CURRENT DRILLING STATE ---\n"
        result += f"Depth: {state.get('depth', 'N/A'):.1f} ft\n"
        result += f"Formation: {state.get('formation', 'N/A')}\n"
        result += f"Bit Type: {state.get('bit_type', 'N/A')}\n"
        result += f"Bit Hours: {state.get('bit_hours', 'N/A'):.1f} hrs\n\n"
        
        result += "--- CURRENT PARAMETERS ---\n"
        result += f"Weight on Bit (WOB): {state.get('weight_on_bit', 'N/A'):.1f} kips\n"
        result += f"Rotary Speed (RPM): {state.get('rotary_speed', 'N/A'):.1f} RPM\n"
        result += f"Flow Rate: {state.get('flow_rate', 'N/A'):.1f} GPM\n\n"
        
        result += "--- KEY INDICATORS ---\n"
        result += f"ROP: {state.get('rop', 'N/A'):.1f} ft/hr\n"
        result += f"Vibration: {state.get('vibration', 'N/A'):.2f}\n"
        
        # Add ML predictions if available
        if self.models.rop_model is not None and self.models.bit_failure_model is not None:
            df = pd.DataFrame([state])
            predicted_rop = self.models.predict_rop(df)[0]
            bit_failure_prob = self.models.predict_bit_failure_prob(df)[0]
            
            result += f"Predicted ROP: {predicted_rop:.1f} ft/hr\n"
            result += f"Bit Failure Probability: {bit_failure_prob:.2f}\n"
        
        return result
    
    def query_llm(self, query):
        logger.info(Fore.LIGHTYELLOW_EX + f"Querying LLM: {query[:60]}..." + Style.RESET_ALL)
        # Prepare system prompt
        system_prompt = """
        You are DrillGPT, an AI drilling engineering assistant with expertise in drilling optimization, 
        well operations, and petroleum engineering. You help drilling engineers interpret data, optimize 
        parameters, troubleshoot issues, and make data-driven decisions. 
        
        You should always:
        1. Analyze the drilling data provided before responding
        2. Consider both data-driven insights and engineering principles
        3. Provide specific, actionable recommendations when possible
        4. Explain your reasoning clearly
        """
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Current drilling context:\n\n{self.format_current_state()}"}
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add the new query
        messages.append({"role": "user", "content": query})
        
        # Get response from LLM
        response = self.llm_service.chat_completion(messages)
        response_content = response["choices"][0]["message"]["content"]
        
        # Add the query and response to conversation history
        self.add_message("user", query)
        self.add_message("assistant", response_content)
        
        return response_content
    
    def generate_report(self):
        logger.info(Fore.LIGHTYELLOW_EX + "Generating drilling report using LLM..." + Style.RESET_ALL)
        return self.query_llm("Generate a comprehensive daily drilling report based on the current data.")
    
    def get_optimization_recommendation(self, optimal_params):
        logger.info(Fore.LIGHTYELLOW_EX + "Getting optimization recommendation from LLM..." + Style.RESET_ALL)
        current = self.current_state
        prompt = f"""
        The ML model recommends the following parameter changes:
        - WOB: {current.get('weight_on_bit', 'N/A'):.1f} → {optimal_params.get('weight_on_bit', 'N/A'):.1f} kips
        - RPM: {current.get('rotary_speed', 'N/A'):.1f} → {optimal_params.get('rotary_speed', 'N/A'):.1f} RPM
        - Flow Rate: {current.get('flow_rate', 'N/A'):.1f} → {optimal_params.get('flow_rate', 'N/A'):.1f} GPM
        
        Predicted ROP improvement: {optimal_params.get('predicted_rop', 0) - current.get('rop', 0):.1f} ft/hr
        
        Please explain why these changes would help and any precautions that should be taken.
        """
        return self.query_llm(prompt) 