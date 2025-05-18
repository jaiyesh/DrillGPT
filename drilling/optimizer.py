import numpy as np
import pandas as pd
import logging
from colorama import Fore, Style

logger = logging.getLogger("DrillGPT")

class ParameterOptimizer:
    """Optimizer for drilling parameters."""
    
    def __init__(self, models):
        """Initialize the optimizer."""
        self.models = models
    
    def optimize_parameters(self, current_state):
        logger.info(Fore.LIGHTMAGENTA_EX + "Optimizing drilling parameters for current state..." + Style.RESET_ALL)
        logger.info(Fore.LIGHTMAGENTA_EX + f"Current WOB: {current_state.get('weight_on_bit', 'N/A')}, RPM: {current_state.get('rotary_speed', 'N/A')}, Flow Rate: {current_state.get('flow_rate', 'N/A')}" + Style.RESET_ALL)
        # Define parameter ranges to explore
        param_ranges = {
            'weight_on_bit': np.linspace(5, 30, 6),  # kips
            'rotary_speed': np.linspace(40, 200, 6),  # RPM
            'flow_rate': np.linspace(300, 1000, 6),  # GPM
        }
        
        # Test all parameter combinations
        results = []
        
        for wob in param_ranges['weight_on_bit']:
            for rpm in param_ranges['rotary_speed']:
                for flow in param_ranges['flow_rate']:
                    # Create test dataframe with current parameters
                    test_state = current_state.copy()
                    test_state['weight_on_bit'] = wob
                    test_state['rotary_speed'] = rpm
                    test_state['flow_rate'] = flow
                    
                    # Create a DataFrame for prediction
                    df_test = pd.DataFrame([test_state])
                    
                    # Use the models to predict ROP and bit failure probability
                    predicted_rop = self.models.predict_rop(df_test)[0]
                    bit_failure_prob = self.models.predict_bit_failure_prob(df_test)[0]
                    
                    # Store results
                    results.append({
                        'weight_on_bit': wob,
                        'rotary_speed': rpm,
                        'flow_rate': flow,
                        'predicted_rop': predicted_rop,
                        'bit_failure_prob': bit_failure_prob
                    })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find optimal parameters (max ROP with failure prob < 0.5)
        valid_results = results_df[results_df['bit_failure_prob'] < 0.5]
        if len(valid_results) > 0:
            optimal_row = valid_results.loc[valid_results['predicted_rop'].idxmax()]
        else:
            # If all combinations have high failure probability, just pick max ROP
            optimal_row = results_df.loc[results_df['predicted_rop'].idxmax()]
        
        return results_df, optimal_row