"""
Fallback Node - Handles low-confidence predictions through human-in-the-loop.
"""

from typing import Dict, Any, Optional, List, Callable


class FallbackNode:
    """
    Manages fallback strategies when prediction confidence is low.
    Can request user clarification or use backup models.
    """
    
    def __init__(
        self,
        backup_model: Optional[Any] = None,
        interactive_mode: bool = True
    ):
        """
        Initialize fallback node.
        
        Args:
            backup_model: Optional backup classifier
            interactive_mode: Whether to ask user for input (CLI mode)
        """
        self.backup_model = backup_model
        self.interactive_mode = interactive_mode
        print(f"✅ Fallback node initialized (interactive: {interactive_mode})")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute fallback strategy.
        
        Args:
            state: Current state with low-confidence prediction
        
        Returns:
            Updated state with corrected/confirmed label
        """
        predicted_label = state['predicted_label']
        confidence = state['confidence']
        all_probs = state['all_probabilities']
        
        # Get top predictions for suggestion
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        top_2 = sorted_probs[:2]
        
        state['fallback_triggered'] = True
        state['fallback_method'] = None
        
        # Strategy 1: Try backup model (if available)
        if self.backup_model:
            backup_result = self._use_backup_model(state)
            if backup_result.get('backup_confidence', 0) >= 0.80:
                state['final_label'] = backup_result['backup_label']
                state['fallback_method'] = 'backup_model'
                state['backup_used'] = True
                return state
        
        # Strategy 2: Ask user for clarification (if interactive)
        if self.interactive_mode:
            state = self._request_user_clarification(state, top_2)
            state['fallback_method'] = 'user_clarification'
            return state
        
        # Strategy 3: Default to highest probability (non-interactive)
        state['final_label'] = predicted_label
        state['fallback_method'] = 'default_highest'
        state['fallback_note'] = "Non-interactive mode: using highest probability"
        
        return state
    
    def _use_backup_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Use backup model for prediction."""
        # Placeholder for backup model logic
        # In bonus implementation, this would call a zero-shot classifier
        return {
            'backup_label': state['predicted_label'],
            'backup_confidence': 0.75
        }
    
    def _request_user_clarification(
        self,
        state: Dict[str, Any],
        top_predictions: List[tuple]
    ) -> Dict[str, Any]:
        """
        Request user input for clarification.
        This will be called from the CLI interface.
        """
        predicted_label = state['predicted_label']
        confidence = state['confidence']
        
        # Store clarification request info
        state['clarification_needed'] = True
        state['suggested_labels'] = [label for label, _ in top_predictions]
        
        # Generate prompt HERE (single place!)
        prompt = f"\n⚠️  Confidence too low ({confidence:.0%}). Need clarification...\n"
        prompt += f"\nModel predicted: {predicted_label} ({confidence:.0%})\n"
        prompt += f"Top candidates:\n"
        for i, (label, prob) in enumerate(top_predictions, 1):
            prompt += f"  {i}. {label} ({prob:.0%})\n"
        prompt += f"\nWas this expressing: "
        prompt += ", ".join([f"{label} ({prob:.0%})" for label, prob in top_predictions])
        prompt += "?\n"
        
        state['clarification_prompt'] = prompt
        
        return state
    
    def process_user_input(
        self,
        state: Dict[str, Any],
        user_input: str
    ) -> Dict[str, Any]:
        """
        Process user's clarification input.
        
        Args:
            state: Current state
            user_input: User's response
        
        Returns:
            Updated state with corrected label
        """
        suggested_labels = state.get('suggested_labels', [])
        
        # Try to parse user input
        user_input = user_input.strip().lower()
        
        # Check if it's a number (selecting from options)
        if user_input.isdigit():
            choice = int(user_input) - 1
            if 0 <= choice < len(suggested_labels):
                corrected_label = suggested_labels[choice]
                state['final_label'] = corrected_label
                state['user_corrected'] = True
                state['user_input'] = user_input
                return state
        
        # Check if it's a label name
        for label in suggested_labels:
            if label.lower() in user_input:
                state['final_label'] = label
                state['user_corrected'] = True
                state['user_input'] = user_input
                return state
        
        # If unclear, ask again or use original prediction
        state['final_label'] = state['predicted_label']
        state['user_corrected'] = False
        state['clarification_unclear'] = True
        
        return state


def create_fallback_node(backup_model: Optional[Any] = None, interactive_mode: bool = True):
    """Factory function to create fallback node."""
    return FallbackNode(backup_model, interactive_mode)