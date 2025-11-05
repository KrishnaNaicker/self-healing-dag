"""
Confidence Check Node - Decides whether to accept prediction or trigger fallback.
"""

from typing import Dict, Any, Literal


class ConfidenceCheckNode:
    """
    Evaluates prediction confidence against a threshold.
    Routes to either acceptance or fallback based on confidence.
    """
    
    def __init__(self, confidence_threshold: float = 0.70):
        """
        Initialize confidence checker.
        
        Args:
            confidence_threshold: Minimum confidence to accept prediction (default: 0.70)
        """
        self.threshold = confidence_threshold
        print(f"✅ Confidence check node initialized (threshold: {self.threshold:.0%})")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if prediction confidence meets threshold.
        
        Args:
            state: Dictionary containing 'confidence'
        
        Returns:
            Updated state with 'should_fallback' decision
        """
        confidence = state['confidence']
        
        # Decision logic
        if confidence >= self.threshold:
            state['should_fallback'] = False
            state['confidence_status'] = 'HIGH'
            state['decision_reason'] = f"Confidence {confidence:.2%} >= threshold {self.threshold:.0%}"
        else:
            state['should_fallback'] = True
            state['confidence_status'] = 'LOW'
            state['decision_reason'] = f"Confidence {confidence:.2%} < threshold {self.threshold:.0%}"
        
        return state
    
    def should_fallback(self, state: Dict[str, Any]) -> Literal["accept", "fallback"]:
        """
        Routing function for LangGraph conditional edges.
        
        Args:
            state: Current state
        
        Returns:
            "accept" if confidence is high, "fallback" if low
        """
        return "fallback" if state['should_fallback'] else "accept"


def create_confidence_check_node(confidence_threshold: float = 0.70):
    """Factory function to create confidence check node."""
    return ConfidenceCheckNode(confidence_threshold)