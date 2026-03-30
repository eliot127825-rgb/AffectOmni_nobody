"""
Event extraction module
Extract key events from the model's <think> output
"""

import re
import json
from typing import List, Dict, Optional
import warnings


class Event:
    """Event data structure"""
    def __init__(self, anchor: str, query: str):
        """
        Args:
            anchor: Anchor text for locating in the original text (preferably original sentence/fragment)
            query: Query phrase for CLIP matching (shorter and more "visual")
        """
        self.anchor = anchor
        self.query = query
    
    def __repr__(self):
        return f"Event(anchor='{self.anchor}', query='{self.query}')"
    
    def to_dict(self):
        return {"anchor": self.anchor, "query": self.query}


def extract_events_with_llm(
    think_text: str,
    model,
    processor,
    max_events: int = 10
) -> List[Event]:
    """
    Use LLM to extract events from think text (recommended approach)
    
    Args:
        think_text: Original <think> text
        model: Language model
        processor: Model processor
        max_events: Maximum number of events to extract
    
    Returns:
        events: List of Event objects
    
    Strategy:
        Have the model output a "list of event key points" instead of timestamps.
        This does not change the model's writing paradigm, yielding a high success rate.
    """
    prompt = f"""Based on the following reasoning text, extract key visual events as short phrases.

Reasoning text:
{think_text}

Requirements:
1. Extract {max_events} or fewer key visual events that can be observed in video frames
2. For each event, provide:
   - anchor: The original phrase from the reasoning text (keep it as close to original as possible)
   - query: A short, visual description for image matching (3-8 words, noun-focused)
3. Events should be in chronological order as they appear in the reasoning
4. Output ONLY valid JSON in this format:

{{"events": [
  {{"anchor": "the woman picks up the rose", "query": "woman picking up red rose"}},
  {{"anchor": "she smiles at the man", "query": "woman smiling at man"}}
]}}

JSON output:"""

    # Construct messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that extracts key visual events from text."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    
    # Apply chat template
    try:
        texts = processor.apply_chat_template(
            [messages],
            tokenize=False,
            add_generation_prompt=True
        )
        text = texts[0]
        
        # Process inputs
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # Generate
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=False
            )
        
        # Decode
        generated_text = processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        # Parse JSON
        events = _parse_events_json(generated_text)
        return events[:max_events]
        
    except Exception as e:
        warnings.warn(f"LLM extraction failed: {e}, falling back to rule-based")
        return extract_events_rule_based(think_text, max_events)


def extract_events_rule_based(
    think_text: str,
    max_events: int = 10
) -> List[Event]:
    """
    Rule-based event extraction (Fallback)
    
    Strategy:
        1. Sentence splitting
        2. Keyword filtering (action words, visual words)
        3. Phrase extraction
    """
    # Clean text
    text = think_text.strip()
    
    # Sentence splitting (simple version)
    sentences = re.split(r'[.!?]\s+', text)
    
    # Visual action keywords (more specific)
    visual_action_keywords = [
        'pick', 'hold', 'give', 'receive', 'grab', 'touch', 'point',
        'smile', 'frown', 'laugh', 'cry', 'nod', 'shake', 'turn',
        'wear', 'dress', 'put on', 'take off',
        'stand', 'sit', 'walk', 'run', 'jump', 'lean', 'bend',
        'raise', 'lower', 'open', 'close', 'wave', 'gesture',
        'kiss', 'hug', 'push', 'pull', 'throw', 'catch'
    ]
    
    # Visual object keywords
    visual_object_keywords = [
        'woman', 'man', 'person', 'people', 'child', 'baby',
        'hair', 'face', 'eyes', 'eyebrows', 'hand', 'arm', 'leg',
        'dress', 'suit', 'jacket', 'shirt', 'hat', 'glasses',
        'rose', 'flower', 'book', 'phone', 'bag', 'backpack',
        'table', 'chair', 'door', 'window', 'car', 'room'
    ]
    
    # Reasoning words to filter out (extended)
    reasoning_keywords = [
        'think', 'consider', 'seem', 'suggest', 'indicate', 'imply',
        'therefore', 'so', 'thus', 'hence', 'because', 'since',
        'option', 'choice', 'answer', 'question', 'let me', 'okay',
        "i'm", "i see", "i hear", "i notice", "now,", "first,", 
        "looking at", "listening to", "focus on", "take a look",
        "makes me think", "makes me", "probably", "definitely",
        "if i", "best fit", "based on", "whole place", "clearly"
    ]
    
    events = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:  # higher minimum length
            continue
        
        sentence_lower = sentence.lower()
        
        # Filter out reasoning sentences (stricter)
        if any(kw in sentence_lower for kw in reasoning_keywords):
            continue
        
        # Must contain both action and object words (stricter)
        has_action = any(kw in sentence_lower for kw in visual_action_keywords)
        has_object = any(kw in sentence_lower for kw in visual_object_keywords)
        
        # Only keep sentences containing both action and object
        if has_action and has_object:
            anchor = sentence
            query = _simplify_to_query(sentence)
            events.append(Event(anchor, query))
            
            if len(events) >= max_events:
                break
    
    # If too few events extracted, lower the threshold (at least action or object)
    if len(events) < 3:
        events = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 15:
                continue
            
            sentence_lower = sentence.lower()
            
            # Still filter out reasoning sentences
            if any(kw in sentence_lower for kw in reasoning_keywords):
                continue
            
            has_action = any(kw in sentence_lower for kw in visual_action_keywords)
            has_object = any(kw in sentence_lower for kw in visual_object_keywords)
            
            if has_action or has_object:
                anchor = sentence
                query = _simplify_to_query(sentence)
                events.append(Event(anchor, query))
                
                if len(events) >= max_events:
                    break
    
    return events


def _simplify_to_query(sentence: str) -> str:
    """
    Simplify a sentence into a CLIP-friendly query phrase
    
    Strategy:
        - Remove adverbs, conjunctions
        - Keep main nouns and verbs
        - Limit length
    """
    # Simple version: keep first 8 words
    words = sentence.split()[:8]
    
    # Remove some stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'that', 'this', 'it'}
    filtered = [w for w in words if w.lower() not in stop_words]
    
    return ' '.join(filtered[:6])  # limit to 6 words


def _parse_events_json(text: str) -> List[Event]:
    """
    Parse JSON-formatted events from generated text
    
    Supported formats:
        {"events": [...]}
        or a direct array [...]
    """
    # Extract JSON part (may be inside markdown code blocks)
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to extract JSON object directly
        json_match = re.search(r'\{.*"events".*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError("No valid JSON found in generated text")
    
    # Parse JSON
    data = json.loads(json_str)
    
    if isinstance(data, dict) and 'events' in data:
        events_data = data['events']
    elif isinstance(data, list):
        events_data = data
    else:
        raise ValueError("Invalid JSON structure")
    
    # Convert to Event objects
    events = []
    for item in events_data:
        if isinstance(item, dict) and 'anchor' in item and 'query' in item:
            events.append(Event(item['anchor'], item['query']))
        else:
            warnings.warn(f"Invalid event format: {item}")
    
    return events


def extract_events(
    think_text: str,
    method: str = "llm",
    model=None,
    processor=None,
    max_events: int = 10
) -> List[Event]:
    """
    Unified event extraction interface
    
    Args:
        think_text: <think> text
        method: "llm" or "rule"
        model, processor: Required for LLM method
        max_events: Maximum number of events to extract
    
    Returns:
        events: List of Event objects
    """
    if method == "llm":
        if model is None or processor is None:
            warnings.warn("model/processor not provided, falling back to rule-based")
            return extract_events_rule_based(think_text, max_events)
        return extract_events_with_llm(think_text, model, processor, max_events)
    elif method == "rule":
        return extract_events_rule_based(think_text, max_events)
    else:
        raise ValueError(f"Unknown method: {method}")


# Convenience functions
def events_to_queries(events: List[Event]) -> List[str]:
    """Extract all query strings"""
    return [e.query for e in events]


def events_to_dict_list(events: List[Event]) -> List[Dict]:
    """Convert to list of dicts (for saving/debugging)"""
    return [e.to_dict() for e in events]
