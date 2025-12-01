"""
Caption Quality Analyzer - Deep NLP Analysis for Caption Assessment

Analyzes caption quality across multiple dimensions:
- Specificity vs genericness
- Grammatical correctness
- Completeness (coverage of visible elements)
- Alignment with image content
- Descriptive richness
- Technical accuracy
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import re
from collections import Counter
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from base_engine import BaseEngine

# Import NLP libraries
try:
    import spacy
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import wordnet
    from textblob import TextBlob
    
    # Download required NLTK data if not present
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    for resource in ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'brown']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
            
except ImportError as e:
    print(f"Error: Missing required NLP library: {e}", file=sys.stderr)
    print("Install with: pip install spacy textblob nltk", file=sys.stderr)
    print("Then: python -m spacy download en_core_web_sm", file=sys.stderr)
    sys.exit(1)


class CaptionQualityAnalyzer(BaseEngine):
    """
    Analyzes caption quality using multiple NLP techniques
    """
    
    def __init__(self):
        super().__init__("caption_quality_analyzer", "1.0.0")
        
        self.nlp = None
        self.generic_words = {
            'image', 'photo', 'picture', 'showing', 'shows', 'depicts',
            'featuring', 'contains', 'includes', 'something', 'things',
            'stuff', 'various', 'different', 'several', 'multiple',
            'some', 'many', 'few', 'lot', 'lots', 'bunch'
        }
        
        self.quality_words = {
            'beautiful', 'stunning', 'amazing', 'gorgeous', 'breathtaking',
            'magnificent', 'spectacular', 'impressive', 'striking', 'dramatic',
            'vibrant', 'serene', 'peaceful', 'majestic', 'elegant'
        }
        
        self.technical_terms = {
            'bokeh', 'composition', 'lighting', 'exposure', 'contrast',
            'saturation', 'perspective', 'depth', 'foreground', 'background',
            'midground', 'silhouette', 'reflection', 'shadow', 'highlight'
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize Caption Quality Analyzer"""
        try:
            self.logger.info("Initializing Caption Quality Analyzer...")
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model not found, using basic analysis")
                self.nlp = None
            
            self._mark_initialized()
            
            return {
                "status": "ready",
                "nlp_available": self.nlp is not None,
                "capabilities": [
                    "specificity_analysis",
                    "grammar_check",
                    "completeness_assessment",
                    "alignment_scoring",
                    "richness_evaluation"
                ]
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            raise
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process caption quality analysis requests"""
        action = request.get('action')
        
        if action == 'health_check':
            return await self.health_check()
        
        elif action == 'analyze_caption':
            return await self._analyze_caption(
                request['caption'],
                request.get('image_features', {}),
                request.get('detailed', False)
            )
        
        elif action == 'batch_analyze':
            return await self._batch_analyze(
                request['captions'],
                request.get('image_features_list', [])
            )
        
        elif action == 'compare_captions':
            return await self._compare_captions(
                request['caption1'],
                request['caption2']
            )
        
        elif action == 'suggest_improvements':
            return await self._suggest_improvements(
                request['caption'],
                request.get('image_features', {})
            )
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _analyze_caption(self, caption: str, image_features: Dict, detailed: bool = False) -> Dict[str, Any]:
        """
        Comprehensive caption quality analysis
        """
        try:
            if not caption or not isinstance(caption, str):
                return {
                    "error": "Invalid caption",
                    "scores": self._get_zero_scores()
                }
            
            # Clean caption
            caption = caption.strip()
            
            # Perform various analyses
            analysis = {
                "caption": caption,
                "length": len(caption),
                "word_count": len(caption.split()),
                "sentence_count": len(sent_tokenize(caption)),
                
                # Quality scores (0-1)
                "scores": {
                    "specificity": self._calculate_specificity(caption),
                    "grammar": self._check_grammar(caption),
                    "completeness": self._assess_completeness(caption, image_features),
                    "richness": self._evaluate_richness(caption),
                    "technical": self._evaluate_technical_accuracy(caption),
                    "alignment": self._calculate_alignment(caption, image_features)
                }
            }
            
            # Overall quality score
            weights = {
                "specificity": 0.25,
                "grammar": 0.15,
                "completeness": 0.20,
                "richness": 0.15,
                "technical": 0.10,
                "alignment": 0.15
            }
            
            analysis["overall_score"] = sum(
                score * weights[key] 
                for key, score in analysis["scores"].items()
            )
            
            # Quality tier
            if analysis["overall_score"] >= 0.8:
                analysis["quality_tier"] = "excellent"
            elif analysis["overall_score"] >= 0.6:
                analysis["quality_tier"] = "good"
            elif analysis["overall_score"] >= 0.4:
                analysis["quality_tier"] = "fair"
            else:
                analysis["quality_tier"] = "poor"
            
            # Detailed analysis if requested
            if detailed:
                analysis["detailed"] = {
                    "entities": self._extract_entities(caption),
                    "keywords": self._extract_keywords(caption),
                    "sentiment": self._analyze_sentiment(caption),
                    "complexity": self._analyze_complexity(caption),
                    "issues": self._identify_issues(caption, analysis["scores"])
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Caption analysis failed: {e}")
            return {
                "error": str(e),
                "scores": self._get_zero_scores()
            }
    
    def _calculate_specificity(self, caption: str) -> float:
        """
        Calculate how specific vs generic the caption is
        """
        words = set(word.lower() for word in caption.split())
        
        # Count generic words
        generic_count = len(words.intersection(self.generic_words))
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Check for specific details
        has_numbers = bool(re.search(r'\d+', caption))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', caption))
        has_colors = bool(re.search(r'\b(red|blue|green|yellow|orange|purple|black|white|gray|brown)\b', caption.lower()))
        has_directions = bool(re.search(r'\b(left|right|center|top|bottom|foreground|background)\b', caption.lower()))
        
        # Calculate specificity score
        generic_ratio = generic_count / total_words
        specificity = 1.0 - generic_ratio
        
        # Bonus for specific details
        if has_numbers:
            specificity += 0.1
        if has_proper_nouns:
            specificity += 0.1
        if has_colors:
            specificity += 0.05
        if has_directions:
            specificity += 0.05
        
        return min(specificity, 1.0)
    
    def _check_grammar(self, caption: str) -> float:
        """
        Check grammatical correctness
        """
        try:
            blob = TextBlob(caption)
            
            # Basic grammar checks
            issues = 0
            
            # Check for sentence structure
            sentences = sent_tokenize(caption)
            for sentence in sentences:
                # Check if sentence starts with capital
                if sentence and not sentence[0].isupper():
                    issues += 1
                
                # Check for basic punctuation
                if sentence and sentence[-1] not in '.!?':
                    issues += 1
                
                # Check for double spaces
                if '  ' in sentence:
                    issues += 1
            
            # Use TextBlob for additional checks
            try:
                # Check if it can be parsed
                _ = blob.parse()
                parse_score = 1.0
            except:
                parse_score = 0.5
                issues += 2
            
            # Calculate grammar score
            max_issues = len(sentences) * 3
            grammar_score = max(0, 1.0 - (issues / max(max_issues, 1)))
            
            return (grammar_score + parse_score) / 2
            
        except Exception as e:
            self.logger.debug(f"Grammar check failed: {e}")
            return 0.5  # Default to neutral score
    
    def _assess_completeness(self, caption: str, image_features: Dict) -> float:
        """
        Assess if caption covers visible elements
        """
        if not image_features:
            # Can't assess without image features
            return 0.5
        
        completeness_score = 0.5  # Base score
        covered_elements = 0
        total_elements = 0
        
        # Check if caption mentions detected content
        caption_lower = caption.lower()
        
        # Check content type
        if 'content_type' in image_features:
            total_elements += 1
            if image_features['content_type'].lower() in caption_lower:
                covered_elements += 1
        
        # Check for people if detected
        if image_features.get('has_people'):
            total_elements += 1
            people_words = ['person', 'people', 'man', 'woman', 'child', 'boy', 'girl']
            if any(word in caption_lower for word in people_words):
                covered_elements += 1
        
        # Check for colors
        if 'dominant_color' in image_features:
            total_elements += 1
            if image_features['dominant_color'].lower() in caption_lower:
                covered_elements += 1
        
        # Check shot scale
        if 'shot_scale' in image_features:
            total_elements += 1
            scale_words = {
                'close_up': ['close', 'closeup', 'detail', 'macro'],
                'medium': ['medium', 'mid'],
                'wide': ['wide', 'landscape', 'panorama', 'distant']
            }
            if any(word in caption_lower for word in scale_words.get(image_features['shot_scale'], [])):
                covered_elements += 1
        
        # Check time of day if available
        if 'time_of_day' in image_features.get('temporal', {}).get('derived', {}):
            total_elements += 1
            time = image_features['temporal']['derived']['time_of_day']
            if time.lower() in caption_lower:
                covered_elements += 1
        
        # Calculate completeness
        if total_elements > 0:
            completeness_score = covered_elements / total_elements
        
        # Adjust based on caption length
        word_count = len(caption.split())
        if word_count < 5:
            completeness_score *= 0.7  # Too short
        elif word_count > 50:
            completeness_score *= 1.1  # Detailed
        
        return min(completeness_score, 1.0)
    
    def _evaluate_richness(self, caption: str) -> float:
        """
        Evaluate descriptive richness
        """
        words = caption.lower().split()
        
        # Calculate various richness metrics
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / max(len(words), 1)
        
        # Check for descriptive adjectives
        adjectives = 0
        if self.nlp:
            try:
                doc = self.nlp(caption)
                adjectives = len([token for token in doc if token.pos_ == "ADJ"])
            except:
                # Fallback to simple heuristic
                adjective_endings = ['ful', 'ous', 'ive', 'ing', 'ed']
                adjectives = sum(1 for word in words if any(word.endswith(end) for end in adjective_endings))
        
        adjective_ratio = adjectives / max(len(words), 1)
        
        # Check for quality descriptors
        quality_count = len(set(words).intersection(self.quality_words))
        
        # Check for sensory words
        sensory_words = {
            'bright', 'dark', 'colorful', 'vivid', 'soft', 'hard',
            'smooth', 'rough', 'warm', 'cool', 'cold', 'hot'
        }
        sensory_count = len(set(words).intersection(sensory_words))
        
        # Calculate richness score
        richness = (
            vocabulary_diversity * 0.3 +
            min(adjective_ratio * 3, 1.0) * 0.3 +
            min(quality_count / 3, 1.0) * 0.2 +
            min(sensory_count / 2, 1.0) * 0.2
        )
        
        return min(richness, 1.0)
    
    def _evaluate_technical_accuracy(self, caption: str) -> float:
        """
        Evaluate use of technical/photographic terms
        """
        words = set(caption.lower().split())
        
        # Check for technical terms
        technical_count = len(words.intersection(self.technical_terms))
        
        # Check for composition terms
        composition_terms = {
            'rule of thirds', 'leading lines', 'symmetry', 'framing',
            'golden ratio', 'depth of field', 'focal point'
        }
        
        caption_lower = caption.lower()
        composition_count = sum(1 for term in composition_terms if term in caption_lower)
        
        # Check for lighting descriptions
        lighting_terms = {
            'backlit', 'sidelit', 'front lit', 'natural light',
            'golden hour', 'blue hour', 'harsh light', 'soft light'
        }
        lighting_count = sum(1 for term in lighting_terms if term in caption_lower)
        
        # Calculate technical score
        technical_score = min(
            (technical_count / 3) + 
            (composition_count / 2) + 
            (lighting_count / 2),
            1.0
        )
        
        # Don't penalize non-technical captions too much
        return 0.5 + (technical_score * 0.5)
    
    def _calculate_alignment(self, caption: str, image_features: Dict) -> float:
        """
        Calculate how well caption aligns with image features
        """
        if not image_features:
            return 0.5  # Neutral if no features
        
        alignment_score = 0.5
        caption_lower = caption.lower()
        
        # Check brightness alignment
        if 'brightness_score' in image_features:
            brightness = image_features['brightness_score']
            if brightness < 0.3 and any(word in caption_lower for word in ['dark', 'night', 'shadow', 'dim']):
                alignment_score += 0.1
            elif brightness > 0.7 and any(word in caption_lower for word in ['bright', 'sunny', 'light', 'illuminated']):
                alignment_score += 0.1
        
        # Check complexity alignment
        if 'visual_complexity_score' in image_features:
            complexity = image_features['visual_complexity_score']
            if complexity < 0.3 and any(word in caption_lower for word in ['simple', 'minimal', 'clean', 'empty']):
                alignment_score += 0.1
            elif complexity > 0.7 and any(word in caption_lower for word in ['busy', 'complex', 'detailed', 'crowded']):
                alignment_score += 0.1
        
        # Check color alignment
        if 'dominant_color' in image_features:
            if image_features['dominant_color'].lower() in caption_lower:
                alignment_score += 0.15
        
        # Check vibrancy alignment
        if 'color_vibrancy_score' in image_features:
            vibrancy = image_features['color_vibrancy_score']
            if vibrancy > 0.7 and any(word in caption_lower for word in ['vibrant', 'colorful', 'vivid', 'saturated']):
                alignment_score += 0.1
            elif vibrancy < 0.3 and any(word in caption_lower for word in ['muted', 'desaturated', 'pale', 'subdued']):
                alignment_score += 0.1
        
        return min(alignment_score, 1.0)
    
    def _extract_entities(self, caption: str) -> List[str]:
        """Extract named entities from caption"""
        entities = []
        
        if self.nlp:
            try:
                doc = self.nlp(caption)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
            except:
                pass
        
        # Fallback: extract capitalized words
        if not entities:
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', caption)
        
        return entities
    
    def _extract_keywords(self, caption: str) -> List[str]:
        """Extract key descriptive words"""
        words = caption.lower().split()
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Get most common
        word_freq = Counter(keywords)
        return [word for word, _ in word_freq.most_common(5)]
    
    def _analyze_sentiment(self, caption: str) -> Dict[str, float]:
        """Analyze caption sentiment"""
        try:
            blob = TextBlob(caption)
            return {
                "polarity": blob.sentiment.polarity,  # -1 to 1
                "subjectivity": blob.sentiment.subjectivity  # 0 to 1
            }
        except:
            return {"polarity": 0.0, "subjectivity": 0.5}
    
    def _analyze_complexity(self, caption: str) -> Dict[str, Any]:
        """Analyze linguistic complexity"""
        words = caption.split()
        sentences = sent_tokenize(caption)
        
        return {
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "vocabulary_size": len(set(words)),
            "lexical_diversity": len(set(words)) / max(len(words), 1)
        }
    
    def _identify_issues(self, caption: str, scores: Dict[str, float]) -> List[str]:
        """Identify specific issues with the caption"""
        issues = []
        
        if scores["specificity"] < 0.4:
            issues.append("Caption is too generic")
        
        if scores["grammar"] < 0.6:
            issues.append("Grammar issues detected")
        
        if scores["completeness"] < 0.4:
            issues.append("Caption may be missing important details")
        
        if scores["richness"] < 0.3:
            issues.append("Caption lacks descriptive richness")
        
        if len(caption.split()) < 5:
            issues.append("Caption is too short")
        elif len(caption.split()) > 100:
            issues.append("Caption is excessively long")
        
        if caption.isupper():
            issues.append("Caption is all uppercase")
        
        if not caption[0].isupper():
            issues.append("Caption doesn't start with capital letter")
        
        if caption[-1] not in '.!?':
            issues.append("Caption lacks proper punctuation")
        
        return issues
    
    async def _batch_analyze(self, captions: List[str], image_features_list: List[Dict]) -> Dict[str, Any]:
        """Analyze multiple captions"""
        results = []
        
        # Ensure we have matching features for each caption
        if not image_features_list:
            image_features_list = [{}] * len(captions)
        elif len(image_features_list) < len(captions):
            image_features_list.extend([{}] * (len(captions) - len(image_features_list)))
        
        for caption, features in zip(captions, image_features_list):
            analysis = await self._analyze_caption(caption, features, detailed=False)
            results.append(analysis)
        
        # Calculate aggregate statistics
        scores = [r["overall_score"] for r in results if "overall_score" in r]
        
        return {
            "count": len(results),
            "results": results,
            "statistics": {
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "quality_distribution": {
                    "excellent": sum(1 for r in results if r.get("quality_tier") == "excellent"),
                    "good": sum(1 for r in results if r.get("quality_tier") == "good"),
                    "fair": sum(1 for r in results if r.get("quality_tier") == "fair"),
                    "poor": sum(1 for r in results if r.get("quality_tier") == "poor")
                }
            }
        }
    
    async def _compare_captions(self, caption1: str, caption2: str) -> Dict[str, Any]:
        """Compare two captions"""
        analysis1 = await self._analyze_caption(caption1, {}, detailed=True)
        analysis2 = await self._analyze_caption(caption2, {}, detailed=True)
        
        comparison = {
            "caption1": analysis1,
            "caption2": analysis2,
            "winner": "caption1" if analysis1["overall_score"] > analysis2["overall_score"] else "caption2",
            "score_difference": abs(analysis1["overall_score"] - analysis2["overall_score"]),
            "detailed_comparison": {}
        }
        
        # Compare individual metrics
        for metric in analysis1["scores"]:
            comparison["detailed_comparison"][metric] = {
                "caption1": analysis1["scores"][metric],
                "caption2": analysis2["scores"][metric],
                "difference": analysis1["scores"][metric] - analysis2["scores"][metric]
            }
        
        return comparison
    
    async def _suggest_improvements(self, caption: str, image_features: Dict) -> Dict[str, Any]:
        """Suggest improvements for a caption"""
        analysis = await self._analyze_caption(caption, image_features, detailed=True)
        
        suggestions = []
        
        # Based on scores
        if analysis["scores"]["specificity"] < 0.5:
            suggestions.append({
                "issue": "Low specificity",
                "suggestion": "Add specific details like numbers, proper nouns, or precise descriptions",
                "example": "Instead of 'a building', say 'a three-story Victorian house'"
            })
        
        if analysis["scores"]["completeness"] < 0.5 and image_features:
            missing = []
            if image_features.get("dominant_color") and image_features["dominant_color"].lower() not in caption.lower():
                missing.append(f"dominant color ({image_features['dominant_color']})")
            if image_features.get("shot_scale") and image_features["shot_scale"] not in caption.lower():
                missing.append(f"shot scale ({image_features['shot_scale']})")
            
            if missing:
                suggestions.append({
                    "issue": "Missing elements",
                    "suggestion": f"Consider mentioning: {', '.join(missing)}",
                    "priority": "high"
                })
        
        if analysis["scores"]["richness"] < 0.4:
            suggestions.append({
                "issue": "Lacks descriptive richness",
                "suggestion": "Add sensory details, adjectives, or atmospheric descriptions",
                "example": "Describe textures, lighting, mood, or atmosphere"
            })
        
        if analysis["word_count"] < 8:
            suggestions.append({
                "issue": "Caption too short",
                "suggestion": "Expand to at least 10-15 words for better context",
                "priority": "medium"
            })
        
        # Grammar issues
        if analysis["detailed"]["issues"]:
            for issue in analysis["detailed"]["issues"][:3]:  # Top 3 issues
                suggestions.append({
                    "issue": issue,
                    "priority": "high" if "grammar" in issue.lower() else "medium"
                })
        
        return {
            "original_caption": caption,
            "current_score": analysis["overall_score"],
            "suggestions": suggestions,
            "potential_improvement": min(1.0 - analysis["overall_score"], 0.3)  # Max 30% improvement
        }
    
    def _get_zero_scores(self) -> Dict[str, float]:
        """Return zero scores for error cases"""
        return {
            "specificity": 0.0,
            "grammar": 0.0,
            "completeness": 0.0,
            "richness": 0.0,
            "technical": 0.0,
            "alignment": 0.0
        }


async def main():
    """Main entry point for testing"""
    analyzer = CaptionQualityAnalyzer()
    await analyzer.initialize()
    
    # Run as service
    print(json.dumps({"event": "initialized", "service": "caption_quality_analyzer"}), flush=True)
    
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = await analyzer.handle_request(request)
            print(json.dumps(response), flush=True)
        except Exception as e:
            error_response = {
                "success": False,
                "error": {"message": str(e)}
            }
            print(json.dumps(error_response), flush=True)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
