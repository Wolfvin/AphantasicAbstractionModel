"""Rule Learner with TransE KG Embedding
TransE: vec(h) + vec(r) ≈ vec(t), d=64, margin ranking loss
Observations use actual triplet score
Bug fix: grad_neg_neg_t → grad_neg_t
"""
import numpy as np
from typing import Optional, Dict, List

class TransEEmbedding:
    """TransE Knowledge Graph Embedding"""
    
    def __init__(self, d: int = 64, margin: float = 1.0, lr: float = 0.01):
        self.d = d
        self.margin = margin
        self.lr = lr
        self.entity_embeddings = {}  # entity -> np.array
        self.relation_embeddings = {}  # relation -> np.array
        self.triplets = []  # (h, r, t) triplets
    
    def _init_embedding(self, name: str, is_relation: bool = False) -> np.array:
        """Initialize a new embedding"""
        emb = np.random.randn(self.d) * 0.1
        if is_relation:
            # Normalize relation vectors
            emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    
    def add_triplet(self, h: str, r: str, t: str):
        """Add a triplet and update embeddings"""
        if h not in self.entity_embeddings:
            self.entity_embeddings[h] = self._init_embedding(h)
        if t not in self.entity_embeddings:
            self.entity_embeddings[t] = self._init_embedding(t)
        if r not in self.relation_embeddings:
            self.relation_embeddings[r] = self._init_embedding(r, is_relation=True)
        
        self.triplets.append((h, r, t))
        self._train_step(h, r, t)
    
    def _train_step(self, h: str, r: str, t: str):
        """Single training step with margin ranking loss"""
        # Positive score
        h_emb = self.entity_embeddings[h]
        r_emb = self.relation_embeddings[r]
        t_emb = self.entity_embeddings[t]
        
        pos_score = np.linalg.norm(h_emb + r_emb - t_emb)
        
        # Negative sample (corrupt tail)
        neg_t = list(self.entity_embeddings.keys())
        if len(neg_t) > 1:
            neg_t = [nt for nt in neg_t if nt != t]
            import random
            neg_t_name = random.choice(neg_t)
            neg_t_emb = self.entity_embeddings[neg_t_name]
            neg_score = np.linalg.norm(h_emb + r_emb - neg_t_emb)
            
            # Margin ranking loss
            loss = max(0, self.margin + pos_score - neg_score)
            
            if loss > 0:
                # Gradient update
                grad_h = 2 * (h_emb + r_emb - t_emb) - 2 * (h_emb + r_emb - neg_t_emb)
                grad_r = grad_h  # Same gradient direction
                grad_t = -2 * (h_emb + r_emb - t_emb)
                grad_neg_t = 2 * (h_emb + r_emb - neg_t_emb)  # Bug fix: was grad_neg_neg_t
                
                self.entity_embeddings[h] -= self.lr * grad_h
                self.relation_embeddings[r] -= self.lr * grad_r
                self.entity_embeddings[t] -= self.lr * grad_t
                self.entity_embeddings[neg_t_name] -= self.lr * grad_neg_t  # Bug fix: was grad_neg_neg_t
    
    def score_triplet(self, h: str, r: str, t: str) -> float:
        """Score a triplet using actual triplet score"""
        if h not in self.entity_embeddings or r not in self.relation_embeddings or t not in self.entity_embeddings:
            return 0.0
        h_emb = self.entity_embeddings[h]
        r_emb = self.relation_embeddings[r]
        t_emb = self.entity_embeddings[t]
        return -np.linalg.norm(h_emb + r_emb - t_emb)  # Higher = better
    
    def predict_tail(self, h: str, r: str, top_k: int = 5) -> List[tuple]:
        """Predict most likely tail entity"""
        if h not in self.entity_embeddings or r not in self.relation_embeddings:
            return []
        
        h_emb = self.entity_embeddings[h]
        r_emb = self.relation_embeddings[r]
        target = h_emb + r_emb
        
        scores = []
        for entity, emb in self.entity_embeddings.items():
            if entity != h:
                score = -np.linalg.norm(target - emb)
                scores.append((entity, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class RuleLearner:
    """Learns rules from observations using TransE"""
    
    def __init__(self, self_core=None):
        self.self_core = self_core
        self.transe = TransEEmbedding(d=64)
        self.learned_rules = {}  # rule_id -> {conditions, conclusion, confidence, observations}
        self.observation_count = 0
        self.promotion_threshold = 3  # Need 3 observations to promote a rule
    
    def observe(self, triplet: dict, feedback: bool = None):
        """Record an observation for rule learning"""
        h = triplet.get('subject', '')
        r = triplet.get('predicate', '')
        t = triplet.get('object', '')
        
        if h and r and t:
            self.transe.add_triplet(h, r, t)
            self.observation_count += 1
            
            # Check if we can promote a rule
            self._try_promote_rule(r)
    
    def _try_promote_rule(self, relation: str):
        """Try to promote a relation to a learned rule"""
        # Count observations with this relation
        count = sum(1 for (h, r, t) in self.transe.triplets if r == relation)
        
        if count >= self.promotion_threshold:
            rule_id = f"rule_{relation}_{count}"
            if rule_id not in self.learned_rules:
                self.learned_rules[rule_id] = {
                    'conditions': [relation],
                    'conclusion': relation,
                    'confidence': min(0.9, 0.3 + count * 0.1),
                    'observations': count
                }
    
    def get_confidence(self, relation: str) -> float:
        """Get confidence for a relation based on observations"""
        count = sum(1 for (h, r, t) in self.transe.triplets if r == relation)
        if count == 0:
            return 0.0
        return min(0.9, 0.3 + count * 0.1)
