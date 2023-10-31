import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadAttention, PositionwiseFeedForward


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class FakeNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.claim_text_embedding = nn.Sequential(
            nn.Linear(config['text_dim'], config['hidden_dim']),
            # Mish()
            nn.ReLU()
        )
        self.evidence_text_embedding = nn.Sequential(
            nn.Linear(config['text_dim'], config['hidden_dim']),
            # Mish()
            nn.ReLU()
        )
        
        self.claim_qa_embedding = nn.Sequential(
            nn.Linear(config['text_dim'], config['hidden_dim']),
            # Mish()
            nn.ReLU()
        )
        self.evidence_qa_embedding = nn.Sequential(
            nn.Linear(config['text_dim'], config['hidden_dim']),
            # Mish()
            nn.ReLU()
        )

        self.qa_embedding = nn.Sequential(
            nn.Linear(config['qa_dim'], config['hidden_dim']),
            # Mish()
            nn.ReLU()
        )
        self.evidence_qa_embedding = nn.Sequential(
            nn.Linear(config['qa_dim'], config['hidden_dim']),
            # Mish()
            nn.ReLU()
        )

        self.claim_evidence_text_attention = MultiHeadAttention(config['head'], config['hidden_dim'], config['hidden_dim'], config['hidden_dim'], dropout=config['dropout'])
        self.claim_evidence_text_pos_ffn = PositionwiseFeedForward(config['hidden_dim'], config['hidden_dim']*2, dropout=config['dropout'])

        self.claim_evidence_qa_attention = MultiHeadAttention(config['head'], config['hidden_dim'], config['hidden_dim'], config['hidden_dim'], dropout=config['dropout'])
        self.claim_evidence_qa_pos_ffn = PositionwiseFeedForward(config['hidden_dim'], config['hidden_dim']*2, dropout=config['dropout'])

        self.text_qa_attention = MultiHeadAttention(config['head'], config['hidden_dim'], config['hidden_dim'], config['hidden_dim'], dropout=config['dropout'])
        self.text_qa_pos_ffn = PositionwiseFeedForward(config['hidden_dim'], config['hidden_dim']*2, dropout=config['dropout'])
        
        self.qa_text_attention = MultiHeadAttention(config['head'], config['hidden_dim'], config['hidden_dim'], config['hidden_dim'], dropout=config['dropout'])
        self.qa_text_pos_ffn = PositionwiseFeedForward(config['hidden_dim'], config['hidden_dim']*2, dropout=config['dropout'])

        self.claim_evidence_text_qa_attention = MultiHeadAttention(config['head'], config['hidden_dim'], config['hidden_dim'], config['hidden_dim'], dropout=config['dropout'])
        self.claim_evidence_text_qa_pos_ffn = PositionwiseFeedForward(config['hidden_dim'], config['hidden_dim']*2, dropout=config['dropout'])
        
        self.claim_evidence_qa_text_attention = MultiHeadAttention(config['head'], config['hidden_dim'], config['hidden_dim'], config['hidden_dim'], dropout=config['dropout'])
        self.claim_evidence_qa_text_pos_ffn = PositionwiseFeedForward(config['hidden_dim'], config['hidden_dim']*2, dropout=config['dropout'])

        self.attention_fusion = nn.Sequential(
            nn.Linear(config['hidden_dim']*16, config['hidden_dim']),
            nn.ReLU(),
        )

        self.feature_embedding = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(config['hidden_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, claim_text, evidence_text, claim_qa, evidence_qa):
        # transform to embeddings
        claim_text_embedding = self.claim_text_embedding(claim_text)
        evidence_text_embedding = self.evidence_text_embedding(evidence_text)
        claim_qa_embedding = self.claim_qa_embedding(claim_qa)
        evidence_qa_embedding = self.evidence_qa_embedding(evidence_qa)

        # CT + ET
        claim_evidence_text, _ = self.claim_evidence_text_attention(claim_text_embedding, evidence_text_embedding, evidence_text_embedding)
        claim_evidence_text = self.claim_evidence_text_pos_ffn(claim_evidence_text)
        evidence_claim_text, _ = self.claim_evidence_text_attention(evidence_text_embedding, claim_text_embedding, claim_text_embedding)
        evidence_claim_text = self.claim_evidence_text_pos_ffn(evidence_claim_text)
        # CQA + EQA
        claim_evidence_qa, _ = self.claim_evidence_qa_attention(claim_qa_embedding, evidence_qa_embedding, evidence_qa_embedding)
        claim_evidence_qa = self.claim_evidence_qa_pos_ffn(claim_evidence_qa)
        evidence_claim_qa, _ = self.claim_evidence_qa_attention(evidence_qa_embedding, claim_qa_embedding, claim_qa_embedding)
        evidence_claim_qa = self.claim_evidence_qa_pos_ffn(evidence_claim_qa)
        # CT + CQA
        claim_text_qa, _ = self.text_qa_attention(claim_text_embedding, claim_qa_embedding, claim_qa_embedding)
        claim_text_qa = self.text_qa_pos_ffn(claim_text_qa)
        claim_qa_text, _ = self.qa_text_attention(claim_qa_embedding, claim_text_embedding, claim_text_embedding)
        claim_qa_text = self.qa_text_pos_ffn(claim_qa_text)
        # ET + EQA
        evidence_text_qa, _ = self.text_qa_attention(evidence_text_embedding, evidence_qa_embedding, evidence_qa_embedding)
        evidence_text_qa = self.text_qa_pos_ffn(evidence_text_qa)
        evidence_qa_text, _ = self.qa_text_attention(evidence_qa_embedding, evidence_text_embedding, evidence_text_embedding)
        evidence_qa_text = self.qa_text_pos_ffn(evidence_qa_text)
        # CT + EQA
        claim_text_evidence_qa, _ = self.text_qa_attention(claim_text_embedding, evidence_qa_embedding, evidence_qa_embedding)
        claim_text_evidence_qa = self.text_qa_pos_ffn(claim_text_evidence_qa)
        evidence_qa_claim_text, _ = self.claim_evidence_qa_text_attention(evidence_qa_embedding, claim_text_embedding, claim_text_embedding)
        evidence_qa_claim_text = self.claim_evidence_text_qa_pos_ffn(evidence_qa_claim_text)
        # CQA + ET
        claim_qa_evidence_text, _ = self.qa_text_attention(claim_qa_embedding, evidence_text_embedding, evidence_text_embedding)
        claim_qa_evidence_text = self.qa_text_pos_ffn(claim_qa_evidence_text)
        evidence_text_claim_qa, _ = self.claim_evidence_text_qa_attention(evidence_text_embedding, claim_qa_embedding, claim_qa_embedding)
        evidence_text_claim_qa = self.claim_evidence_qa_text_pos_ffn(evidence_text_claim_qa)

        # aggregate word and qa embedding to sentence embedding
        # embeddings
        claim_text_embedding = torch.mean(claim_text_embedding, dim=1)
        evidence_text_embedding = torch.mean(evidence_text_embedding, dim=1)
        claim_qa_embedding = torch.mean(claim_qa_embedding, dim=1)
        evidence_qa_embedding = torch.mean(evidence_qa_embedding, dim=1)
        # CT + ET
        claim_evidence_text = torch.mean(claim_evidence_text, dim=1)
        evidence_claim_text = torch.mean(evidence_claim_text, dim=1)
        # CQA + EQA
        claim_evidence_qa = torch.mean(claim_evidence_qa, dim=1)
        evidence_claim_qa = torch.mean(evidence_claim_qa, dim=1)
        # CT + CQA
        claim_text_qa = torch.mean(claim_text_qa, dim=1)
        claim_qa_text = torch.mean(claim_qa_text, dim=1)
        # ET + EQA
        evidence_text_qa = torch.mean(evidence_text_qa, dim=1)
        evidence_qa_text = torch.mean(evidence_qa_text, dim=1)
        # CT + EQA
        claim_text_evidence_qa = torch.mean(claim_text_evidence_qa, dim=1)
        evidence_qa_claim_text = torch.mean(evidence_qa_claim_text, dim=1)
        # CQA + ET
        claim_qa_evidence_text = torch.mean(claim_qa_evidence_text, dim=1)
        evidence_text_claim_qa = torch.mean(evidence_text_claim_qa, dim=1)
        
        # concat together
        concat_text_qa_embeddings = torch.cat((
            claim_text_embedding,
            evidence_text_embedding,
            claim_qa_embedding,
            evidence_qa_embedding,
            claim_evidence_text,
            evidence_claim_text,
            claim_evidence_qa,
            evidence_claim_qa,
            claim_text_qa,
            claim_qa_text,
            evidence_text_qa,
            evidence_qa_text,
            claim_text_evidence_qa,
            evidence_qa_claim_text,
            claim_qa_evidence_text,
            evidence_text_claim_qa
        ), dim=-1)

        text_qa_embeddings = self.attention_fusion(concat_text_qa_embeddings)
        # feature_embeddings = self.feature_embedding(add_feature)

        # concat_embeddings = torch.cat((text_qa_embeddings, feature_embeddings), dim=-1)

        predicted_output = self.classifier(text_qa_embeddings)

        return predicted_output, text_qa_embeddings