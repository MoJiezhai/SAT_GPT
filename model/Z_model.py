import torch
from torch import nn
import torch.nn.functional as F


class CompositionDecoder(nn.Module):

    def __init__(
        self,
        num_elements=118,
        max_atoms=52,
        num_cond=0,
        d_model=256,
        nhead=8,
        num_layers=8,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()

        # ----------------------------
        # token definition
        # ----------------------------
        self.PAD = 0

        # 1 ~ 118 : elements
        self.num_elements = num_elements

        # special tokens
        self.BOS = num_elements + 1
        self.EOS = num_elements + 2

        self.vocab_size = num_elements + 3

        self.max_atoms = max_atoms
        self.num_cond = num_cond
        
        self.d_model = d_model
        self.nhead = nhead

        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # ----------------------------
        # embeddings
        # ----------------------------
        self.element_embed = nn.Embedding(
            self.vocab_size,
            d_model,
            padding_idx=self.PAD
        )

        if num_cond > 0:
            self.cond_embed = nn.Linear(1, d_model)

        # token type:
        # 0 = cond
        # 1 = sequence token
        self.type_embed = nn.Embedding(2, d_model)

        # IMPORTANT:
        # positional embedding
        self.pos_embed = nn.Embedding(
            max_atoms + num_cond + 8,
            d_model
        )

        # ----------------------------
        # transformer
        # ----------------------------
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )

        # ----------------------------
        # output head
        # ----------------------------
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.vocab_size)
        )

    def build_causal_mask(self, T, device):

        mask = torch.triu(
            torch.ones(T, T, device=device),
            diagonal=1
        )

        return mask.bool()

    def forward(
        self,
        Z,
        cond=None,
        return_loss=False,
    ):
        """
        Z:
            [B, T]

        Example training sequence:

            input:
                BOS Li O O EOS PAD

            target:
                Li O O EOS PAD

        cond:
            [B, num_cond]

        returns:
            logits: [B, T, vocab_size]
        """

        B, T = Z.shape
        device = Z.device

        # -----------------------------------
        # token embedding
        # -----------------------------------
        atom_emb = self.element_embed(Z)

        atom_type_ids = torch.ones(
            (B, T),
            device=device,
            dtype=torch.long
        )

        atom_emb = atom_emb + self.type_embed(atom_type_ids)

        tokens = []
        num_prefix = 0

        # -----------------------------------
        # condition prefix
        # -----------------------------------
        if self.num_cond > 0 and cond is not None:

            cond = cond.unsqueeze(-1)

            cond_emb = self.cond_embed(cond)

            cond_type_ids = torch.zeros(
                (B, cond_emb.size(1)),
                device=device,
                dtype=torch.long
            )

            cond_emb = cond_emb + self.type_embed(cond_type_ids)

            tokens.append(cond_emb)

            num_prefix += cond_emb.size(1)

        # -----------------------------------
        # append sequence
        # -----------------------------------
        tokens.append(atom_emb)

        x = torch.cat(tokens, dim=1)

        total_len = x.size(1)

        # -----------------------------------
        # positional embedding
        # -----------------------------------
        pos_ids = torch.arange(
            total_len,
            device=device
        ).unsqueeze(0)

        x = x + self.pos_embed(pos_ids)

        # -----------------------------------
        # masks
        # -----------------------------------
        causal_mask = self.build_causal_mask(
            total_len,
            device
        )

        atom_padding = (Z == self.PAD)

        if num_prefix > 0:

            prefix_padding = torch.zeros(
                B,
                num_prefix,
                device=device,
                dtype=torch.bool
            )

            key_padding_mask = torch.cat(
                [prefix_padding, atom_padding],
                dim=1
            )

        else:

            key_padding_mask = atom_padding

        # -----------------------------------
        # transformer
        # -----------------------------------
        x = self.decoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )

        x = x[:, num_prefix:]

        logits = self.head(x)

        if not return_loss:
            return logits

        # -----------------------------------
        # autoregressive loss
        # -----------------------------------
        pred = logits[:, :-1]
        target = Z[:, 1:]

        loss = F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),
            target.reshape(-1),
            ignore_index=self.PAD
        )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        batch_size=1,
        cond=None,
        temperature=1.0,
        top_k=5,
        max_len=None
    ):

        if cond is not None:
            device = cond.device
            B = cond.size(0)
        else:
            device = next(self.parameters()).device
            B = batch_size

        if max_len is None:
            max_len = self.max_atoms

        # -----------------------------------
        # initialize with BOS
        # -----------------------------------
        Z = torch.full(
            (B, 1),
            self.BOS,
            device=device,
            dtype=torch.long
        )

        finished = torch.zeros(
            B,
            device=device,
            dtype=torch.bool
        )

        # -----------------------------------
        # autoregressive generation
        # -----------------------------------
        for step in range(max_len):

            logits = self.forward(
                Z=Z,
                cond=cond
            )

            next_logits = logits[:, -1]

            # forbid PAD/BOS
            next_logits[:, self.PAD] = -1e9
            next_logits[:, self.BOS] = -1e9

            # forbid EOS at first step
            if step == 0:
                next_logits[:, self.EOS] = -1e9

            # temperature
            next_logits = next_logits / temperature

            # top-k sampling
            if top_k is not None:

                values, indices = torch.topk(
                    next_logits,
                    top_k,
                    dim=-1
                )

                probs = F.softmax(values, dim=-1)

                sample = torch.multinomial(
                    probs,
                    1
                )

                next_token = indices.gather(
                    -1,
                    sample
                )

            else:

                probs = F.softmax(
                    next_logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    probs,
                    1
                )

            # already finished -> PAD
            next_token[finished] = self.PAD

            # append
            Z = torch.cat(
                [Z, next_token],
                dim=1
            )

            # update finished
            finished |= (
                next_token.squeeze(-1) == self.EOS
            )

            # all finished
            if finished.all():
                break

        # -----------------------------------
        # remove BOS
        # -----------------------------------
        return Z


import torch
from torch import nn
import torch.nn.functional as F


class CompositionPredictor(nn.Module):
    """
    Composition generator with:

        1. total atom prediction
        2. element distribution prediction

    generation:

        N_atoms ~ predicted
        counts  ~ Multinomial(N_atoms, p(z))

    advantages:

        - stable
        - permutation invariant
        - exact atom count control
        - no EOS needed
        - physically meaningful
    """

    def __init__(
        self,
        num_elements=118,
        num_cond=0,
        d_model=256,
        hidden_dim=512,
        max_atoms=20,
        dropout=0.1,
    ):
        super().__init__()

        self.num_elements = num_elements
        self.num_cond = num_cond
        self.max_atoms = max_atoms

        # =================================================
        # embeddings
        # =================================================

        self.element_embed = nn.Embedding(
            num_elements + 1,
            d_model,
            padding_idx=0
        )

        # =================================================
        # condition encoder
        # =================================================

        if num_cond > 0:

            self.cond_proj = nn.Sequential(
                nn.Linear(num_cond, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        # =================================================
        # backbone
        # =================================================

        self.encoder = nn.Sequential(

            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, d_model),
        )

        # =================================================
        # heads
        # =================================================

        # total atom number
        self.num_head = nn.Linear(
            d_model,
            max_atoms + 1
        )

        # element distribution
        self.comp_head = nn.Linear(
            d_model,
            num_elements
        )
        
        self.latent_dim = 4

        self.latent_encoder = nn.Linear(
            self.num_elements,
            self.latent_dim
        )

        self.latent_proj = nn.Linear(
    self.num_elements,
    d_model
)

    # =====================================================
    # target construction
    # =====================================================

    def build_targets(self, Z):
        """
        Z:
            [B, L]

        returns:

            counts:
                [B, num_elements]

            total_atoms:
                [B]
        """

        B, L = Z.shape
        device = Z.device

        counts = torch.zeros(
            B,
            self.num_elements,
            device=device,
            dtype=torch.float
        )

        for z in range(1, self.num_elements + 1):

            counts[:, z - 1] = (
                (Z == z)
                .sum(dim=1)
                .float()
            )

        total_atoms = counts.sum(dim=1)

        return counts, total_atoms
    # =====================================================
    # forward
    # =====================================================

    def infer_latent(self, Z):

        B, L = Z.shape
        device = Z.device

        counts = torch.zeros(
            B,
            self.num_elements,
            device=device
        )

        for z in range(1, self.num_elements + 1):

            counts[:, z - 1] = (
                (Z == z)
                .sum(dim=1)
                .float()
            )

        probs = counts / counts.sum(
            dim=1,
            keepdim=True
        ).clamp(min=1)

        return probs

    def forward(
        self,
        Z,
        cond=None,
        latent=None,
        return_loss=False,
    ):

        B, L = Z.shape


        emb = self.element_embed(Z)

        mask = (Z != 0).float().unsqueeze(-1)

        pooled = (
            emb * mask
        ).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if latent is None and Z is not None:

            latent = self.infer_latent(Z)

        # -------------------------------------------------
        # latent conditioning
        # -------------------------------------------------

        if latent is not None:

            latent_feat = self.latent_proj(latent)

            pooled = pooled + latent_feat

        if self.num_cond > 0 and cond is not None:

            cond_feat = self.cond_proj(cond)

            pooled = pooled + cond_feat


        h = self.encoder(pooled)


        num_logits = self.num_head(h)

        comp_logits = self.comp_head(h)

        comp_probs = F.softmax(
            comp_logits,
            dim=-1
        )

        if not return_loss:

            return {
                "num_logits": num_logits,
                "comp_probs": comp_probs,
            }


        target_counts, target_total = self.build_targets(Z)

        # composition distribution target
        target_probs = (
            target_counts
            / target_total.unsqueeze(-1).clamp(min=1)
        )


        loss_num = F.cross_entropy(
            num_logits,
            target_total.long()
        )


        loss_comp = F.kl_div(
            torch.log(
                comp_probs.clamp(min=1e-8)
            ),
            target_probs,
            reduction='batchmean'
        )

        loss = loss_num + loss_comp

        return {
            "num_logits": num_logits,
            "comp_probs": comp_probs,
            "loss_num": loss_num,
            "loss_comp": loss_comp,
            "loss": loss,
        }

    # =====================================================
    # generate
    # =====================================================

    @torch.no_grad()
    def generate(
        self,
        cond=None,
        batch_size=1,
        temperature=1.0,
    ):
        """
        returns:

            Z_gen:
                [B, L]

            lengths:
                [B]
        """

        if cond is not None:

            device = cond.device
            B = cond.size(0)

        else:

            device = next(self.parameters()).device
            B = batch_size

        latent = torch.rand(
    B,
    self.num_elements,
    device=device
)

        latent = latent / latent.sum(
            dim=-1,
            keepdim=True
        )
        # -------------------------------------------------
        # dummy input
        # -------------------------------------------------

        dummy_Z = torch.zeros(
            B,
            1,
            device=device,
            dtype=torch.long
        )

        outputs = self.forward(
            dummy_Z,
            cond=cond,
            latent=latent,
            return_loss=False
        )

        num_logits = outputs["num_logits"]

        comp_probs = outputs["comp_probs"]

        # =================================================
        # sample atom number
        # =================================================

        num_probs = F.softmax(
            num_logits / temperature,
            dim=-1
        )

        lengths = torch.multinomial(
            num_probs,
            1
        ).squeeze(-1)

        # avoid empty structures
        lengths = torch.clamp(
            lengths,
            min=1
        )

        # =================================================
        # sample composition
        # =================================================

        sequences = []

        max_len = lengths.max().item()

        for b in range(B):

            N = lengths[b].item()

            probs = comp_probs[b]

            # multinomial sampling
            sampled = torch.multinomial(
                probs,
                N,
                replacement=True
            )

            # convert to atomic numbers
            atoms = sampled + 1

            # optional shuffle
            perm = torch.randperm(
                N,
                device=device
            )

            atoms = atoms[perm]

            sequences.append(atoms)

        # =================================================
        # padding
        # =================================================

        Z_gen = torch.zeros(
            B,
            max_len,
            device=device,
            dtype=torch.long
        )

        for b, seq in enumerate(sequences):

            Z_gen[b, :len(seq)] = seq

        return Z_gen, lengths