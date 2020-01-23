import torch
import fairseq.utils
from fairseq.models import transformer, register_model, register_model_architecture


@register_model('conquest')
class ConquestModel(transformer.TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return ConquestDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )


class ConquestDecoder(transformer.TransformerDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left2right = True

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        assert dim >= 2
        res = tensor.new(dim, dim)
        fairseq.utils.fill_with_neg_inf(res[0][1:])
        x = fairseq.utils.fill_with_neg_inf(tensor.new(dim-1, dim-1))
        if self.left2right:
            x = torch.triu(x, 1)
        else:
            x = torch.tril(x, -1)
            fairseq.utils.fill_with_neg_inf(x[0])
        res[1:, 1:].copy_(x)
        res[:, 0] = 0
        return res

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        full_context_alignment=False,
        alignment_layer=None,
        alignment_heads=None,
        **unused,
    ):
        self.left2right = True
        l2r, _ = super().extract_features(prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer, alignment_heads=alignment_heads)

        self.left2right = False
        r2l, _ = super().extract_features(prev_output_tokens,
              encoder_out=encoder_out,
              incremental_state=incremental_state,
              full_context_alignment=full_context_alignment,
              alignment_layer=alignment_layer,
              alignment_heads=alignment_heads)

        batch_size, tgt_lengths, dim = r2l.size()
        padding = prev_output_tokens.eq(self.dictionary.pad())
        if padding.any():
            padding = padding.view(batch_size, tgt_lengths, 1).repeat(1, 1, dim)
            r2l = torch.masked_fill(r2l, padding, 0)

        l2r[:, :-2] += r2l[:, 2:]
        return l2r[:, :-1], {}


@register_model_architecture('conquest', 'conquest_base')
def base_architecture(args):
    transformer.base_architecture(args)


@register_model_architecture('conquest', 'conquest_iwslt_de_en')
def conquest_iwslt_de_en(args):
    transformer.transformer_iwslt_de_en(args)


@register_model_architecture('conquest', 'conquest_wmt_en_de_big')
def conquest_wmt_en_de_big(args):
    transformer.transformer_wmt_en_de_big(args)


@register_model_architecture('conquest', 'conquest_vaswani_wmt_en_fr_big')
def conquest_vaswani_wmt_en_fr_big(args):
    transformer.transformer_vaswani_wmt_en_fr_big(args)
