# Derived unchanged from TITL training/export.py, Apache-2.0; see LICENSE/NOTICE.
import torch
from torch import nn
from torch.nn import functional as F

class ConvSTFT(nn.Module):
    """An exportable STFT that matches torch.stft / torch.istft, which the decoder was trained with. Kokoro's
    CustomSTFT differs from them: replicate instead of reflect padding, float noise in the DC and Nyquist imaginary
    parts that flips their phase between +pi and -pi, and an inverse that neither halves DC/Nyquist nor divides by
    the window overlap. Upstream Paradee measured about 0.5 UTMOS lost with it."""

    def __init__(self, n, hop):
        super().__init__()
        self.n, self.hop = n, hop
        w = torch.hann_window(n)
        k = torch.arange(n // 2 + 1).float()[:, None]
        ang = 2 * torch.pi * k * torch.arange(n).float() / n
        re, im = torch.cos(ang) * w, -torch.sin(ang) * w
        im[0] = 0
        im[-1] = 0
        self.register_buffer("fwd", torch.cat([re, im])[:, None])
        c = torch.full((n // 2 + 1, 1), 2.0)
        c[0] = c[-1] = 1
        self.register_buffer("inv", (torch.cat([torch.cos(ang) * c, -torch.sin(ang) * c]) / n * w)[:, None])
        self.register_buffer("w2", (w**2)[None, None])

    def transform(self, x):
        y = F.conv1d(F.pad(x[:, None], (self.n // 2, self.n // 2), mode="reflect"), self.fwd, stride=self.hop)
        K = self.n // 2 + 1
        re, im = y[:, :K], y[:, K:]
        ph = torch.where((im == 0) & (re < 0), torch.full_like(re, torch.pi), torch.atan2(im, re))  # torch.angle gives +pi there
        return torch.sqrt(re**2 + im**2), ph

    def inverse(self, mag, phase):
        z = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], 1)
        y = F.conv_transpose1d(z, self.inv, stride=self.hop)[:, 0]
        env = F.conv_transpose1d(torch.ones(1, 1, mag.shape[-1]), self.w2, stride=self.hop)[:, 0]
        return (y / env.clamp_min(1e-8))[:, self.n // 2 : -(self.n // 2)][:, None]

class Speaker(nn.Module):
    """Text student + decoder for one unpadded sentence, written without masks or packed sequences."""

    def __init__(self, ts, ds):
        super().__init__()
        self.ts, self.ds = ts, ds

    def text(self, ids):
        t = self.ts
        p, s = t.predictor, t.style
        x = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
        sx = s[:, :, None].expand(-1, -1, ids.shape[1])
        x = torch.cat([x, sx], 1)
        for b in p.text_encoder.lstms:
            if isinstance(b, nn.LSTM):
                x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
            else:
                x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
        d = x.transpose(-1, -2)  # [1, T, hidden + style]
        dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]  # [T]
        e = t.text_encoder
        h = e.embedding(ids).transpose(1, 2)
        for c in e.cnn:
            h = c(h)
        asr_tok = t.asr_proj(e.lstm(h.transpose(1, 2))[0]).transpose(-1, -2)  # [1, 512, T]
        return d, dur, asr_tok

    def forward(self, input_ids, speed):
        d, dur, asr_tok = self.text(input_ids)
        n = torch.clamp(torch.round(dur / speed), min=1).long()
        end = torch.cumsum(n, 0)
        start = end - n
        f = torch.arange(end[-1])[None]
        aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]  # [1, T, F]
        F0, N = self.ts.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, self.ts.style)
        return self.ds(asr_tok @ aln, F0, N).squeeze(1)
