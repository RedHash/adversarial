import torch
import torch.nn as nn

from scipy.io.wavfile import write


class KSStrong(nn.Module):
    sample_rate = 16000  # Speech dataset sample rate

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fr_s = torch.nn.Parameter(torch.tensor([220, 220 * 5 // 4, 220 * 6 // 4, 220 * 2],
                                                    dtype=torch.float))
        self.vol_s = torch.nn.Parameter(torch.tensor([0.5 for _ in range(4)],
                                                     dtype=torch.float))

    def forward(self, n_samples):
        wave = torch.zeros(n_samples)
        if not self.config['no_cuda']:
            wave = wave.cuda()

        for i in range(len(self.fr_s)):
            wave += self.generate(self.fr_s[i], self.vol_s[i], n_samples)

        return wave

    def generate(self, fr_i, vol_i, nsamples):
        delay_period = int((self.sample_rate // fr_i).item())
        buf = torch.rand(delay_period) * 2 - 1
        samples = torch.empty(nsamples, dtype=torch.float)

        if not self.config['no_cuda']:
            buf = buf.cuda()
            samples = samples.cuda()

        gamma = (4 / torch.log(fr_i)) ** (1 / delay_period)

        for i in range(nsamples):
            samples[i] = buf[i % delay_period]
            buf[i % delay_period] = gamma * (buf[i % delay_period] + buf[(1 + i) % delay_period]) / 2

        return samples * vol_i


if __name__ == "__main__":
    ks = KSStrong({'no_cuda': True})
    samples = ks.sample_rate * 5
    wave = ks(samples)

    loss = torch.nn.MSELoss()(wave, torch.zeros_like(wave))
    loss.backward()

    assert loss.item() != 0

    assert torch.sum(ks.fr_s.grad) != 0
    assert torch.sum(ks.vol_s.grad) != 0

    wave = wave.detach().cpu().numpy()
    write('test.wav', ks.sample_rate, wave)
