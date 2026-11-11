"""CPU generated-input test; no real data, no clinical accuracy claims."""
import io
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from rehabfm.config import MODALITIES, WINDOW_SEC
from rehabfm.data import tokenize, collate_fn
from rehabfm.model import RehabFM

def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    rows = []
    for i, modalities in enumerate([list(MODALITIES), ['EEG', 'EMG'], ['IMU']]):
        recording = {m: rng.normal(size=(3 if m == 'SKEL' else 2, int(WINDOW_SEC * MODALITIES[m]['canonical_fs'] * 12))).astype(np.float32) for m in modalities}
        tokens, stats, waves = tokenize(recording)
        rows.append(dict(tokens=tokens, stats=stats, waves=waves, y=i % 2, dataset=i, subject=i))
    batch = collate_fn(rows)
    model = RehabFM(n_datasets=48, geometry=True, native=False)
    model.eval()
    output = model(batch)
    assert set(output['z']) == set(MODALITIES)
    for mod, z in output['z'].items():
        assert z.shape == (len(output['owners'][mod]), 64) and torch.isfinite(z).all()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    output = model(batch)
    loss = sum((z - torch.linspace(-1, 1, 64)).square().mean() for z in output['z'].values())
    optimizer.zero_grad(); loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) > 0 for g in gradients)
    optimizer.step()
    assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
    buffer = io.BytesIO()
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, buffer)
    buffer.seek(0); state = torch.load(buffer, map_location='cpu', weights_only=True)
    restored = RehabFM(n_datasets=48, geometry=True, native=False)
    restored.load_state_dict(state['model'], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(state['optimizer'])
    for key, value in model.state_dict().items(): assert torch.equal(value, restored.state_dict()[key])
    for key, item in optimizer.state_dict()['state'].items():
        for field, value in item.items():
            other = restored_optimizer.state_dict()['state'][key][field]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    model.eval(); restored.eval()
    with torch.no_grad():
        left, right = model(batch), restored(batch)
    for mod in left['z']: assert torch.equal(left['z'][mod], right['z'][mod])
    print(json.dumps({'status': 'passed', 'device': 'cpu', 'modalities': len(MODALITIES), 'generated_records': 3, 'optimizer_updates': 1, 'exact_model_optimizer_restore': True, 'real_patient_records': 0}))

if __name__ == '__main__': main()
