"""
Export a trained PyTorch OthelloNNet checkpoint to ONNX so the C side can run
it via ONNX Runtime.

The exported graph has:
  - Input  "board"  : float32, shape (B, 1, N, N), B is dynamic.
  - Output "policy" : float32, shape (B, action_size), log-softmax.
  - Output "value"  : float32, shape (B, 1), tanh.

Usage:
    python python/export_onnx.py --pth /path/to/best.pth.tar \
                                 --out /path/to/best.onnx \
                                 [--n 6]

To export a fresh random-init network for parity testing (no checkpoint
required), use --no-pth.
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

from othello.OthelloGame import OthelloGame  # noqa: E402
from othello.pytorch.NNet import NNetWrapper  # noqa: E402


class _ExportWrapper(torch.nn.Module):
    """Wrap OthelloNNet so the input is (B, 1, N, N) instead of (B, N, N)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        # x: (B, 1, N, N) -> (B, N, N)
        x = x.squeeze(1)
        return self.inner(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', help='input PyTorch checkpoint (.pth.tar)')
    ap.add_argument('--no-pth', action='store_true',
                    help='export the random-init network (for parity testing)')
    ap.add_argument('--out', required=True, help='output .onnx path')
    ap.add_argument('--n', type=int, default=6, help='Othello board size')
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    if not args.no_pth and not args.pth:
        ap.error('either --pth FILE or --no-pth must be given')

    game = OthelloGame(args.n)
    wrapper = NNetWrapper(game)
    if args.pth:
        folder, fname = os.path.split(args.pth)
        wrapper.load_checkpoint(folder=folder or '.', filename=fname)
    else:
        # --no-pth: also persist the freshly-initialised weights so callers
        # (e.g. verify_onnx_parity.py) can compare against the same model.
        sidecar_pth = args.out + '.pth.tar'
        out_dir = os.path.dirname(sidecar_pth) or '.'
        out_name = os.path.basename(sidecar_pth)
        wrapper.save_checkpoint(folder=out_dir, filename=out_name)

    inner = wrapper.nnet
    inner.eval()

    export_model = _ExportWrapper(inner).eval()

    # Sample input: (1, 1, N, N) float32. The dynamic_axes mark batch as variable.
    sample = torch.zeros(1, 1, args.n, args.n, dtype=torch.float32)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    torch.onnx.export(
        export_model,
        sample,
        args.out,
        input_names=['board'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'board':  {0: 'batch'},
            'policy': {0: 'batch'},
            'value':  {0: 'batch'},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # Newer torch.onnx writes 9.7M floats to a sidecar .data file by default.
    # Re-save as a single self-contained file so the C side has one path to load.
    import onnx
    sidecar = args.out + '.data'
    if os.path.exists(sidecar):
        model = onnx.load(args.out, load_external_data=True)
        # Strip the external_data references so the next save embeds in-place.
        for tensor in model.graph.initializer:
            if tensor.HasField('data_location') and tensor.data_location == onnx.TensorProto.EXTERNAL:
                tensor.ClearField('external_data')
                tensor.data_location = onnx.TensorProto.DEFAULT
        onnx.save(model, args.out)
        os.remove(sidecar)

    print(f"exported {args.out}")


if __name__ == '__main__':
    main()
