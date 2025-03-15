import argparse
from functools import reduce
from pathlib import Path
import torch


class Renderer:
    def __init__(self, args, exp_name, load_flame=True, load_renderer=True):
        self.args = args
        self.load_flame = load_flame
        self.load_renderer = load_renderer
        self.no_context_audio_feat = args.no_context_audio_feat
        self.device = torch.device(args.device)

        # data loader
        self.dataloader = None

        # FLAME model
        if self.load_flame:
            from models.flame import FLAME, FLAMEConfig
            self.flame = FLAME(FLAMEConfig)
            self.flame.to(self.device)
            self.flame.eval()

        self.default_output_dir = Path('demo/output') / exp_name  # run_id

    def params_to_vertices(self, coef_dict, flame, rot_repr='aa', ignore_global_rot=False, flame_batch_size=512):
        shape = coef_dict['exp'].shape[:-1]
        coef_dict = {k: v.view(-1, v.shape[-1]) for k, v in coef_dict.items()}
        n_samples = reduce(lambda x, y: x * y, shape, 1)

        # Convert to vertices
        vert_list = []
        for i in range(0, n_samples, flame_batch_size):
            batch_coef_dict = {k: v[i:i + flame_batch_size] for k, v in coef_dict.items()}
            if rot_repr == 'aa':
                vert, _, _ = flame(
                    batch_coef_dict['shape'], batch_coef_dict['exp'], batch_coef_dict['pose'],
                    pose2rot=True, ignore_global_rot=ignore_global_rot, return_lm2d=False, return_lm3d=False)
            else:
                raise ValueError(f'Unknown rot_repr: {rot_repr}')
            vert_list.append(vert)

        vert_list = torch.cat(vert_list, dim=0)  # (n_samples, 5023, 3)
        vert_list = vert_list.view(*shape, -1, 3)  # (..., 5023, 3)

        return vert_list

    def forward(self, params, ignore_global_rot=False):
        # shape; exp; pose
        coef_dict = NotImplemented

        assert self.load_flame, 'FLAME model is not loaded'
        verts_list = self.coef_dict_to_vertices(coef_dict, self.flame, self.rot_repr,
                                                ignore_global_rot=ignore_global_rot).detach().cpu().numpy()

    @torch.no_grad()
    def infer_params(self):
        pass


def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render FLAME Params"
    )

    parser.add_argument("--flame_params_path", type=str, required=True, help="Path to FLAME params")

    args = parser.parse_args()
    main(args)
