import sys
sys.path.insert(0, "/data/inswapper")

from restoration import *
from swapper import restore_face

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                 codebook_size=1024,
                                                 n_head=8,
                                                 n_layers=9,
                                                 connect_list=["32", "64", "128", "256"],
                                                 ).to(device)
ckpt_path = "/data/CodeFormer/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()


class CodeFormerFaceEnhancement:
    def __init__(self,
                 path_to_enhance=None,
                 size=512,
                 batch_size=1
                 ):
        print("CodeFormerFaceEnhancement initialized!")
        super(CodeFormerFaceEnhancement, self).__init__()
        self.source_face = None

    def enhance(self, ff, has_aligned=False, only_center_face=True, paste_back=True):
        return None, None, restore_face(ff, codeformer_fidelity=0.5, background_enhance=False, face_upsample=1,
                            upsampler=None, upscale=1, device=device, codeformer_net=codeformer_net)

    def enhance_from_image(self, img, resize_factor=2, bg_upsampler=None):
        return restore_face(img, codeformer_fidelity=0.5, background_enhance=False, face_upsample=1,
                            upsampler=None, upscale=1, device=device, codeformer_net=codeformer_net)

    def set_id_image(self, img):
        self.source_face = get_one_face(face_analyser, img)

    def swap_single(self, f):
        if self.source_face is None:
            return f
        target_face = get_one_face(face_analyser, f)
        return face_swapper.get(f, target_face, self.source_face, paste_back=True)
