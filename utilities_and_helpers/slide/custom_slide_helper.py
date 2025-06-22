from fastai.vision.all import *

'''
Custom slide helper functions adapted from the original slide_helper.py by Wilm et al. (2023).

Main adaptation: patch retrieval now a precomputed patch grid (by index)
instead of random sampling. Other core visualization and normalization logic is retained.
'''

# Define colors for visualization
COLORS = np.array([[128, 128, 128],  # Excluded
                   [255, 255, 255],  # BG
                   [0, 0, 255],      # Normal
                   [255, 128, 0]])   # Tumor

# Normalize image pixels to range [0, 1]
class MinMaxNormalize(Transform):
    def __init__(self, min_val=0, max_val=255):
        self.min_val = min_val
        self.max_val = max_val

    def encodes(self, img: TensorImage):
        return (img - self.min_val) / (self.max_val - self.min_val)

# Custom DataLoader with show_results() override
class customDataLoader(TfmdDL):
    def show_results(self, b, out, max_n=9, ctxs=None, show=True, **kwargs):
        x, y, its = self.show_batch(b, max_n=max_n, show=False)
        seg_out = torch.max(out, dim=1)[1] + 1
        b_out = type(b)(b[:self.n_inp] + (seg_out,))

        _types = self._types[first(self._types.keys())]
        b_out = tuple([cast(x, typ) for x, typ in zip(b_out, _types)])
        b_out = to_cpu(self.after_batch.decode(b_out))

        if not show:
            return (x, y, its, b_out[self.n_inp:])
        show_results(b_out[:self.n_inp], b_out[self.n_inp:], ctxs=ctxs, max_n=max_n, **kwargs)

# Fetch a patch from the precomputed grid
def get_item(slide_container, patch_idx):
    x, y = slide_container.patch_grid[patch_idx]
    patch = slide_container.get_patch(x, y)
    y_patch = slide_container.get_y_patch(x, y)
    return (patch, y_patch)

@typedispatch
def show_batch(x, y, samples, ctxs=None, max_n=6, nrows=None, ncols=1, figsize=None, **kwargs):
    if figsize is None: 
        figsize = (ncols * 12, min(len(x), max_n) * 3)
    if ctxs is None: 
        ctxs = get_grid(min(len(x), max_n), nrows=min(len(x), max_n), ncols=ncols, figsize=figsize)

    for i, ctx in enumerate(ctxs):
        image = tensor(x[i])
        line = image.new_zeros(image.shape[0], image.shape[1], 5)
        mask = y[i]

        overlay = tensor(np.asarray([COLORS[c] for c in np.int16(mask.flatten())], dtype=np.uint8)
                         .reshape((mask.shape[0], mask.shape[1], -1))).permute(2, 0, 1)
        show_image(torch.cat([image, line, overlay], dim=2), ctx=ctx, **kwargs)

