# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
        symmetric_masking=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self.symmetric_masking = symmetric_masking  # exclude mirror blocks for antisymmetric inputs (e.g. GADF)
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _mirror_complement(self, top, left, h, w):
        """Complement mask for the mirror block (left, top, w, h).

        GADF is antisymmetric: GADF[i,j] = -GADF[j,i]. The mirror block
        contains negated but equivalent information, so the encoder should
        not see it when predicting the original block.
        """
        m_top, m_left, m_h, m_w = left, top, w, h
        # Clamp to grid bounds
        m_top = min(m_top, self.height - 1)
        m_left = min(m_left, self.width - 1)
        m_h = min(m_h, self.height - m_top)
        m_w = min(m_w, self.width - m_left)
        complement = torch.ones((self.height, self.width), dtype=torch.int32)
        complement[m_top:m_top+m_h, m_left:m_left+m_w] = 0
        return complement

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement, (int(top), int(left), h, w)

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C, block_coords = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                if self.symmetric_masking:
                    top, left, h, w = block_coords
                    masks_C.append(self._mirror_complement(top, left, h, w))
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred


if __name__ == '__main__':
    # Test: symmetric_masking excluye bloques espejo del encoder
    PATCH_SIZE = 16
    INPUT_SIZE = (224, 224)
    H = W = 224 // PATCH_SIZE  # 14

    collator = MaskCollator(
        input_size=INPUT_SIZE,
        patch_size=PATCH_SIZE,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
        symmetric_masking=True)

    # Batch sintético: 4 imágenes de 1 canal
    batch = [(torch.randn(1, 224, 224), 0) for _ in range(4)]

    all_pass = True
    for trial in range(5):
        _, masks_enc, masks_pred = collator(batch)
        B = masks_enc[0].shape[0]

        for b in range(B):
            enc_indices = set(masks_enc[0][b].tolist())

            for p in range(len(masks_pred)):
                pred_indices = masks_pred[p][b].tolist()
                # Reconstruir posiciones 2D de los patches predictor
                pred_2d = [(idx // W, idx % W) for idx in pred_indices]
                # Calcular posiciones espejo
                mirror_2d = [(col, row) for row, col in pred_2d]
                mirror_indices = set(row * W + col for row, col in mirror_2d)

                overlap = enc_indices & mirror_indices
                if overlap:
                    print(f"  FAIL trial={trial} img={b} pred_block={p}: "
                          f"{len(overlap)} mirror patches in encoder")
                    all_pass = False

    if all_pass:
        print("PASS: ningún bloque espejo aparece en el encoder (5 trials x 4 imgs x 4 pred blocks)")
    else:
        print("FAIL: se encontraron bloques espejo en el encoder")
