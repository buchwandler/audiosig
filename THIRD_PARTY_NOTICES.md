# Third-party notices

AudioSig's core implementation is original work licensed under Apache-2.0.

The silence/VAD module is a clean-room NumPy implementation of the documented
behavior required by PyKokoro. No PyKokoro or librosa source file was available
in this checkout, so no upstream source text or commit identifier was copied.
This statement is retained here to make the provenance boundary explicit for
future migration work. If upstream-derived code is added, its original notice,
copyright, permission terms, and commit reference must be added to this file
and to the relevant source module.

## ESOLA provenance

The ESOLA backend is an original NumPy implementation based on the algorithm
description in S. Rudresh et al., “Epoch-Synchronous Overlap-Add (ESOLA) for
Time- and Pitch-Scale Modification of Speech Signals,” arXiv:1801.06492
(2018), https://arxiv.org/abs/1801.06492. No source code, constants, comments,
or test vectors were copied from the authors’ MATLAB repository or from
unlicensed third-party ESOLA repositories. This notice is provenance guidance,
not a claim about perceptual superiority or legal advice.
