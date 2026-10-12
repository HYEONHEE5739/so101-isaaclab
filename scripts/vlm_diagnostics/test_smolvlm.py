#!/usr/bin/env python3
"""Standalone pretrained image-language diagnostic; no LeRobot or robot control.

Existing CUDA PyTorch environment:
  python -m pip install 'transformers==4.57.1' pillow num2words
  python scripts/vlm_diagnostics/test_smolvlm.py --image /path/to/frame.png

Tested API: Transformers 4.57.1, PyTorch 2.7.0. Uses SDPA, not flash-attn.
"""
import argparse
from pathlib import Path

MODEL_ID = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
QUESTIONS = (
    'Describe the objects visible in this robot workspace.',
    'Is there a red cube in the image? If yes, where is it?',
    'Is there a blue cube in the image? If yes, where is it?',
    'How many cups or containers are visible?',
    'Which cup or container is on the left?',
    'Where is the robot gripper relative to the red cube?',
    'If the instruction is "put the red cube into the left cup", which object should the robot manipulate and where should it place it?',
    # Diagnostic only: SmolVLM2 is NOT a dedicated object grounding model.
    # These numbers are unverified model text, NOT reliable detections or robot targets.
    'Locate the red cube and return its bounding box as [x_min, y_min, x_max, y_max].',
    'Describe this photograph in detail. Include the visible objects, their colors and shapes, '
    'and their positions relative to one another from the viewer\'s perspective. '
    'Describe only what you can see, and say when something is unclear.',
)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', required=True, type=Path, help='Local PNG/JPG image; not a dataset directory')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--max-new-tokens', type=positive_int, default=128)
    parser.add_argument('--max-image-side', type=positive_int, default=None,
                        help='Optional resize before processing, preserving aspect ratio (e.g. 512)')
    parser.add_argument('--no-image-splitting', action='store_true', help='Reduce visual tokens/VRAM, potentially lose small-object detail')
    parser.add_argument('--local-files-only', action='store_true', help='Use downloaded Hugging Face cache without network')
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f'Image not found: {args.image}')

    if args.local_files_only:
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, AutoModelForImageTextToText

    with Image.open(args.image) as source:
        original_size = source.size
        image = ImageOps.exif_transpose(source).convert('RGB')
    if args.max_image_side:
        image.thumbnail((args.max_image_side, args.max_image_side), Image.Resampling.LANCZOS)

    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available in this Python environment')
    device = 'cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu'
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == 'cuda' else torch.float32
    print(f'Model: {MODEL_ID}\nTransformers: {transformers.__version__}\nDevice: {device}; dtype: {dtype}', flush=True)
    print(f'Image: {args.image.resolve()}\nOriginal: {original_size}; input RGB: {image.size}', flush=True)
    print('Each question starts a fresh conversation. Bounding-box output is unverified diagnostic text.', flush=True)

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=args.local_files_only)
        if args.no_image_splitting:
            processor.image_processor.do_image_splitting = False
        # Load the original VLM once, not a SmolVLA checkpoint/action expert.
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=dtype, attn_implementation='sdpa', local_files_only=args.local_files_only,
        ).to(device)
        model.eval()
        model.requires_grad_(False)
        with torch.inference_mode():
            for index, question in enumerate(QUESTIONS, 1):
                print('\n' + '=' * 60 + f'\nQ{index}. {question}\n' + '-' * 60, flush=True)
                messages = [{'role': 'user', 'content': [
                    {'type': 'image'}, {'type': 'text', 'text': question},
                ]}]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                # Exactly one image and one question per batch; previous answers aren't reused.
                inputs = processor(text=prompt, images=[image], return_tensors='pt')
                inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                          for key, value in inputs.items()}
                prompt_length = inputs['input_ids'].shape[1]
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                           num_beams=1, use_cache=True)
                answer = processor.batch_decode(generated[:, prompt_length:].cpu(), skip_special_tokens=True)[0].strip()
                print(f'A: {answer}\n' + '=' * 60, flush=True)
                del inputs, generated
    except torch.cuda.OutOfMemoryError:
        raise SystemExit('CUDA OOM: stop other GPU jobs, then try --no-image-splitting --max-image-side 512 '
                         '--max-new-tokens 64 (or --device cpu).')


if __name__ == '__main__':
    main()
