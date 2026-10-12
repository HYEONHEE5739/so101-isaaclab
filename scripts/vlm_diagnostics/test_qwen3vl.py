#!/usr/bin/env python3
"""Original Qwen3-VL image understanding only; no training or robot control.

python test_qwen3vl.py --image frame.png [--load-in-4bit]
Existing Transformers 4.57.1 supports this model; do not upgrade lerobot-arena.
Optional 4-bit requires bitsandbytes and accelerate (see README).
"""
import argparse
import json
import math
import importlib.util
import os
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-VL-2B-Instruct'
QUESTIONS = (
    '''Carefully inspect this robot workspace.
List every distinct object that you can identify.
For each object, provide:
1. object type
2. color
3. approximate image position: left / center / right
4. approximate vertical position: top / middle / bottom
Do not guess an object if you are uncertain.''',
    'List all cubes or block-shaped objects visible in the image. For each one, give its color.',
    'List all cups or containers visible in the image. For each one, give its color and approximate position.',
    'Locate the robot gripper and describe its approximate position in the image.',
    'What is to the left of the red cube?',
    'What is to the right of the red cube?',
    'Is the red cube on the left, center, or right side of the image?',
    'Where is the robot gripper relative to the red cube? Use relations such as left, right, above, below, near, or far.',
    'Which object is closest to the robot gripper?',
    'Is the red cube closer to the left container or the right container?',
    '''The robot receives this command:
"Put the red cube into the left cup."
Based only on the image and instruction:
1. Which object is the manipulation target?
2. Which object is the destination?
3. Describe their approximate positions.
4. Do not describe robot joint actions.''',
)
GUIDANCE = ('Answer concisely and clearly, using the image viewer\'s perspective. '
            'For distances, describe apparent image-space proximity, not physical 3D distance. '
            'If an object is absent or uncertain, say so.')


BOX_PROMPT = """Inspect the image and locate every distinct visible object you can identify.
Give a separate tight bounding box for each object instance; do not merge separate objects.
Do not invent hidden objects. Ignore viewer UI, text overlays and the image border.
Return ONLY a JSON array of objects with keys "label" (short English name including
the observed color, e.g. "red cube", "green cube", "blue cube", "white cup") and
"bbox" ([x_min, y_min, x_max, y_max]). Use normalized integer coordinates from 0 to 1000:
(0,0) is the top-left and (1000,1000) is the bottom-right of the entire input image.
Determine each object's color from this image, not from the examples above.
Do not use generic numbered labels such as "cube 1" when a color is visible.
If the color is uncertain, use "unknown-color" rather than guessing.
For objects of the same type and color, add a spatial qualifier such as "left white cup".
If nothing is identifiable, return []."""


def save_boxes(answer, image, destination):
    """VLM estimates only, not verified detections. Never overwrite the input."""
    from PIL import ImageDraw
    raw_path = destination.with_suffix('.response.txt')
    raw_path.write_text(answer, encoding='utf-8')
    text = answer.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        objects = json.loads(text)
        if not isinstance(objects, list):
            raise ValueError('Expected a JSON array')
    except (ValueError, IndexError) as exc:
        raise ValueError(f'Invalid/truncated box JSON; raw response: {raw_path}. '
                         'Try increasing --bbox-max-new-tokens.') from exc
    canvas = image.copy(); draw = ImageDraw.Draw(canvas)
    accepted, rejected = [], []
    for item in objects:
        if not isinstance(item, dict):
            rejected.append(item); continue
        box = item.get('bbox'); label = item.get('label')
        if (not isinstance(label, str) or not label.strip() or not isinstance(box, list)
            or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
            or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)):
            rejected.append(item); continue
        pixels = [round(box[0]* (image.width-1)/1000), round(box[1]*(image.height-1)/1000),
                  round(box[2]*(image.width-1)/1000), round(box[3]*(image.height-1)/1000)]
        if pixels[0] >= pixels[2] or pixels[1] >= pixels[3]:
            rejected.append(item); continue
        color = ['#ff6464', '#55ddaa', '#66aaff', '#ffcc55', '#cc88ff'][len(accepted)%5]
        draw.rectangle(pixels, outline=color, width=3)
        caption = f'{len(accepted)+1}: {label}'.encode('ascii', 'replace').decode()
        draw.text((pixels[0]+3, max(0,pixels[1]-12)), caption, fill=color)
        accepted.append(dict(label=label, bbox_normalized=box, bbox_pixels=pixels))
    canvas.save(destination)
    destination.with_suffix('.json').write_text(json.dumps(dict(
        model=MODEL_ID, verified=False, image_size=list(image.size),
        coordinate_system='normalized_0_1000', objects=accepted, rejected=rejected),
        indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Boxes: {len(accepted)} accepted, {len(rejected)} rejected → {destination}')


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('Must be positive')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--max-new-tokens', type=positive, default=256)
    parser.add_argument('--max-pixels', type=positive, default=512*512,
                        help='Processor image pixel budget; default 262144')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--all-objects', action='store_true', help='Only detect/box all identified objects')
    parser.add_argument('--output', type=Path, help='New annotated PNG path (with --all-objects)')
    parser.add_argument('--bbox-max-new-tokens', type=positive, default=1024)
    args = parser.parse_args()
    if args.output and not args.all_objects:
        parser.error('--output requires --all-objects')
    if args.all_objects:
        args.output = args.output or args.image.with_name(args.image.stem + '_qwen_boxes.png')
        if args.output.suffix.lower() != '.png':
            parser.error('--output must end in .png')
        for path in (args.output, args.output.with_suffix('.json'), args.output.with_suffix('.response.txt')):
            if path.exists() or path.resolve() == args.image.resolve():
                parser.error(f'Output already exists or equals input: {path}; choose a new --output')
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.image.is_file():
        parser.error(f'Image not found: {args.image}')
    if args.max_pixels < 32*32:
        parser.error('--max-pixels must be at least 1024')
    if args.local_files_only:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import transformers
    from PIL import Image, ImageOps
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit('Qwen3-VL API unavailable. Use a separate environment with transformers==4.57.1; '
                         'do not upgrade your working LeRobot environment blindly.') from exc
    device = 'cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.load_in_4bit:
        if device != 'cuda':
            parser.error('--load-in-4bit requires CUDA in this script')
        for package in ('bitsandbytes', 'accelerate'):
            if importlib.util.find_spec(package) is None:
                parser.error(f'{package} is missing; see README for isolated 4-bit setup')
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == 'cuda' else torch.float32
    with Image.open(args.image) as src:
        image = ImageOps.exif_transpose(src).convert('RGB')
    print(f'Model: {MODEL_ID}\nTransformers: {transformers.__version__}\nDevice: {device}; dtype: {dtype}; 4-bit: {args.load_in_4bit}', flush=True)
    print(f'Image: {args.image.resolve()} ({image.width} x {image.height})\nMax pixels: {args.max_pixels}\nIndependent questions; no conversation history.', flush=True)
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=args.local_files_only)
        # Qwen3-VL uses 16-pixel patches with spatial merge size 2 (32-pixel grid).
        # Set both limits explicitly to avoid the model default minimum upscaling small inputs.
        image_options = {'min_pixels': 32*32, 'max_pixels': args.max_pixels}
        kwargs = dict(dtype=dtype, attn_implementation='sdpa', local_files_only=args.local_files_only)
        if args.load_in_4bit:
            kwargs.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
        if not args.load_in_4bit:
            model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        with torch.inference_mode():
            for i, question in enumerate((BOX_PROMPT,) if args.all_objects else QUESTIONS, 1):
                print('\n' + '='*60 + f'\nQ{i}. {question}\n' + '-'*60, flush=True)
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': question if args.all_objects else question + '\n\n' + GUIDANCE},
                ]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors='pt', **image_options)
                inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                          for k, v in inputs.items()}
                length = inputs['input_ids'].shape[1]
                output = model.generate(**inputs, max_new_tokens=args.bbox_max_new_tokens if args.all_objects else args.max_new_tokens,
                                        do_sample=False, num_beams=1, use_cache=True)
                answer = processor.batch_decode(output[:, length:].cpu(), skip_special_tokens=True)[0].strip()
                print(f'A: {answer}\n' + '='*60, flush=True)
                del inputs, output
                if args.all_objects:
                    try:
                        save_boxes(answer, image, args.output)
                    except ValueError as exc:
                        raise SystemExit(str(exc)) from exc
    except torch.cuda.OutOfMemoryError:
        raise SystemExit('CUDA OOM: stop other GPU jobs; retry --max-pixels 65536 --max-new-tokens 128; '
                         'or use --load-in-4bit after installing its optional dependencies in an isolated environment.')


if __name__ == '__main__':
    main()
