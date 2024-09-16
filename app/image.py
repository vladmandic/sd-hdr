import os
import json
import cv2
from app.logger import log


def save(args, raw, hdr, ldr, dct, ts):
    if args.ldr:
        fn = os.path.join(args.output, f'{ts}-ldr.png')
        try:
            log.info(f'Save: type=ldr format=png file="{fn}"')
            cv2.imwrite(fn, ldr)
        except Exception as e:
            log.error(f'Save: type=ldr format=png file="{fn}" {e}')
    if args.format == 'png' or args.format == 'all':
        fn = os.path.join(args.output, f'{ts}.png')
        try:
            log.info(f'Save: type=hdr format=png file="{fn}"')
            cv2.imwrite(fn, hdr)
        except Exception as e:
            log.error(f'Save: type=hdr format=png file="{fn}" {e}')
    if args.format == 'hdr' or args.format == 'all':
        fn = os.path.join(args.output, f'{ts}.hdr')
        try:
            log.info(f'Save: type=hdr format=hdr file="{fn}"')
            cv2.imwrite(fn, raw)
        except Exception as e:
            log.error(f'Save: type=hdr format=hdr file="{fn}" {e}')
    if args.format == 'tiff' or args.format == 'all':
        fn = os.path.join(args.output, f'{ts}.tiff')
        try:
            log.info(f'Save: type=hdr format=tif file="{fn}"')
            cv2.imwrite(fn, hdr)
        except Exception as e:
            log.error(f'Save: type=hdr format=tif file="{fn}" {e}')
    if args.format == 'exr' or args.format == 'all':
        fn = os.path.join(args.output, f'{ts}.exr')
        try:
            log.warning(f'Save: type=hdr format=exr file="{fn}" exr is broken in current cv2')
            cv2.imwrite(fn, hdr)
        except Exception as e:
            log.error(f'Save: type=hdr format=exr file="{fn}" {e}')
    if args.format == 'dng' or args.format == 'all':
        fn = os.path.join(args.output, f'{ts}.dng')
        try:
            log.warning(f'Save: type=hdr format=dng file="{fn}" dng is not fully compliant')
            write_dng(fn, hdr, dct)
        except Exception as e:
            log.error(f'Save: type=hdr format=dng file="{fn}" {e}')
    if args.json:
        fn = os.path.join(args.output, f'{ts}.json') if args.json else None
        log.info(f'Save: type=json file="{fn}"')
        with open(fn, 'w', encoding='utf8') as f:
            f.write(json.dumps(dct, indent=4))


def write_dng(name_dng: str, hdr, dct = {}):
    from pidng.core import DNGBASE, DNGTags, Tag
    from pidng.defs import Orientation, PreviewColorSpace, PhotometricInterpretation
    rgb = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
    h, w, _c = rgb.shape
    tags = DNGTags() # tags must be in exact order
    tags.set(Tag.ImageWidth, w) # 256
    tags.set(Tag.ImageLength, h) # 257
    tags.set(Tag.BitsPerSample, 16) # 258
    # tags.set(Tag.Compression, Compression.Uncompressed) # 259 # overriden by raw.convert
    tags.set(Tag.PhotometricInterpretation, PhotometricInterpretation.RGB) # 262
    tags.set(Tag.ImageDescription, json.dumps(dct)) # 270
    tags.set(Tag.Orientation, Orientation.Horizontal) # 274
    tags.set(Tag.SamplesPerPixel, 3) # 277
    # tags.set(Tag.RowsPerStrip, h) # 278 # library sets tiles instead
    # tags.set(Tag.Software, "sdxl-hdr") # 305 # overriden by the library
    # tags.set(Tag.DateTime, datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()) # 306 # should be ISO-8601 but its not recognized
    # tags.set(Tag.DNGVersion, DNGVersion.V1_4) # 50706 # overriden by raw.convert
    tags.set(Tag.UniqueCameraModel, "sdxl-hdr") # 50708
    tags.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB) # 50970
    raw = DNGBASE()
    raw.options(tags, path="", compress=False)
    raw.convert(rgb, filename=name_dng)
    """
    lib: https://github.com/schoolpost/PiDNG
    specs: https://paulbourke.net/dataformats/dng/dng_spec_1_6_0_0.pdf
    issue: https://github.com/schoolpost/PiDNG/issues/85

    tested:
    - BitsPerSample=16          PhotometricInterpretation=Linear_Raw        SamplesPerPixel=1 - pass, ps unprocessed monochrome, windows fails
    - BitsPerSample=16          PhotometricInterpretation=Linear_Raw        SamplesPerPixel=3 - fail invalid SamplesPerPixel, ps fails, windows overexposes
    - BitsPerSample=16,16,16    PhotometricInterpretation=Linear_Raw        SamplesPerPixel=3 - fail invalid SamplesPerPixel, ps fails, windows overexposes
    - BitsPerSample=16,16,16    PhotometricInterpretation=Linear_Raw        SamplesPerPixel=1 - fail invalid SamplesPerPixel, ps fails windows fails
    - BitsPerSample=16,16,16    PhotometricInterpretation=RGB               SamplesPerPixel=1 - fail PhotometricInterpretation requires NewSubFileType, ps fails, windows pass
    - BitsPerSample=16          PhotometricInterpretation=RGB               SamplesPerPixel=3 - fail PhotometricInterpretation requires NewSubFileType, ps fails, windows pass

    min DNG tags:
    IFD0:
        0x14A (SubIFDs), Long, 1 value, offset to raw IFD
        0xC612 (DNGVersion), Byte, 4 values, 0x01 0x04 0x00 0x00
    subIFD:
        0xFE (NewSubFileType), Long, 1 value, 0
        0x100 (ImageWidth), Long, 1 value, image width
        0x101 (ImageLength), Long, 1 value, image height
        0x102 (BItDepth), Short, 1 value, 16
        0x103 (Compression), Short, 1 value, 1 (uncompressed)
        0x106 (PhotometricInterpretation), Short, 1 value, 34892 (linear raw)
        0x111 (StripOffsets), Long, 1 value, offset to image bytes
        0x115 (SamplesPerPixel), Short, 1 value, 1 (number of color channels, 1 for monochrome)
        0x116 (RowsPerStrip), Long, 1 value, image height
        0x117 (StripByteCounts), Long, 1 value, number of bytes in the image (width * height * bytes per pixel)
    """


"""
if __name__ == "__main__":
    import os
    import sys
    import subprocess
    from rich import print as rprint
    import cv2
    args = sys.argv
    args.pop(0)
    rprint(f'args: {args}')
    for f in args:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        fn = os.path.splitext(f)[0] + ".dng"
        rprint(f'image: input="{f}" shape={img.shape} dtype={img.dtype} output="{fn}"')
        write_dng(fn, img)
        proc = subprocess.run(['tools/dng_validate.exe', "-v", fn], check=False, capture_output=True, text=True)
        rprint(proc.stdout)
        rprint(proc.stderr)
"""
