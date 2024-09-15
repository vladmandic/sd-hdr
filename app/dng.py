# TODO: does not produce valid DNG files

import json
import cv2
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import Orientation, PreviewColorSpace, PhotometricInterpretation


def write_dng(name_dng, hdr, dct):
    rgb = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
    h, w, _c = rgb.shape
    tags = DNGTags()
    tags.set(Tag.ImageWidth, w) # 256
    tags.set(Tag.ImageLength, h) # 257
    tags.set(Tag.BitsPerSample, 16) # 258
    # tags.set(Tag.Compression, Compression.Uncompressed) # 259 # overriden by raw.convert
    tags.set(Tag.PhotometricInterpretation, PhotometricInterpretation.RGB) # 262
    tags.set(Tag.ImageDescription, json.dumps(dct)) # 270
    tags.set(Tag.Orientation, Orientation.Horizontal) # 274
    tags.set(Tag.SamplesPerPixel, 3) # 277 # should be 3 but somehow 1 works
    # tags.set(Tag.RowsPerStrip, h) # 278 # library sets tiles instead
    # tags.set(Tag.Software, "sdxl-hdr") # 305 # overriden by the library
    # tags.set(Tag.DateTime, datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()) # 306 # should be ISO-8601 but its not recognized
    # tags.set(Tag.DNGVersion, DNGVersion.V1_4) # 50706 # overriden by raw.convert
    tags.set(Tag.UniqueCameraModel, "sdxl-hdr") # 50708
    tags.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB) # 50970
    raw = RAW2DNG()
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
- BitsPerSample=16,16,16    PhotometricInterpretation=RGB               SamplesPerPixel=1 - fail PhotometricInterpretation requires NewSubFileType, ps fails, windows fails
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
