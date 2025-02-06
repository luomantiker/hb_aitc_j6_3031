import argparse
import os
import numpy as np

from hbdk4.compiler import Module, load, convert, compile, March
from hbdk4.compiler.hbm import Hbm, Hbo
from hbdk4.compiler.extra_apis import (
    dynamic_quantize_convert,
    dynamic_quantize_convert_per_block,
)


def cosine_dist(vec1, vec2):
    vec1 = vec1.flatten().astype(np.float32)
    vec2 = vec2.flatten().astype(np.float32)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # Handle zero vector case

    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def align(src, dst, prefix):
    inputs = {
        v.name: np.random.rand(*v.type.shape).astype(v.type.np_dtype)
        for v in src[0].inputs
    }
    for v in src[0].inputs:
        print(v.name, inputs[v.name].dtype)
    srcOutputs = src[0].feed(inputs)
    dstOutputs = dst[0].feed(inputs)
    for idx, v in enumerate(src[0].outputs):
        src_data = srcOutputs[v.name]
        dst_data = dstOutputs[v.name]
        print(prefix, v.name, "cosine dist", cosine_dist(src_data, dst_data))


def convertModulesFromFile(args):
    march = args.march
    if not isinstance(march, March):
        march = March.get(march)

    input = args.input_mlir
    quantizeHbmPath = args.out_quantize_hbm
    loadedModule: Module = load(input)
    srcLoadedModule = loadedModule.clone()

    if args.block_size >= 32:
        dqConvertedModule = dynamic_quantize_convert_per_block(
            loadedModule, args.block_size
        )
    else:
        dqConvertedModule = dynamic_quantize_convert(loadedModule)

    # Convert and align bc
    srcConvertedModule = None
    dstConvertedModule: Module = convert(dqConvertedModule, march)
    if args.align == "bc" or args.align == "hbm":
        srcConvertedModule: Module = convert(srcLoadedModule, march)
        align(srcConvertedModule, dstConvertedModule, "Stage bc")

    # Compile to hbm and align
    if args.align == "hbm":
        dstHbm: Hbm | Hbo = compile(dstConvertedModule, quantizeHbmPath, march)
        dirName = os.path.dirname(quantizeHbmPath)
        baseName = os.path.basename(quantizeHbmPath)
        quantizeHbmPathTemp = os.path.join(dirName, "temp_" + baseName)
        srcHbm = compile(srcConvertedModule, quantizeHbmPathTemp, march)
        align(srcHbm, dstHbm, "Stage hbm")


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mlir", type=str, help="mlir or bc file")
    parser.add_argument(
        "out_quantize_hbm", type=str, help="dynamic quantize convert hbm path"
    )
    parser.add_argument(
        "--march", default="nash-m", help="BPU march to compile the model"
    )
    parser.add_argument("--align", default="bc", type=str, help="align bc or hbm")
    parser.add_argument("--block_size", default=64, type=int, help="per block size")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = getArgs()
    convertModulesFromFile(args)
