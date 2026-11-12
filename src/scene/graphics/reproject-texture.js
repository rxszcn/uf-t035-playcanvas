import { Debug } from '../../core/debug.js';
import { random } from '../../core/math/random.js';
import { Vec3 } from '../../core/math/vec3.js';
import { Vec4 } from '../../core/math/vec4.js';
import {
    FILTER_NEAREST,
    RENDERTARGET_ORIGIN_BOTTOM,
    TEXTUREPROJECTION_OCTAHEDRAL, TEXTUREPROJECTION_CUBE,
    SEMANTIC_POSITION
} from '../../platform/graphics/constants.js';
import { DebugGraphics } from '../../platform/graphics/debug-graphics.js';
import { DeviceCache } from '../../platform/graphics/device-cache.js';
import { RenderTarget } from '../../platform/graphics/render-target.js';
import { Texture } from '../../platform/graphics/texture.js';
import { ChunkUtils } from '../shader-lib/chunk-utils.js';
import { getProgramLibrary } from '../shader-lib/get-program-library.js';
import { ShaderUtils } from '../shader-lib/shader-utils.js';
import { BlendState } from '../../platform/graphics/blend-state.js';
import { RenderPassShaderQuad } from './render-pass-shader-quad.js';

/**
 * @import { Vec4 } from '../../core/math/vec4.js'
 * @import { Shader } from '../../platform/graphics/shader.js'
 */

const getProjectionName = (projection) => {
    switch (projection) {
        case TEXTUREPROJECTION_CUBE:
            return 'Cubemap';
        case TEXTUREPROJECTION_OCTAHEDRAL:
            return 'Octahedral';
        default: // for anything else, assume equirect
            return 'Equirect';
    }
};

// pack a 32bit floating point value into RGBA8
const packFloat32ToRGBA8 = (value, array, offset) => {
    if (value <= 0) {
        array[offset + 0] = 0;
        array[offset + 1] = 0;
        array[offset + 2] = 0;
        array[offset + 3] = 0;
    } else if (value >= 1.0) {
        array[offset + 0] = 255;
        array[offset + 1] = 0;
        array[offset + 2] = 0;
        array[offset + 3] = 0;
    } else {
        let encX = (1 * value) % 1;
        let encY = (255 * value) % 1;
        let encZ = (65025 * value) % 1;
        const encW = (16581375.0 * value) % 1;

        encX -= encY / 255;
        encY -= encZ / 255;
        encZ -= encW / 255;

        array[offset + 0] = Math.min(255, Math.floor(encX * 256));
        array[offset + 1] = Math.min(255, Math.floor(encY * 256));
        array[offset + 2] = Math.min(255, Math.floor(encZ * 256));
        array[offset + 3] = Math.min(255, Math.floor(encW * 256));
    }
};

// pack samples into texture-ready format
const packSamples = (samples) => {
    const numSamples = samples.length;

    const w = Math.min(numSamples, 512);
    const h = Math.ceil(numSamples / w);
    const data = new Uint8Array(w * h * 4);

    // normalize float data and pack into rgba8
    let off = 0;
    for (let i = 0; i < numSamples; i += 4) {
        packFloat32ToRGBA8(samples[i + 0] * 0.5 + 0.5, data, off + 0);
        packFloat32ToRGBA8(samples[i + 1] * 0.5 + 0.5, data, off + 4);
        packFloat32ToRGBA8(samples[i + 2] * 0.5 + 0.5, data, off + 8);
        packFloat32ToRGBA8(samples[i + 3] / 8, data, off + 12);
        off += 16;
    }

    return {
        width: w,
        height: h,
        data: data
    };
};

// generate a vector on the hemisphere with constant distribution.
// function kept because it's useful for debugging
// vec3 hemisphereSampleUniform(vec2 uv) {
//     float phi = uv.y * 2.0 * PI;
//     float cosTheta = 1.0 - uv.x;
//     float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
//     return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
// }

// generate a vector on the hemisphere with phong reflection distribution
const hemisphereSamplePhong = (dstVec, x, y, specularPower) => {
    const phi = y * 2 * Math.PI;
    const cosTheta = Math.pow(1 - x, 1 / (specularPower + 1));
    const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
    dstVec.set(Math.cos(phi) * sinTheta, Math.sin(phi) * sinTheta, cosTheta).normalize();
};

// generate a vector on the hemisphere with lambert distribution
const hemisphereSampleLambert = (dstVec, x, y) => {
    const phi = y * 2 * Math.PI;
    const cosTheta = Math.sqrt(1 - x);
    const sinTheta = Math.sqrt(x);
    dstVec.set(Math.cos(phi) * sinTheta, Math.sin(phi) * sinTheta, cosTheta).normalize();
};

// generate a vector on the hemisphere with GGX distribution.
// a is linear roughness^2
const hemisphereSampleGGX = (dstVec, x, y, a) => {
    const phi = y * 2 * Math.PI;
    const cosTheta = Math.sqrt((1 - x) / (1 + (a * a - 1) * x));
    const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
    dstVec.set(Math.cos(phi) * sinTheta, Math.sin(phi) * sinTheta, cosTheta).normalize();
};

const D_GGX = (NoH, linearRoughness) => {
    const a = NoH * linearRoughness;
    const k = linearRoughness / (1.0 - NoH * NoH + a * a);
    return k * k * (1 / Math.PI);
};

// generate precomputed samples for phong reflections of the given power
const generatePhongSamples = (numSamples, specularPower) => {
    const H = new Vec3();
    const result = [];

    for (let i = 0; i < numSamples; ++i) {
        hemisphereSamplePhong(H, i / numSamples, random.radicalInverse(i), specularPower);
        result.push(H.x, H.y, H.z, 0);
    }

    return result;
};

// generate precomputed samples for lambert convolution
const generateLambertSamples = (numSamples, sourceTotalPixels) => {
    const pixelsPerSample = sourceTotalPixels / numSamples;

    const H = new Vec3();
    const result = [];

    for (let i = 0; i < numSamples; ++i) {
        hemisphereSampleLambert(H, i / numSamples, random.radicalInverse(i));
        const pdf = H.z / Math.PI;
        const mipLevel = 0.5 * Math.log2(pixelsPerSample / pdf);
        result.push(H.x, H.y, H.z, mipLevel);
    }

    return result;
};

// generate a table storing the number of samples required to get 'numSamples'
// valid samples for the given specularPower.
/* eslint-disable no-unused-vars */
const calculateRequiredSamplesGGX = () => {
    const countValidSamplesGGX = (numSamples, specularPower) => {
        const roughness = 1 - Math.log2(specularPower) / 11.0;
        const a = roughness * roughness;
        const H = new Vec3();
        const L = new Vec3();
        const N = new Vec3(0, 0, 1);

        let validSamples = 0;
        for (let i = 0; i < numSamples; ++i) {
            hemisphereSampleGGX(H, i / numSamples, random.radicalInverse(i), a);

            const NoH = H.z;                                    // since N is (0, 0, 1)
            L.set(H.x, H.y, H.z).mulScalar(2 * NoH).sub(N);

            validSamples += L.z > 0 ? 1 : 0;
        }

        return validSamples;
    };

    const numSamples = [1024, 128, 32, 16];
    const specularPowers = [512, 128, 32, 8, 2];

    const requiredTable = {};
    numSamples.forEach((numSamples) => {
        const table = { };
        specularPowers.forEach((specularPower) => {
            let requiredSamples = numSamples;
            while (countValidSamplesGGX(requiredSamples, specularPower) < numSamples) {
                requiredSamples++;
            }
            table[specularPower] = requiredSamples;
        });
        requiredTable[numSamples] = table;
    });

    return requiredTable;
};

// print to the console the required samples table for GGX reflection convolution
// console.log(calculateRequiredSamplesGGX());

// this is a table with pre-calculated number of samples required for GGX.
// the table is generated by calculateRequiredSamplesGGX()
// the table is organized by [numSamples][specularPower]
//
// we use a repeatable pseudo-random sequence of numbers when generating samples
// for use in prefiltering GGX reflections. however not all the random samples
// will be valid. this is because some resulting reflection vectors will be below
// the hemisphere. this is especially apparent when calculating vectors for the
// higher roughnesses. (since vectors are more wild, more of them are invalid).
// for example, specularPower 2 results in half the generated vectors being
// invalid. (meaning the GPU would spend half the time on vectors that don't
// contribute to the final result).
//
// calculating how many samples are required to generate 'n' valid samples is a
// slow operation, so this table stores the pre-calculated numbers of samples
// required for the sets of (numSamples, specularPowers) pairs we expect to
// encounter at runtime.
const requiredSamplesGGX = {
    '16': {
        '2': 26,
        '8': 20,
        '32': 17,
        '128': 16,
        '512': 16
    },
    '32': {
        '2': 53,
        '8': 40,
        '32': 34,
        '128': 32,
        '512': 32
    },
    '128': {
        '2': 214,
        '8': 163,
        '32': 139,
        '128': 130,
        '512': 128
    },
    '1024': {
        '2': 1722,
        '8': 1310,
        '32': 1114,
        '128': 1041,
        '512': 1025
    }
};

// get the number of random samples required to generate numSamples valid samples.
const getRequiredSamplesGGX = (numSamples, specularPower) => {
    const table = requiredSamplesGGX[numSamples];
    return (table && table[specularPower]) || numSamples;
};

// generate precomputed GGX samples
const generateGGXSamples = (numSamples, specularPower, sourceTotalPixels) => {
    const pixelsPerSample = sourceTotalPixels / numSamples;
    const roughness = 1 - Math.log2(specularPower) / 11.0;
    const a = roughness * roughness;
    const H = new Vec3();
    const L = new Vec3();
    const N = new Vec3(0, 0, 1);
    const result = [];

    const requiredSamples = getRequiredSamplesGGX(numSamples, specularPower);

    for (let i = 0; i < requiredSamples; ++i) {
        hemisphereSampleGGX(H, i / requiredSamples, random.radicalInverse(i), a);

        const NoH = H.z;                                    // since N is (0, 0, 1)
        L.set(H.x, H.y, H.z).mulScalar(2 * NoH).sub(N);

        if (L.z > 0) {
            const pdf = D_GGX(Math.min(1, NoH), a) / 4 + 0.001;
            const mipLevel = 0.5 * Math.log2(pixelsPerSample / pdf);
            result.push(L.x, L.y, L.z, mipLevel);
        }
    }

    while (result.length < numSamples * 4) {
        result.push(0, 0, 0, 0);
    }

    return result;
};

// pack float samples data into an rgba8 texture
const createSamplesTex = (device, name, samples) => {
    const packedSamples = packSamples(samples);
    return new Texture(device, {
        name: name,
        width: packedSamples.width,
        height: packedSamples.height,
        mipmaps: false,
        minFilter: FILTER_NEAREST,
        magFilter: FILTER_NEAREST,
        levels: [packedSamples.data]
    });
};

// simple cache storing key->value
// missFunc is called if the key is not present
class SimpleCache {
    constructor(destroyContent = true) {
        this.destroyContent = destroyContent;
    }

    map = new Map();

    destroy() {
        if (this.destroyContent) {
            this.map.forEach((value, key) => {
                value.destroy();
            });
        }
    }

    get(key, missFunc) {
        if (!this.map.has(key)) {
            const result = missFunc();
            this.map.set(key, result);
            return result;
        }
        return this.map.get(key);
    }
}

// cache, used to store samples. we store these separately from textures since multiple
// devices can use the same set of samples.
const samplesCache = new SimpleCache(false);

// cache, storing samples stored in textures, those are per device
const deviceCache = new DeviceCache();

const getCachedTexture = (device, key, getSamplesFnc) => {
    const cache = deviceCache.get(device, () => {
        return new SimpleCache();
    });

    return cache.get(key, () => {
        return createSamplesTex(device, key, samplesCache.get(key, getSamplesFnc));
    });
};

const generateLambertSamplesTex = (device, numSamples, sourceTotalPixels) => {
    const key = `lambert-samples-${numSamples}-${sourceTotalPixels}`;
    return getCachedTexture(device, key, () => {
        return generateLambertSamples(numSamples, sourceTotalPixels);
    });
};

const generatePhongSamplesTex = (device, numSamples, specularPower) => {
    const key = `phong-samples-${numSamples}-${specularPower}`;
    return getCachedTexture(device, key, () => {
        return generatePhongSamples(numSamples, specularPower);
    });
};

const generateGGXSamplesTex = (device, numSamples, specularPower, sourceTotalPixels) => {
    const key = `ggx-samples-${numSamples}-${specularPower}-${sourceTotalPixels}`;
    return getCachedTexture(device, key, () => {
        return generateGGXSamples(numSamples, specularPower, sourceTotalPixels);
    });
};

// Per-device resources reused across reprojectTexture calls. Reprojecting an environment atlas
// runs dozens of these calls, each drawing into one face of a cubemap or a rectangle of a texture.
// Creating a render target and a render pass per call, and destroying them again, allocates and
// releases the framebuffers and the bind groups on every call, which shows as a stutter and a
// climbing resource count while the lighting is generated repeatedly. The render targets keyed by
// their target texture and face, and the render passes keyed by their shader, are created once and
// reused instead. The DeviceCache destroys them when the device is destroyed.
class ReprojectResources {
    /**
     * Render targets keyed by their target texture and, for a cubemap, its face. A render target
     * only wraps a texture and holds no content of its own, so it can be reused on every
     * reprojection into the same texture.
     *
     * @type {Map<string, RenderTarget>}
     * @private
     */
    _renderTargets = new Map();

    /**
     * Quad render passes keyed by their shader. The pass keeps its QuadRender and its bind groups,
     * and is re-initialized to the render target of each draw.
     *
     * @type {Map<Shader, RenderPassShaderQuad>}
     * @private
     */
    _renderPasses = new Map();

    /**
     * Returns a render target rendering to the texture, reusing one created earlier for the same
     * texture and cubemap face.
     *
     * @param {Texture} target - The texture rendered to.
     * @param {number} face - The cubemap face, or 0 for a 2d texture.
     * @param {string} name - The name of the render target.
     * @returns {RenderTarget} The render target.
     */
    getRenderTarget(target, face, name) {
        const key = `${target.id}:${face}`;
        let renderTarget = this._renderTargets.get(key);

        // a texture resized in place keeps its id, but the render target wraps its old, now stale
        // framebuffer and dimensions, so rebuild it for the new size
        if (renderTarget && (renderTarget.width !== target.width || renderTarget.height !== target.height)) {
            renderTarget.destroy();
            this._renderTargets.delete(key);
            renderTarget = null;
        }

        if (!renderTarget) {
            renderTarget = new RenderTarget({
                name,
                colorBuffer: target,
                face,
                depth: false,
                origin: RENDERTARGET_ORIGIN_BOTTOM
            });
            this._renderTargets.set(key, renderTarget);
        }
        return renderTarget;
    }

    /**
     * Returns a render pass for the shader, reusing one created earlier with the same shader, and
     * initializes it to render into the render target.
     *
     * @param {Shader} shader - The shader drawing the quad.
     * @param {RenderTarget} renderTarget - The render target to draw into.
     * @returns {RenderPassShaderQuad} The initialized render pass.
     */
    getRenderPass(shader, renderTarget) {
        let renderPass = this._renderPasses.get(shader);
        if (!renderPass) {
            renderPass = new RenderPassShaderQuad(shader.device);
            renderPass.shader = shader;
            this._renderPasses.set(shader, renderPass);
        }

        // re-initializing switches the attachments and resets their clear and store operations
        renderPass.init(renderTarget);
        return renderPass;
    }

    /**
     * Releases the render targets whose target texture was destroyed elsewhere. The render passes
     * do not reference a texture once re-initialized, but their number is bounded by the shaders,
     * which are cached, so they are kept until the device is destroyed.
     */
    pruneRenderTargets() {
        for (const [key, renderTarget] of this._renderTargets) {
            if (!renderTarget.colorBuffer?.device) {
                renderTarget.destroy();
                this._renderTargets.delete(key);
            }
        }
    }

    destroy() {
        this._renderTargets.forEach(renderTarget => renderTarget.destroy());
        this._renderTargets.clear();

        this._renderPasses.forEach(renderPass => renderPass.destroy());
        this._renderPasses.clear();
    }
}

// per-device reusable render targets and render passes
const reprojectResourcesCache = new DeviceCache();

const getReprojectResources = (device) => {
    return reprojectResourcesCache.get(device, () => new ReprojectResources());
};

// viewport covering the whole render target, reused when no rect is given. It matches the default
// viewport drawQuadWithShader sets, without allocating on each call
const _fullViewport = new Vec4();

/**
 * This function reprojects textures between cubemap, equirectangular and octahedral formats. The
 * function can read and write textures with pixel data in RGBE, RGBM, linear and sRGB formats.
 * When specularPower is specified it will perform a phong-weighted convolution of the source (for
 * generating a gloss maps).
 *
 * @param {Texture} source - The source texture.
 * @param {Texture} target - The target texture.
 * @param {object} [options] - The options object.
 * @param {number} [options.specularPower] - Optional specular power. When specular power is
 * specified, the source is convolved by a phong-weighted kernel raised to the specified power.
 * Otherwise the function performs a standard resample.
 * @param {number} [options.numSamples] - Optional number of samples (default is 1024).
 * @param {number} [options.face] - Optional cubemap face to update (default is update all faces).
 * @param {string} [options.distribution] - Specify convolution distribution - 'none', 'lambert',
 * 'phong', 'ggx'. Default depends on specularPower.
 * @param {Vec4} [options.rect] - Optional viewport rectangle.
 * @param {number} [options.seamPixels] - Optional number of seam pixels to render
 * @returns {boolean} True if the reprojection was applied and false otherwise (e.g. if rect is empty)
 * @category Graphics
 */
function reprojectTexture(source, target, options = {}) {
    Debug.assert(source instanceof Texture && target instanceof Texture, 'source and target must be textures');

    // calculate inner width and height
    const seamPixels = options.seamPixels ?? 0;
    const innerWidth = (options.rect?.z ?? target.width) - seamPixels * 2;
    const innerHeight = (options.rect?.w ?? target.height) - seamPixels * 2;
    if (innerWidth < 1 || innerHeight < 1) {
        // early out if inner space is empty
        return false;
    }

    // table of distribution -> function name
    const funcNames = {
        'none': 'reproject',
        'lambert': 'prefilterSamplesUnweighted',
        'phong': 'prefilterSamplesUnweighted',
        'ggx': 'prefilterSamples'
    };

    // extract options
    const specularPower = options.hasOwnProperty('specularPower') ? options.specularPower : 1;
    const face = options.hasOwnProperty('face') ? options.face : null;
    const distribution = options.hasOwnProperty('distribution') ? options.distribution : (specularPower === 1) ? 'none' : 'phong';

    const processFunc = funcNames[distribution] || 'reproject';
    const prefilterSamples = processFunc.startsWith('prefilterSamples');
    const decodeFunc = ChunkUtils.decodeFunc(source.encoding);
    const encodeFunc = ChunkUtils.encodeFunc(target.encoding);
    const sourceFunc = `sample${getProjectionName(source.projection)}`;
    const targetFunc = `getDirection${getProjectionName(target.projection)}`;
    const numSamples = options.hasOwnProperty('numSamples') ? options.numSamples : 1024;

    // generate unique shader key
    const shaderKey = `ReprojectShader:${processFunc}_${decodeFunc}_${encodeFunc}_${sourceFunc}_${targetFunc}_${numSamples}`;

    const device = source.device;

    let shader = getProgramLibrary(device).getCachedShader(shaderKey);
    if (!shader) {
        const defines = new Map();
        if (prefilterSamples) defines.set('USE_SAMPLES_TEX', '');
        if (source.cubemap) defines.set('CUBEMAP_SOURCE', '');
        defines.set('{PROCESS_FUNC}', processFunc);
        defines.set('{DECODE_FUNC}', decodeFunc);
        defines.set('{ENCODE_FUNC}', encodeFunc);
        defines.set('{SOURCE_FUNC}', sourceFunc);
        defines.set('{TARGET_FUNC}', targetFunc);
        defines.set('{NUM_SAMPLES}', numSamples);
        defines.set('{NUM_SAMPLES_SQRT}', Math.round(Math.sqrt(numSamples)).toFixed(1));

        shader = ShaderUtils.createShader(device, {
            uniqueName: shaderKey,
            attributes: { vertex_position: SEMANTIC_POSITION },
            vertexChunk: 'reprojectVS',
            fragmentChunk: 'reprojectPS',
            fragmentDefines: defines
        });
    }

    DebugGraphics.pushGpuMarker(device, 'ReprojectTexture');

    // render state
    // TODO: set up other render state here to expected state
    device.setBlendState(BlendState.NOBLEND);

    const constantSource = device.scope.resolve(source.cubemap ? 'sourceCube' : 'sourceTex');
    Debug.assert(constantSource);
    constantSource.setValue(source);

    const constantParams = device.scope.resolve('params');

    const uvModParam = device.scope.resolve('uvMod');
    if (seamPixels > 0) {
        uvModParam.setValue([
            (innerWidth + seamPixels * 2) / innerWidth,
            (innerHeight + seamPixels * 2) / innerHeight,
            -seamPixels / innerWidth,
            -seamPixels / innerHeight
        ]);
    } else {
        uvModParam.setValue([1, 1, 0, 0]);
    }

    const params = [
        0,
        target.width * target.height * (target.cubemap ? 6 : 1),
        source.width * source.height * (source.cubemap ? 6 : 1)
    ];

    if (prefilterSamples) {
        // set or generate the pre-calculated samples data
        const sourceTotalPixels = source.width * source.height * (source.cubemap ? 6 : 1);
        const samplesTex =
            (distribution === 'ggx') ? generateGGXSamplesTex(device, numSamples, specularPower, sourceTotalPixels) :
                ((distribution === 'lambert') ? generateLambertSamplesTex(device, numSamples, sourceTotalPixels) :
                    generatePhongSamplesTex(device, numSamples, specularPower));
        device.scope.resolve('samplesTex').setValue(samplesTex);
        device.scope.resolve('samplesTexInverseSize').setValue([1.0 / samplesTex.width, 1.0 / samplesTex.height]);
    }

    // reuse the render targets and the render pass across calls, so generating an environment
    // atlas does not allocate and destroy a framebuffer and bind groups per face and per call
    const resources = getReprojectResources(device);
    resources.pruneRenderTargets();

    const renderTargetName = {
        none: 'Reproject',
        phong: 'PrefilterPhong',
        ggx: 'PrefilterGGX',
        lambert: 'PrefilterLambert'
    }[distribution] ?? 'Reproject';

    for (let f = 0; f < (target.cubemap ? 6 : 1); f++) {
        if (face === null || f === face) {
            const renderTarget = resources.getRenderTarget(target, f, renderTargetName);
            const renderPass = resources.getRenderPass(shader, renderTarget);

            // the quad is drawn with the blend state currently set on the device, as
            // drawQuadWithShader did, and into the viewport rectangle, or the whole target
            renderPass.blendState = device.blendState;
            if (options?.rect) {
                renderPass.viewport = options.rect;
            } else {
                renderPass.viewport = _fullViewport.set(0, 0, renderTarget.width, renderTarget.height);
            }
            renderPass.scissor = undefined;

            params[0] = f;
            constantParams.setValue(params);

            renderPass.render();
        }
    }

    DebugGraphics.popGpuMarker(device);

    return true;
}

export { reprojectTexture };
