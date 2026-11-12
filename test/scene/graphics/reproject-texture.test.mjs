import { expect } from 'chai';
import sinon from 'sinon';

import {
    ADDRESS_CLAMP_TO_EDGE,
    PIXELFORMAT_RGBA8,
    TEXTURETYPE_RGBP
} from '../../../src/platform/graphics/constants.js';
import { RenderTarget } from '../../../src/platform/graphics/render-target.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import { EnvLighting } from '../../../src/scene/graphics/env-lighting.js';
import { reprojectTexture } from '../../../src/scene/graphics/reproject-texture.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

/**
 * @import { Application } from '../../../src/framework/application.js'
 * @import { GraphicsDevice } from '../../../src/platform/graphics/graphics-device.js'
 */

describe('reprojectTexture', function () {
    /** @type {Application} */
    let app;

    /** @type {GraphicsDevice} */
    let device;

    beforeEach(function () {
        jsdomSetup();
        app = createApp();
        device = app.graphicsDevice;
    });

    afterEach(function () {
        sinon.restore();
        app?.destroy();
        app = null;
        jsdomTeardown();
    });

    const create2dTexture = (name, size) => new Texture(device, {
        name,
        width: size,
        height: size,
        format: PIXELFORMAT_RGBA8,
        type: TEXTURETYPE_RGBP,
        addressU: ADDRESS_CLAMP_TO_EDGE,
        addressV: ADDRESS_CLAMP_TO_EDGE,
        mipmaps: false
    });

    const createCubemap = (name, size, mipmaps = false) => new Texture(device, {
        name,
        cubemap: true,
        width: size,
        height: size,
        format: PIXELFORMAT_RGBA8,
        type: TEXTURETYPE_RGBP,
        addressU: ADDRESS_CLAMP_TO_EDGE,
        addressV: ADDRESS_CLAMP_TO_EDGE,
        mipmaps
    });

    describe('render target reuse', function () {

        it('does not destroy the render targets between repeated reprojections', function () {
            const source = create2dTexture('source', 16);
            const target = create2dTexture('target', 16);
            const destroy = sinon.spy(RenderTarget.prototype, 'destroy');

            reprojectTexture(source, target, { numSamples: 1 });
            destroy.resetHistory();

            // reusing the render target means nothing is created and destroyed on the next calls
            reprojectTexture(source, target, { numSamples: 1 });
            reprojectTexture(source, target, { numSamples: 1 });

            expect(destroy.called).to.equal(false);

            target.destroy();
            source.destroy();
        });

        it('keeps one render target per cubemap face across calls', function () {
            const source = createCubemap('source', 16);
            const target = createCubemap('target', 16);
            const destroy = sinon.spy(RenderTarget.prototype, 'destroy');

            reprojectTexture(source, target, { numSamples: 1 });

            // the six faces each have a render target, kept for the next call
            const faceTargets = destroy.thisValues.filter(rt => rt.colorBuffer === target);
            expect(faceTargets.length).to.equal(0);

            reprojectTexture(source, target, { numSamples: 1 });
            const faceTargetsAfterSecond = destroy.thisValues.filter(rt => rt.colorBuffer === target);
            expect(faceTargetsAfterSecond.length).to.equal(0);

            target.destroy();
            source.destroy();
        });

        it('destroys the render target of a texture once the texture is destroyed and pruned', function () {
            const source = create2dTexture('source', 16);
            const target = create2dTexture('target', 16);
            const destroy = sinon.spy(RenderTarget.prototype, 'destroy');

            reprojectTexture(source, target, { numSamples: 1 });
            target.destroy();

            // the next reprojection prunes the render target of the destroyed texture
            const other = create2dTexture('other', 16);
            reprojectTexture(source, other, { numSamples: 1 });

            const pruned = destroy.thisValues.filter(rt => rt.colorBuffer === target);
            expect(pruned.length).to.equal(1);

            other.destroy();
            source.destroy();
        });

        it('rebuilds the render target when the target texture is resized in place', function () {
            const source = create2dTexture('source', 16);
            const target = create2dTexture('target', 16);
            const destroy = sinon.spy(RenderTarget.prototype, 'destroy');

            reprojectTexture(source, target, { numSamples: 1 });

            // resizing keeps the texture and its id, but the cached render target becomes stale
            target.resize(32, 32);
            reprojectTexture(source, target, { numSamples: 1 });

            const rebuilt = destroy.thisValues.filter(rt => rt.colorBuffer === target);
            expect(rebuilt.length).to.equal(1);
            expect(rebuilt[0].width).to.equal(16);

            // a further call at the new size reuses the rebuilt render target
            destroy.resetHistory();
            reprojectTexture(source, target, { numSamples: 1 });
            expect(destroy.called).to.equal(false);

            target.destroy();
            source.destroy();
        });
    });

    describe('EnvLighting atlas generation', function () {

        it('allocates the render targets only on the first atlas generation and reuses them', function () {
            const source = createCubemap('lighting-source', 32, true);
            const atlas = create2dTexture('env-atlas', 128);
            const destroy = sinon.spy(RenderTarget.prototype, 'destroy');

            EnvLighting.generateAtlas(source, { target: atlas, size: 128 });
            destroy.resetHistory();

            // regenerating the same atlas repeatedly must not churn render targets
            EnvLighting.generateAtlas(source, { target: atlas, size: 128 });
            EnvLighting.generateAtlas(source, { target: atlas, size: 128 });

            const atlasTargets = destroy.thisValues.filter(rt => rt.colorBuffer === atlas || rt.colorBuffer === source);
            expect(atlasTargets.length).to.equal(0);

            atlas.destroy();
            source.destroy();
        });
    });
});
