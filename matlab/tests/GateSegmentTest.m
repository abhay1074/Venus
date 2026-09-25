classdef GateSegmentTest < matlab.unittest.TestCase
    %GATESEGMENTTEST  Stage 0 (without networks) and Stage 1 (classical) on the shipped samples.

    properties
        healthy
    end

    methods (TestClassSetup)
        function loadSample(t)
            t.healthy = imread(fullfile(drscreen.repoRoot(), 'samples', 'normal_right_eye.jpg'));
        end
    end

    methods (Test)
        function fovIsSquare512AndConvex(t)
            [image, mask, geometry] = drscreen.normaliseFov(t.healthy, 512);
            t.verifyEqual(size(image), [512 512 3]); t.verifyEqual(size(mask), [512 512]);
            t.verifyGreaterThan(geometry.circularity, 0.7);
            t.verifyGreaterThan(nnz(mask), 0.5 * 512 * 512);
            cc = bwconncomp(mask); t.verifyEqual(cc.NumObjects, 1);
        end

        function blurIsRejectedWithReason(t)
            blurred = imgaussfilt(t.healthy, 9);
            [image, mask, geometry] = drscreen.normaliseFov(blurred, 512);
            [label, ~, reason] = drscreen.qualityLabel(drscreen.qualityFeatures(image, mask, geometry));
            t.verifyEqual(label, 'reject'); t.verifyTrue(contains(lower(reason), 'blurry'));
        end

        function goodImageIsGoodAndEnhanceStaysInsideFov(t)
            [image, mask, geometry] = drscreen.normaliseFov(t.healthy, 512);
            label = drscreen.qualityLabel(drscreen.qualityFeatures(image, mask, geometry));
            t.verifyEqual(label, 'good');
            out = drscreen.enhance(image, mask);
            outside = out(repmat(~mask, [1 1 3]));
            t.verifyTrue(all(outside == 0));
        end

        function classicalSegmentationIsPlausibleOnHealthy(t)
            [image, mask] = drscreen.normaliseFov(t.healthy, 512);
            s1 = drscreen.segment(image, mask, struct());
            t.verifyEqual(s1.method, 'classical');
            c = s1.opticDisc.centre; t.verifyTrue(mask(c(2), c(1)));
            t.verifyGreaterThan(s1.opticDisc.radius, 15); t.verifyLessThan(s1.opticDisc.radius, 80);
            dd = 2 * s1.opticDisc.radius;
            dist = norm(s1.fovea.centre - s1.opticDisc.centre);
            t.verifyGreaterThanOrEqual(dist, 1.7 * dd); t.verifyLessThanOrEqual(dist, 3.3 * dd);
            t.verifyGreaterThanOrEqual(s1.vessels.fraction, 0.02); t.verifyLessThanOrEqual(s1.vessels.fraction, 0.15);
            t.verifyEqual(s1.lesions.HE.count, 0); t.verifyLessThanOrEqual(s1.lesions.MA.count, 2);
        end

        function opticDiscIsNotAnImageLabel(t)
            % Same regression cases as backend/tests/test_stages.py: the disc
            % once landed on the white "A" printed in dr_exudates.png, and on a
            % random patch of a dark capture after enhancement flattened it.
            cases = {'dr_exudates.png', [139 222]; 'usable_dark_vignetted.jpg', [420 253]; ...
                     'npdr_hemorrhages_nei.jpg', [40 216]; 'normal_right_eye.jpg', [424 253]};
            for i = 1:size(cases, 1)
                raw = imread(fullfile(drscreen.repoRoot(), 'samples', cases{i, 1}));
                [image, mask] = drscreen.normaliseFov(raw, 512);
                s1 = drscreen.segment(image, mask, struct(), image);
                d = norm(double(s1.opticDisc.centre) - cases{i, 2});
                t.verifyLessThan(d, 1.5 * s1.opticDisc.radius, sprintf('%s: disc at %s', cases{i, 1}, mat2str(s1.opticDisc.centre)));
            end
        end
    end
end
