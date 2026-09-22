classdef CrossCheckTest < matlab.unittest.TestCase
%CROSSCHECKTEST  The MATLAB pipeline against the Python reference.
%   Two kinds of check. (1) Network for network, on identical inputs saved
%   by Python (tests/reference/*.mat): the imported gate / quality / grader
%   and the natively built U-Net must reproduce Python's outputs to 1e-4,
%   and the closed-form Grad-CAM must reproduce the gradient-tape maps.
%   (2) End to end on the shipped samples (python_samples.json from
%   backend.eval.matlab_reference): the decisions must agree - accepted,
%   referable, review flag, tier, CNN grade, rule grade within one, and
%   P(referable) within 0.05 - the tolerance left by image primitives that
%   differ between the two implementations (resize, bilateral filter).
%   Skipped when the exported models are not present.

    properties
        Models
        Here
    end

    methods (TestClassSetup)
        function load(t)
            t.Here = fileparts(mfilename('fullpath'));
            addpath(fullfile(t.Here, '..'));
            exportDir = fullfile(drscreen.repoRoot(), 'models', 'export');
            t.assumeTrue(isfolder(fullfile(exportDir, 'grader_v2')), 'models/export not present (run backend.eval.export_models)');
            t.Models = drscreen.loadModels(exportDir);
        end
    end

    methods (Test)
        function importedNetworksMatchPython(t)
            r = load(fullfile(t.Here, 'reference', 'nets_npdr.mat'));
            pairs = {'gate', r.gate_in, r.gate_out; 'quality', r.quality_in, r.quality_out; 'grader', r.grader_in, r.grader_out};
            for i = 1:size(pairs, 1)
                y = double(drscreen.predictNet(t.Models.(pairs{i, 1}), pairs{i, 2}));
                t.verifyEqual(y(:), double(pairs{i, 3}(:)), 'AbsTol', 1e-4, sprintf('%s output differs from Python', pairs{i, 1}));
            end
        end

        function nativeUnetMatchesPython(t)
            r = load(fullfile(t.Here, 'reference', 'unet_npdr.mat'));
            probs = double(drscreen.predictNet(t.Models.unet, r.rgb));
            t.verifyEqual(probs, double(r.probs), 'AbsTol', 1e-4);
        end

        function closedFormGradCamMatchesPython(t)
            r = load(fullfile(t.Here, 'reference', 'cam_npdr.mat'));
            A = drscreen.predictNet(t.Models.graderFeatures, r.grader_in);
            for k = 1:4
                w = t.Models.graderHead.dense_kernel(:, k);
                map = max(sum(double(A) .* reshape(w, 1, 1, []), 3), 0); map = map / max(map(:));
                t.verifyEqual(map, double(squeeze(r.cams(k, :, :))), 'AbsTol', 1e-3, sprintf('CAM head %d', k));
            end
        end

        function sampleDecisionsAgreeWithPython(t)
            ref = jsondecode(fileread(fullfile(t.Here, 'reference', 'python_samples.json')));
            names = fieldnames(ref.samples);
            for i = 1:numel(names)
                file = regexprep(names{i}, '_(jpg|jpeg|png)$', '.$1');       % jsondecode mangles the dot
                py = ref.samples.(names{i});
                r = drscreen.screenImage(fullfile(drscreen.repoRoot(), 'samples', file), t.Models, struct('report', false));
                t.verifyEqual(logical(r.accepted), logical(py.accepted), [file ': accepted']);
                t.verifyEqual(r.stage5.tier, py.tier, [file ': tier']);
                if ~py.accepted, continue; end
                t.verifyEqual(logical(r.stage2.fusion.referable), logical(py.referable), [file ': referable']);
                t.verifyEqual(logical(r.stage2.fusion.flagForReview), logical(py.flag), [file ': review flag']);
                t.verifyEqual(r.stage2.cnn.grade, py.cnn_grade, [file ': CNN grade']);
                t.verifyLessThanOrEqual(abs(r.stage2.rule.grade - py.rule_grade), 1, [file ': rule grade']);
                t.verifyEqual(r.stage2.fusion.pReferable, py.p_referable, 'AbsTol', 0.05, [file ': P(referable)']);
            end
        end
    end
end
